"""커맨드라인 인터페이스.

예)
  python -m vlm analyze data/images/road.jpg
  python -m vlm batch data/images --query "포트홀 찾아줘"
  python -m vlm query "도로 균열이랑 패임 표시해줘"
  python -m vlm vqa data/images/road.jpg "이 도로에 손상이 있나요?"
  python -m vlm info
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

from .config import build_backend, build_detector, load_config
from .pipeline import VLMProcessor


def _make_processor(args) -> VLMProcessor:
    cfg = load_config(args.config)
    if args.backend:
        cfg["backend"] = args.backend
    if args.model:
        cfg["model"] = args.model
    backend = build_backend(cfg)
    return VLMProcessor(backend, output_dir=cfg.get("output_dir", "data/outputs"))


def _print_json(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def cmd_analyze(args) -> int:
    proc = _make_processor(args)
    result = proc.process(args.image, query=args.query)
    _print_json(result.model_dump())
    if not args.no_save:
        out = proc.save_result(result)
        print(f"\n[saved] {out}", file=sys.stderr)
    return 0


def cmd_batch(args) -> int:
    proc = _make_processor(args)
    import time

    t0 = time.perf_counter()
    results = proc.process_dir(args.image_dir, query=args.query, delay=args.delay)
    elapsed = time.perf_counter() - t0

    out = proc.save_dataset(results, name=args.name)
    per = (elapsed / len(results)) if results else 0.0
    print(
        f"처리 {len(results)}장 / {elapsed:.2f}s "
        f"(장당 {per * 1000:.0f}ms, 백엔드={proc.backend.name})"
    )
    tok = sum(r.usage.input_tokens + r.usage.output_tokens for r in results if r.usage)
    cost = sum(
        r.usage.estimated_cost_usd
        for r in results
        if r.usage and r.usage.estimated_cost_usd
    )
    if tok:
        print(f"토큰 합계 {tok:,} / 예상 비용 ${cost:.4f}")
    print(f"[saved] {out}")
    return 0


def cmd_query(args) -> int:
    proc = _make_processor(args)
    concept = proc.parse_query(args.text)
    _print_json(concept.model_dump())
    return 0


def cmd_vqa(args) -> int:
    proc = _make_processor(args)
    print(proc.vqa(args.image, args.question))
    return 0


def cmd_detect(args) -> int:
    import time
    from datetime import datetime, timezone

    from .backends.base import image_size
    from .keywords import keyword_parse_query
    from .overlay import draw_detections
    from .schemas import DetectionResult

    cfg = load_config(args.config)  # .env 로드 + output_dir
    if args.detector:
        cfg["detector"] = args.detector
    if args.model:
        cfg["detector_model"] = args.model

    if args.concepts:
        concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
    elif args.query:
        cp = keyword_parse_query(args.query)
        concepts = cp.concepts or [args.query]
    else:
        concepts = ["pothole", "crack"]

    det = build_detector(cfg)
    out_dir = Path(cfg.get("output_dir", "data/outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = str(out_dir / "masks") if args.masks else None

    w, h = image_size(args.image)
    t0 = time.perf_counter()
    detections = det.detect(args.image, concepts, mask_dir=mask_dir)
    lat = (time.perf_counter() - t0) * 1000.0

    image_id = Path(args.image).stem
    overlay_path = None
    if not args.no_overlay and detections:
        overlay_path = str(out_dir / f"{image_id}_detected.png")
        draw_detections(args.image, detections, overlay_path, with_masks=args.masks)

    result = DetectionResult(
        image_id=image_id, image_path=args.image, width=w, height=h,
        concepts=concepts, detections=detections,
        backend=det.name, model=det.model, latency_ms=round(lat, 1),
        created_at=datetime.now(timezone.utc).isoformat(),
        overlay_path=overlay_path,
    )
    out = out_dir / f"{image_id}_detect.json"
    out.write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"개념: {concepts}  ->  탐지 {len(detections)}건 "
          f"(backend={det.name}, {lat:.0f}ms)")
    for dn in detections:
        print(f"  - {dn.label} {dn.confidence:.2f} box={dn.box}")
    if overlay_path:
        print(f"[overlay] {overlay_path}")
    print(f"[saved] {out}")
    return 0


def cmd_label(args) -> int:
    import time
    from datetime import datetime, timezone

    from .backends.base import image_size, is_image_file
    from .keywords import keyword_parse_query
    from .labels import to_yolo, write_coco
    from .schemas import DetectionResult

    cfg = load_config(args.config)
    if args.detector:
        cfg["detector"] = args.detector
    if args.model:
        cfg["detector_model"] = args.model

    if args.concepts:
        concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
    elif args.query:
        concepts = keyword_parse_query(args.query).concepts or [args.query]
    else:
        concepts = ["pothole", "crack"]

    det = build_detector(cfg)
    paths = sorted(p for p in Path(args.image_dir).iterdir() if is_image_file(p))
    if args.limit > 0:
        paths = paths[: args.limit]

    results = []
    for i, p in enumerate(paths):
        if args.delay and i and det.name != "mock":
            time.sleep(args.delay)
        try:
            w, h = image_size(p)
            t0 = time.perf_counter()
            dets = det.detect(str(p), concepts)
            lat = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            print(f"[warn] {p.name} 탐지 실패: {e}")
            continue
        results.append(DetectionResult(
            image_id=p.stem, image_path=str(p), width=w, height=h,
            concepts=concepts, detections=dets, backend=det.name, model=det.model,
            latency_ms=round(lat, 1),
            created_at=datetime.now(timezone.utc).isoformat(),
        ))

    out_dir = Path(cfg.get("output_dir", "data/outputs")) / "labels_export"
    out_dir.mkdir(parents=True, exist_ok=True)
    total = sum(
        len([d for d in r.detections if d.confidence >= args.min_conf])
        for r in results
    )
    if args.format in ("coco", "both"):
        cp = write_coco(results, out_dir / "annotations_coco.json", min_conf=args.min_conf)
        print(f"[COCO] {cp}")
    if args.format in ("yolo", "both"):
        ld, names = to_yolo(results, out_dir, min_conf=args.min_conf)
        print(f"[YOLO] {ld} (classes={names})")
    print(f"이미지 {len(results)}장 / 라벨 {total}건 "
          f"(min_conf={args.min_conf}, backend={det.name})")
    return 0


def cmd_compare(args) -> int:
    from .compare import compare, format_report

    cfg = load_config(args.config)  # .env 로드 + output_dir
    names = [s.strip() for s in args.backends.split(",") if s.strip()]
    report = compare(args.image_dir, names, delay=args.delay, limit=args.limit)

    out_dir = Path(cfg.get("output_dir", "data/outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.name}.json"
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(format_report(report))
    print(f"\n[saved] {out}")
    return 0


def cmd_review(args) -> int:
    from .review import compare_coco, triage

    cfg = load_config(args.config)
    out_dir = Path(cfg.get("output_dir", "data/outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "prep":
        if not args.coco:
            print("[실패] --coco <auto COCO 경로> 가 필요합니다.")
            return 1
        coco = json.loads(Path(args.coco).read_text(encoding="utf-8"))
        man = triage(coco, args.confirm_thr)
        p = out_dir / "review_manifest.json"
        p.write_text(json.dumps(man.model_dump(), ensure_ascii=False, indent=2),
                     encoding="utf-8")
        print(f"트리아지: 총 {man.total} / 자동확정 {man.auto_confirmed} / "
              f"검수대상 {man.needs_review}")
        print(f"자동확정률 {man.auto_confirm_rate * 100:.1f}% (임계값 {args.confirm_thr})")
        print(f"[saved] {p}")
        return 0

    # report
    if not (args.auto and args.reviewed):
        print("[실패] --auto 와 --reviewed COCO 경로가 필요합니다.")
        return 1
    auto = json.loads(Path(args.auto).read_text(encoding="utf-8"))
    rev = json.loads(Path(args.reviewed).read_text(encoding="utf-8"))
    rep = compare_coco(auto, rev, args.iou)
    p = out_dir / "review_report.json"
    p.write_text(json.dumps(rep.model_dump(), ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print(f"auto {rep.auto_count} / reviewed {rep.reviewed_count} / 매칭 {rep.matched}")
    print(f"검수자 삭제(오탐) {rep.removed_by_reviewer} / 추가(미탐) {rep.added_by_reviewer}")
    print(f"수정 비율 {rep.correction_rate * 100:.1f}% "
          f"(precision {rep.precision_vs_human * 100:.0f}% / "
          f"recall {rep.recall_vs_human * 100:.0f}%)")
    print(f"[saved] {p}")
    return 0


def cmd_eval(args) -> int:
    from .eval import eval_detector, load_yolo_gt

    cfg = load_config(args.config)
    if args.detector:
        cfg["detector"] = args.detector
    images_dir = Path(args.dataset) / "images"
    labels_dir = Path(args.dataset) / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        print(f"[실패] {args.dataset} 아래 images/ 와 labels/ 가 필요합니다.")
        return 1
    names = [s.strip() for s in args.names.split(",") if s.strip()]
    concepts = [s.strip() for s in (args.concepts or args.names).split(",") if s.strip()]

    gt = load_yolo_gt(images_dir, labels_dir, names)
    det = build_detector(cfg)
    rep, n = eval_detector(det, images_dir, gt, concepts,
                           limit=args.limit, delay=args.delay)
    tp, fp, fn = rep.matched, rep.removed_by_reviewer, rep.added_by_reviewer
    print(f"평가: {n}장, detector={det.name} (model={det.model}), IoU>=0.5")
    print(f"TP {tp} / FP {fp} / FN {fn}")
    print(f"precision {rep.precision_vs_human * 100:.1f}% / "
          f"recall {rep.recall_vs_human * 100:.1f}%")

    out = Path(cfg.get("output_dir", "data/outputs")) / f"eval_{det.name}.json"
    out.write_text(json.dumps(
        {"detector": det.name, "model": det.model, "images": n, **rep.model_dump()},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out}")
    return 0


def cmd_demo(args) -> int:
    load_config(args.config)  # .env 로드 (GEMINI_API_KEY 등이 데모에서 잡히도록)
    from .demo import build_demo

    build_demo().launch(
        server_name=args.host, server_port=args.port, share=args.share
    )
    return 0


def cmd_web(args) -> int:
    from .web import main

    return main(host=args.host, port=args.port)


def cmd_doctor(args) -> int:
    """환경 진단: 패키지·키·백엔드 확인. --ping 시 실제 API 호출로 키 검증."""
    cfg = load_config(args.config)  # .env 로드 포함
    if args.backend:
        cfg["backend"] = args.backend
    if args.model:
        cfg["model"] = args.model

    print("== 패키지 ==")
    for mod, label in [
        ("pydantic", "core"),
        ("PIL", "core"),
        ("yaml", "core"),
        ("dotenv", "optional"),
        ("google.genai", "gemini 백엔드"),
        ("anthropic", "anthropic 백엔드"),
        ("openai", "openai 백엔드"),
        ("torch", "qwen 백엔드"),
        ("transformers", "qwen 백엔드"),
    ]:
        try:
            importlib.import_module(mod)
            print(f"  [OK]  {mod:<13} ({label})")
        except Exception:
            print(f"  [--]  {mod:<13} ({label}) 미설치")

    print("== 환경변수 키 ==")
    for env in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"]:
        v = os.getenv(env)
        print(f"  {env}: {'설정됨 (' + v[:7] + '...)' if v else '없음'}")

    print(f"== 선택 백엔드: {cfg['backend']} (model={cfg.get('model') or '기본값'}) ==")
    try:
        backend = build_backend(cfg)
    except Exception as e:
        print(f"  [실패] 백엔드 초기화: {e}")
        return 1
    print(f"  [OK] 초기화: name={backend.name}, model={backend.model}")

    if args.ping:
        print("  핑(실제 API 호출) 중...")
        try:
            print("  " + backend.health_check())
        except Exception as e:
            print(f"  [실패] {e}")
            return 1
    return 0


def cmd_info(args) -> int:
    cfg = load_config(args.config)
    if args.backend:
        cfg["backend"] = args.backend
    print("현재 설정:")
    _print_json(cfg)
    try:
        backend = build_backend(cfg)
        print(f"\n백엔드 준비 완료: name={backend.name}, model={backend.model}")
    except Exception as e:
        print(f"\n[경고] 백엔드 초기화 실패: {e}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vlm", description="멀티모달 VLM 이미지 이해 모듈"
    )
    p.add_argument("--config", help="설정 YAML 경로 (기본: config/default.yaml)")
    p.add_argument("--backend", help="백엔드 강제 지정 (mock/gemini/anthropic/openai/qwen)")
    p.add_argument("--model", help="모델 강제 지정")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="이미지 1장 분석 -> 메타데이터 JSON")
    a.add_argument("image", help="이미지 경로")
    a.add_argument("--query", help="(선택) 자연어 질의 -> 개념 프롬프트 동시 생성")
    a.add_argument("--no-save", action="store_true", help="JSON 저장 안 함")
    a.set_defaults(func=cmd_analyze)

    b = sub.add_parser("batch", help="디렉터리 내 이미지 일괄 분석 + 시간 측정")
    b.add_argument("image_dir", help="이미지 디렉터리")
    b.add_argument("--query", help="(선택) 공통 자연어 질의")
    b.add_argument("--name", default="dataset", help="출력 데이터셋 파일명")
    b.add_argument("--delay", type=float, default=0.0,
                   help="이미지 간 대기(초). Gemini 무료 티어(5 RPM)는 13 권장")
    b.set_defaults(func=cmd_batch)

    q = sub.add_parser("query", help="자연어 질의 -> 탐지 개념 프롬프트")
    q.add_argument("text", help="질의문 (예: '포트홀 찾아줘')")
    q.set_defaults(func=cmd_query)

    dt = sub.add_parser("detect", help="탐지+분할(③): 질의/개념 -> bbox + 오버레이")
    dt.add_argument("image", help="이미지 경로")
    dt.add_argument("query", nargs="?", help="자연어 질의 (예: '포트홀 찾아줘')")
    dt.add_argument("--concepts", help="개념 직접 지정 (쉼표구분, 질의보다 우선)")
    dt.add_argument("--detector", help="탐지 백엔드 (mock/gemini)")
    dt.add_argument("--masks", action="store_true", help="분할 mask도 생성(gemini)")
    dt.add_argument("--no-overlay", action="store_true", help="오버레이 이미지 저장 안 함")
    dt.set_defaults(func=cmd_detect)

    lb = sub.add_parser("label", help="④ 자동 라벨 생성: 디렉터리 탐지 -> COCO/YOLO")
    lb.add_argument("image_dir", help="이미지 디렉터리")
    lb.add_argument("query", nargs="?", help="자연어 질의 (예: '포트홀 찾아줘')")
    lb.add_argument("--concepts", help="개념 직접 지정 (쉼표구분)")
    lb.add_argument("--detector", help="탐지 백엔드 (mock/gemini)")
    lb.add_argument("--format", choices=["coco", "yolo", "both"], default="both")
    lb.add_argument("--min-conf", type=float, default=0.0, help="confidence 임계값(필터)")
    lb.add_argument("--limit", type=int, default=0, help="처리 이미지 수 제한 (0=전체)")
    lb.add_argument("--delay", type=float, default=0.0,
                    help="이미지 간 대기(초). 무료 Gemini는 13 권장")
    lb.set_defaults(func=cmd_label)

    rv = sub.add_parser("review", help="⑤ 검수: prep(자동확정률) / report(수정 비율)")
    rv.add_argument("mode", choices=["prep", "report"])
    rv.add_argument("--coco", help="prep: auto COCO 경로")
    rv.add_argument("--confirm-thr", type=float, default=0.5, help="prep: 자동확정 임계값")
    rv.add_argument("--auto", help="report: auto COCO 경로")
    rv.add_argument("--reviewed", help="report: 사람-수정 COCO 경로")
    rv.add_argument("--iou", type=float, default=0.5, help="report: 매칭 IoU 임계값")
    rv.set_defaults(func=cmd_review)

    ev = sub.add_parser("eval", help="탐지기를 GT(YOLO 라벨) 대비 평가 (precision/recall)")
    ev.add_argument("dataset", help="images/ 와 labels/ 가 있는 split 디렉터리")
    ev.add_argument("--names", default="pothole", help="클래스명 (쉼표, index순)")
    ev.add_argument("--concepts", help="탐지 개념 (기본=names)")
    ev.add_argument("--detector", help="mock/gemini/yolo")
    ev.add_argument("--limit", type=int, default=0, help="평가 이미지 수 제한 (0=전체)")
    ev.add_argument("--delay", type=float, default=0.0, help="API 백엔드 페이싱(무료 Gemini 13)")
    ev.set_defaults(func=cmd_eval)

    cp = sub.add_parser("compare", help="여러 백엔드 비교 (정량 비교표 + JSON)")
    cp.add_argument("image_dir", help="이미지 디렉터리")
    cp.add_argument("--backends", default="mock,gemini", help="쉼표구분 (예: mock,gemini)")
    cp.add_argument("--delay", type=float, default=0.0,
                    help="비-mock 백엔드 호출 간 대기(초). 무료 Gemini는 13 권장")
    cp.add_argument("--limit", type=int, default=0, help="처리 이미지 수 제한 (0=전체)")
    cp.add_argument("--name", default="comparison", help="출력 파일명")
    cp.set_defaults(func=cmd_compare)

    v = sub.add_parser("vqa", help="이미지 질의응답")
    v.add_argument("image", help="이미지 경로")
    v.add_argument("question", help="질문")
    v.set_defaults(func=cmd_vqa)

    dm = sub.add_parser("demo", help="Gradio 데모 UI 실행 (업로드->질의->탐지->오버레이)")
    dm.add_argument("--host", default="127.0.0.1")
    dm.add_argument("--port", type=int, default=7860)
    dm.add_argument("--share", action="store_true", help="공개 공유 링크 생성")
    dm.set_defaults(func=cmd_demo)

    wb = sub.add_parser("web", help="커스텀 웹 데모(FastAPI) 실행 - 디자인 구현")
    wb.add_argument("--host", default="127.0.0.1")
    wb.add_argument("--port", type=int, default=8000)
    wb.set_defaults(func=cmd_web)

    i = sub.add_parser("info", help="현재 설정/백엔드 확인")
    i.set_defaults(func=cmd_info)

    d = sub.add_parser("doctor", help="환경 진단(패키지/키/백엔드). --ping 으로 키 검증")
    d.add_argument("--ping", action="store_true", help="실제 API를 호출해 키/연결 검증")
    d.set_defaults(func=cmd_doctor)

    return p


def main(argv=None) -> int:
    # 레거시 Windows 콘솔(cp949 등)에서 비-인코딩 문자가 print될 때 크래시하지 않도록
    # 대체문자 처리로 전환(인코딩 자체는 유지 → 한글은 그대로 렌더링).
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(errors="replace")
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
