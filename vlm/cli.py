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

from .config import build_backend, load_config
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

    v = sub.add_parser("vqa", help="이미지 질의응답")
    v.add_argument("image", help="이미지 경로")
    v.add_argument("question", help="질문")
    v.set_defaults(func=cmd_vqa)

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
