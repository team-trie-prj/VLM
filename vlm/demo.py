"""Gradio 데모 UI — 제안서 앵커 시나리오 "포트홀 영역을 찾아줘".

업로드 → 자연어 질의 → (yolo/gemini/mock) 탐지 → 박스 오버레이 + 탐지표 + COCO 다운로드.
탐지 백엔드는 DetectorBackend 추상화를 그대로 재사용한다.

실행:  python -m vlm demo
"""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .backends.base import image_size
from .config import build_detector
from .keywords import keyword_parse_query
from .labels import to_coco
from .overlay import draw_detections
from .schemas import DetectionResult

_INTRO = (
    "# 도로 손상 탐지 데모\n"
    "도로 이미지를 올리고 자연어로 질의하면 포트홀/균열을 탐지해 표시합니다. "
    "(yolo = 파인튜닝 모델·로컬·무료 / gemini = 무료 API·키 필요 / mock = 키 없이 테스트)"
)


def _run(image_path, query, detector_name, conf=0.25):
    if not image_path:
        return None, [], "이미지를 올려주세요.", None
    try:
        conf = float(conf)
        concepts = ["pothole"]
        if query and query.strip():
            cp = keyword_parse_query(query)
            concepts = cp.concepts or [query.strip()]
        cfg = {"detector": detector_name}
        if detector_name == "yolo":
            cfg["yolo_conf"] = conf  # yolo는 모델 단계에서 임계값 적용
        det = build_detector(cfg)
        dets = [d for d in det.detect(image_path, concepts) if d.confidence >= conf]

        overlay = tempfile.mktemp(suffix=".png")
        if dets:
            draw_detections(image_path, dets, overlay)
        else:
            shutil.copy(image_path, overlay)

        w, h = image_size(image_path)
        result = DetectionResult(
            image_id=Path(image_path).stem, image_path=image_path, width=w, height=h,
            concepts=concepts, detections=dets, backend=det.name, model=det.model,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        coco_path = Path(tempfile.mkdtemp()) / f"{result.image_id}_coco.json"
        coco_path.write_text(
            json.dumps(to_coco([result]), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rows = [[d.label, round(d.confidence, 2), str(d.box)] for d in dets]
        summary = (f"개념 {concepts} · {det.name}({det.model}) · 탐지 {len(dets)}건")
        return overlay, rows, summary, str(coco_path)
    except Exception as e:
        return image_path, [], f"[오류] {e}", None


def build_demo():
    import gradio as gr

    with gr.Blocks(title="VLM 도로손상 탐지 데모") as demo:
        gr.Markdown(_INTRO)
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="filepath", label="도로 이미지")
                query = gr.Textbox(value="포트홀 찾아줘", label="질의")
                detector = gr.Radio(["yolo", "gemini"], value="yolo",
                                    label="탐지 모델 (yolo=파인튜닝·로컬, gemini=무료 API)")
                conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05,
                                 label="신뢰도 임계값 (낮출수록 더 많이 탐지)")
                btn = gr.Button("탐지", variant="primary")
            with gr.Column():
                out_img = gr.Image(label="결과 (박스 오버레이)")
                summary = gr.Textbox(label="요약", interactive=False)
                table = gr.Dataframe(headers=["label", "confidence", "box"],
                                     label="탐지 결과", interactive=False)
                coco = gr.File(label="COCO 라벨 다운로드")
        btn.click(_run, [img, query, detector, conf], [out_img, table, summary, coco])
    return demo


def main():
    build_demo().launch()


if __name__ == "__main__":
    main()
