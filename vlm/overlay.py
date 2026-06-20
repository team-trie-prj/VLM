"""탐지 결과를 이미지에 그려 저장 (데모용 오버레이)."""

from __future__ import annotations

from pathlib import Path
from typing import List

from .schemas import Detection

_COLOR = (255, 45, 45)


def draw_detections(image_path: str, detections: List[Detection], out_path: str) -> str:
    from PIL import Image, ImageDraw

    im = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(im)
    for det in detections:
        x0, y0, x1, y1 = det.box
        d.rectangle([x0, y0, x1, y1], outline=_COLOR, width=3)
        cap = f"{det.label} {det.confidence:.2f}"
        ty = y0 - 12 if y0 >= 12 else y0 + 2
        d.text((x0 + 3, ty), cap, fill=_COLOR)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)
    return out_path
