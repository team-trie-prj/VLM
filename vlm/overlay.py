"""탐지 결과를 이미지에 그려 저장 (데모용 오버레이)."""

from __future__ import annotations

from pathlib import Path
from typing import List

from .schemas import Detection

_COLOR = (255, 45, 45)


def draw_detections(
    image_path: str,
    detections: List[Detection],
    out_path: str,
    with_masks: bool = False,
) -> str:
    from PIL import Image, ImageDraw

    im = Image.open(image_path).convert("RGB")

    # 분할 마스크를 반투명 채색으로 합성
    if with_masks:
        tint = Image.new("RGB", im.size, _COLOR)
        for det in detections:
            if det.mask_path and Path(det.mask_path).exists():
                m = Image.open(det.mask_path).convert("L")
                if m.size != im.size:
                    m = m.resize(im.size)
                im = Image.composite(Image.blend(im, tint, 0.45), im, m)

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
