"""테스트용 합성 도로 이미지 생성기.

실제 데이터(AI Hub / RDD2022 등)를 받기 전에 파이프라인을 돌려보기 위한
간단한 더미 이미지. 파일명에 손상 종류 힌트를 넣어 Mock 백엔드 출력이 다양해진다.

  python scripts/make_sample_images.py
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "data" / "images"


def _road(draw: ImageDraw.ImageDraw, w: int, h: int) -> None:
    draw.rectangle([0, 0, w, h], fill=(90, 90, 95))  # 아스팔트
    # 중앙 차선
    for y in range(0, h, 60):
        draw.rectangle([w // 2 - 6, y, w // 2 + 6, y + 30], fill=(220, 210, 120))


def _pothole(draw, w, h):
    cx, cy = w // 2 + 80, h // 2 + 40
    draw.ellipse([cx - 40, cy - 26, cx + 40, cy + 26], fill=(20, 20, 22))


def _crack(draw, w, h):
    x, y = 60, 40
    pts = [(x, y)]
    for _ in range(14):
        x += random.randint(10, 40)
        y += random.randint(-20, 40)
        pts.append((x, y))
    draw.line(pts, fill=(25, 25, 28), width=4)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    specs = [
        ("road_pothole_01.png", _pothole),
        ("road_crack_02.png", _crack),
        ("road_clean_03.png", None),
        ("road_pothole_crack_04.png", "both"),
    ]
    for name, kind in specs:
        img = Image.new("RGB", (640, 400))
        d = ImageDraw.Draw(img)
        _road(d, 640, 400)
        if kind == "both":
            _pothole(d, 640, 400)
            _crack(d, 640, 400)
        elif callable(kind):
            kind(d, 640, 400)
        img.save(OUT / name)
        print(f"saved {OUT / name}")


if __name__ == "__main__":
    main()
