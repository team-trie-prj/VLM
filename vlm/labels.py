"""탐지 결과 → COCO / YOLO 라벨 변환 (파이프라인 ④).

confidence 임계값으로 필터링하며, 검수 도구(CVAT/Label Studio)로 넘길
표준 라벨셋을 생성한다. (분할 mask의 polygon/RLE 인코딩은 후속 증분)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .schemas import DetectionResult


def _categories(
    results: List[DetectionResult], explicit: Optional[List[str]] = None
) -> Tuple[List[dict], Dict[str, int]]:
    names = explicit or sorted({d.label for r in results for d in r.detections})
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(names)]
    return cats, {n: i + 1 for i, n in enumerate(names)}


def to_coco(
    results: List[DetectionResult],
    min_conf: float = 0.0,
    categories: Optional[List[str]] = None,
) -> dict:
    """COCO detection 포맷(dict). bbox=[x,y,w,h], score 포함."""
    cats, cat_id = _categories(results, categories)
    images, anns, ann_id = [], [], 1
    for img_id, r in enumerate(results, 1):
        images.append({
            "id": img_id, "file_name": Path(r.image_path).name,
            "width": r.width, "height": r.height,
        })
        for d in r.detections:
            if d.confidence < min_conf:
                continue
            x0, y0, x1, y1 = d.box
            anns.append({
                "id": ann_id, "image_id": img_id,
                "category_id": cat_id.get(d.label, 1),
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "area": (x1 - x0) * (y1 - y0),
                "iscrowd": 0, "score": d.confidence,
            })
            ann_id += 1
    return {"images": images, "annotations": anns, "categories": cats}


def to_yolo(
    results: List[DetectionResult],
    out_dir: str,
    min_conf: float = 0.0,
    categories: Optional[List[str]] = None,
) -> Tuple[Path, List[str]]:
    """YOLO 포맷: labels/<id>.txt (class cx cy w h, 정규화) + classes.txt."""
    cats, cat_id = _categories(results, categories)
    labels_dir = Path(out_dir) / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        w, h = (r.width or 1), (r.height or 1)
        lines = []
        for d in r.detections:
            if d.confidence < min_conf:
                continue
            x0, y0, x1, y1 = d.box
            cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
            bw, bh = (x1 - x0) / w, (y1 - y0) / h
            cls = cat_id.get(d.label, 1) - 1  # YOLO는 0-based
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (labels_dir / f"{r.image_id}.txt").write_text("\n".join(lines), encoding="utf-8")
    names = [c["name"] for c in cats]
    (Path(out_dir) / "classes.txt").write_text("\n".join(names), encoding="utf-8")
    return labels_dir, names


def write_coco(results, out_path, min_conf=0.0, categories=None) -> Path:
    data = to_coco(results, min_conf=min_conf, categories=categories)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return Path(out_path)
