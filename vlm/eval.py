"""탐지기 평가 — 정답(GT) 대비 정밀도/재현율 (모델 비교·검증).

YOLO 데이터셋의 GT 라벨을 기준으로 탐지기(gemini/yolo/...)를 평가한다.
review.compare_coco(IoU 매칭)를 재사용: 매칭=TP, pred 단독=FP, GT 단독=FN.
이미지 id는 파일 stem(문자열)로 통일해 pred/GT가 같은 이미지에 매칭되게 한다.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Sequence

from .backends.base import image_size, is_image_file
from .detectors.base import DetectorBackend
from .review import compare_coco

_LOCAL = ("mock", "yolo")  # rate-limit 페이싱 불필요한 로컬 백엔드


def load_yolo_gt(images_dir, labels_dir, names: Sequence[str]) -> dict:
    """YOLO 라벨 디렉터리 → GT COCO (image_id = 파일 stem)."""
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(names)]
    images, anns, ann_id = [], [], 1
    for p in sorted(x for x in images_dir.iterdir() if is_image_file(x)):
        w, h = image_size(p)
        w, h = w or 1, h or 1
        images.append({"id": p.stem, "file_name": p.name, "width": w, "height": h})
        lf = labels_dir / (p.stem + ".txt")
        if not lf.exists():
            continue
        for line in lf.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            x, y = (cx - bw / 2) * w, (cy - bh / 2) * h
            anns.append({
                "id": ann_id, "image_id": p.stem, "category_id": cls + 1,
                "bbox": [round(x), round(y), round(bw * w), round(bh * h)],
                "area": round(bw * w * bh * h), "iscrowd": 0,
            })
            ann_id += 1
    return {"images": images, "annotations": anns, "categories": cats}


def eval_detector(
    detector: DetectorBackend,
    images_dir,
    gt_coco: dict,
    concepts: List[str],
    limit: int = 0,
    delay: float = 0.0,
):
    """탐지기를 GT 대비 평가 → ReviewReport(precision_vs_human=정밀도, recall=재현율)."""
    paths = sorted(x for x in Path(images_dir).iterdir() if is_image_file(x))
    if limit > 0:
        paths = paths[:limit]
    cat_id = {c["name"]: c["id"] for c in gt_coco["categories"]}

    pred_anns, ann_id = [], 1
    for i, p in enumerate(paths):
        if delay and i and detector.name not in _LOCAL:
            time.sleep(delay)
        for d in detector.detect(str(p), concepts):
            x0, y0, x1, y1 = d.box
            pred_anns.append({
                "id": ann_id, "image_id": p.stem,
                "category_id": cat_id.get(d.label, 1),
                "bbox": [x0, y0, x1 - x0, y1 - y0], "score": d.confidence,
            })
            ann_id += 1

    eval_ids = {p.stem for p in paths}
    gt_sub = {
        "categories": gt_coco["categories"],
        "images": [im for im in gt_coco["images"] if im["id"] in eval_ids],
        "annotations": [a for a in gt_coco["annotations"] if a["image_id"] in eval_ids],
    }
    pred = {"categories": gt_coco["categories"], "images": gt_sub["images"],
            "annotations": pred_anns}
    # pred=auto, GT=reviewed(truth): matched=TP, removed(pred단독)=FP, added(GT단독)=FN
    return compare_coco(pred, gt_sub, iou_thr=0.5), len(paths)
