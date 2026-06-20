"""검수(⑤) 테스트 — 트리아지 / IoU / 수정 비교 (오프라인)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlm.review import _iou, compare_coco, triage  # noqa: E402


def _coco(anns):
    return {
        "images": [{"id": 1}],
        "annotations": anns,
        "categories": [{"id": 1, "name": "pothole"}],
    }


def test_triage_auto_confirm_rate():
    coco = _coco([
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"id": 2, "image_id": 1, "category_id": 1, "bbox": [5, 5, 10, 10], "score": 0.2},
    ])
    m = triage(coco, 0.5)
    assert m.total == 2 and m.auto_confirmed == 1 and m.needs_review == 1
    assert abs(m.auto_confirm_rate - 0.5) < 1e-6
    assert m.items[0].status == "confirmed"
    assert m.items[1].status == "pending"


def test_iou():
    assert _iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert _iou([0, 0, 10, 10], [100, 100, 10, 10]) == 0.0


def test_compare_coco_corrections():
    auto = _coco([
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"id": 2, "image_id": 1, "category_id": 1, "bbox": [50, 50, 10, 10], "score": 0.4},
    ])
    reviewed = _coco([
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 10, 10]},   # box1 유지
        {"id": 3, "image_id": 1, "category_id": 1, "bbox": [80, 80, 12, 12]},  # 새로 추가
    ])
    rep = compare_coco(auto, reviewed, 0.5)
    assert rep.matched == 1
    assert rep.removed_by_reviewer == 1  # box2 삭제됨
    assert rep.added_by_reviewer == 1    # 새 박스 추가됨
    assert abs(rep.correction_rate - 1.0) < 1e-6  # (1+1)/2


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok - {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
