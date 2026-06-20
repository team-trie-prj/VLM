"""eval(탐지기 GT 평가) 테스트 — mock 탐지기 (오프라인)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from vlm.detectors.mock import MockDetector  # noqa: E402
from vlm.eval import eval_detector, load_yolo_gt  # noqa: E402
from vlm.schemas import ReviewReport  # noqa: E402


def _dataset(n=2):
    d = Path(tempfile.mkdtemp())
    (d / "images").mkdir()
    (d / "labels").mkdir()
    for i in range(n):
        Image.new("RGB", (320, 240), (90, 90, 95)).save(d / "images" / f"img{i}.png")
        # 중앙 박스 GT
        (d / "labels" / f"img{i}.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    return d


def test_load_yolo_gt():
    d = _dataset(2)
    gt = load_yolo_gt(d / "images", d / "labels", ["pothole"])
    assert len(gt["images"]) == 2
    assert len(gt["annotations"]) == 2
    a = gt["annotations"][0]
    assert a["category_id"] == 1
    assert a["image_id"] == "img0"
    # 중앙 0.5,0.5 size 0.3,0.3 → bbox [x,y,w,h] 정상 범위
    x, y, w, h = a["bbox"]
    assert 0 <= x and 0 <= y and w > 0 and h > 0


def test_eval_detector_mock():
    d = _dataset(2)
    gt = load_yolo_gt(d / "images", d / "labels", ["pothole"])
    rep, n = eval_detector(MockDetector(), d / "images", gt, ["pothole"])
    assert isinstance(rep, ReviewReport)
    assert n == 2
    assert rep.reviewed_count == 2  # GT 2건
    # 카운트 일관성: matched + removed = pred(auto), matched + added = GT(reviewed)
    assert rep.matched + rep.removed_by_reviewer == rep.auto_count
    assert rep.matched + rep.added_by_reviewer == rep.reviewed_count


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok - {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
