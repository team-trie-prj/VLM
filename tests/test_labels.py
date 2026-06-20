"""라벨 변환(④) 테스트 — COCO/YOLO (키/네트워크 불필요)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlm.labels import to_coco, to_yolo  # noqa: E402
from vlm.schemas import Detection, DetectionResult  # noqa: E402


def _result(image_id, w, h, dets):
    return DetectionResult(
        image_id=image_id, image_path=f"{image_id}.jpg", width=w, height=h,
        concepts=["pothole"], detections=dets, backend="mock", model="m",
        created_at="2026-01-01T00:00:00Z",
    )


def test_to_coco_bbox_and_filter():
    r = _result("img1", 100, 100, [
        Detection(label="pothole", box=[10, 20, 40, 60], confidence=0.9),
        Detection(label="crack", box=[0, 0, 10, 10], confidence=0.2),
    ])
    coco = to_coco([r], min_conf=0.5)
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 1  # crack(0.2)는 임계값으로 제외
    a = coco["annotations"][0]
    assert a["bbox"] == [10, 20, 30, 40]  # [x, y, w, h]
    assert a["score"] == 0.9
    assert "pothole" in {c["name"] for c in coco["categories"]}


def test_to_yolo_normalized():
    r = _result("img2", 200, 100, [
        Detection(label="pothole", box=[50, 25, 150, 75], confidence=0.8),
    ])
    out = tempfile.mkdtemp()
    labels_dir, names = to_yolo([r], out, min_conf=0.0)
    txt = (labels_dir / "img2.txt").read_text().strip()
    cls, cx, cy, bw, bh = txt.split()
    assert cls == "0"  # 0-based
    assert abs(float(cx) - 0.5) < 1e-6   # (50+150)/2 / 200
    assert abs(float(cy) - 0.5) < 1e-6   # (25+75)/2 / 100
    assert abs(float(bw) - 0.5) < 1e-6   # 100/200
    assert abs(float(bh) - 0.5) < 1e-6   # 50/100
    assert names == ["pothole"]
    assert (Path(out) / "classes.txt").exists()


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok - {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
