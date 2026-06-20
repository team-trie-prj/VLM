"""탐지(③) 테스트 — mock 탐지기 + 오버레이 (키/네트워크 불필요)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from vlm.config import build_detector  # noqa: E402
from vlm.detectors.mock import MockDetector  # noqa: E402
from vlm.overlay import draw_detections  # noqa: E402
from vlm.schemas import Detection  # noqa: E402


def _tmp_image(w=640, h=480) -> str:
    d = tempfile.mkdtemp()
    p = Path(d) / "road_pothole.png"
    Image.new("RGB", (w, h), (90, 90, 95)).save(p)
    return str(p)


def test_mock_detector_boxes_within_image():
    img = _tmp_image(640, 480)
    dets = MockDetector().detect(img, ["pothole"])
    assert len(dets) >= 1
    for d in dets:
        assert isinstance(d, Detection)
        x0, y0, x1, y1 = d.box
        assert 0 <= x0 < x1 <= 640
        assert 0 <= y0 < y1 <= 480
        assert d.label == "pothole"


def test_build_detector_mock():
    det = build_detector({"detector": "mock"})
    assert det.name == "mock"


def test_yolo_detector_missing_weights():
    # 가중치 없으면 친절한 FileNotFoundError (Colab에서 받아오라는 안내)
    try:
        build_detector({"detector": "yolo", "yolo_weights": "models/__none__.pt"})
        assert False, "should raise"
    except FileNotFoundError as e:
        assert "models/__none__.pt" in str(e)


def test_overlay_creates_file():
    img = _tmp_image()
    dets = MockDetector().detect(img, ["crack"])
    out = str(Path(tempfile.mkdtemp()) / "out.png")
    draw_detections(img, dets, out)
    assert Path(out).exists()
    with Image.open(out) as im:
        assert im.size == (640, 480)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok - {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
