"""데모 _run 로직 테스트 — mock 탐지기 (gradio/키/네트워크 불필요)."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from vlm.demo import _run  # noqa: E402  (모듈 로드 시 gradio import 안 함)


def _img():
    p = Path(tempfile.mkdtemp()) / "road_pothole.png"
    Image.new("RGB", (320, 240), (90, 90, 95)).save(p)
    return str(p)


def test_run_mock_returns_overlay_and_coco():
    overlay, rows, summary, coco = _run(_img(), "포트홀 찾아줘", "mock")
    assert overlay and os.path.exists(overlay)
    assert coco and os.path.exists(coco)
    assert "mock" in summary
    assert isinstance(rows, list) and len(rows) >= 1


def test_run_no_image():
    overlay, rows, summary, coco = _run(None, "포트홀 찾아줘", "mock")
    assert overlay is None and coco is None
    assert rows == []


def test_build_demo_if_gradio_installed():
    if importlib.util.find_spec("gradio") is None:
        print("skip - gradio 미설치")
        return
    from vlm.demo import build_demo
    assert build_demo() is not None


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok - {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
