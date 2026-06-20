"""compare 모듈 테스트 (mock 백엔드, 키/네트워크 불필요).

  python -m pytest -q tests/test_compare.py
  python tests/test_compare.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from vlm.compare import _jaccard, compare, format_report  # noqa: E402


def _tmp_images(n: int) -> str:
    d = tempfile.mkdtemp()
    for i in range(n):
        # 파일명 힌트로 손상 유형을 다양화 (mock 결정론)
        name = ["road_pothole", "road_crack", "road_clean"][i % 3]
        Image.new("RGB", (320, 240), (90, 90, 95)).save(Path(d) / f"{name}_{i}.png")
    return d


def test_jaccard():
    assert _jaccard(set(), set()) == 1.0
    assert _jaccard({"pothole"}, {"pothole"}) == 1.0
    assert _jaccard({"pothole"}, set()) == 0.0
    assert _jaccard({"a", "b"}, {"b", "c"}) == 1 / 3


def test_compare_structure_mock():
    d = _tmp_images(6)
    report = compare(d, ["mock"], delay=0.0)
    assert report["count"] == 6
    assert "mock" in report["per_backend"]
    pb = report["per_backend"]["mock"]
    assert pb["images_ok"] == 6
    assert 0.0 <= pb["damage_present_rate"] <= 1.0
    assert len(report["per_image"]) == 6
    # mock 단일 백엔드는 pairwise 없음
    assert report["pairwise"] == []
    # 리포트 포맷이 문자열을 반환
    assert "backend" in format_report(report)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok - {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
