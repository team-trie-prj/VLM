"""Mock 백엔드 + 파이프라인 스모크 테스트 (키/GPU/네트워크 불필요).

  python -m pytest -q        # pytest 있을 때
  python tests/test_mock.py  # 직접 실행
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from vlm.backends.mock import MockVLM  # noqa: E402
from vlm.keywords import keyword_parse_query  # noqa: E402
from vlm.pipeline import VLMProcessor  # noqa: E402
from vlm.schemas import ImageAnalysis  # noqa: E402


def _tmp_image(dirpath: Path, name: str) -> Path:
    p = dirpath / name
    Image.new("RGB", (320, 240), (90, 90, 95)).save(p)
    return p


def test_analyze_returns_valid_schema():
    backend = MockVLM()
    with tempfile.TemporaryDirectory() as d:
        img = _tmp_image(Path(d), "road_pothole_01.png")
        analysis = backend.analyze_image(str(img))
    assert isinstance(analysis, ImageAnalysis)
    assert analysis.damage_present is True
    assert any(dmg.type == "pothole" for dmg in analysis.damages)
    for dmg in analysis.damages:
        assert 0.0 <= dmg.confidence <= 1.0


def test_analyze_is_deterministic():
    backend = MockVLM()
    with tempfile.TemporaryDirectory() as d:
        img = _tmp_image(Path(d), "road_crack_02.png")
        a1 = backend.analyze_image(str(img))
        a2 = backend.analyze_image(str(img))
    assert a1.model_dump() == a2.model_dump()


def test_parse_query_korean():
    cp = keyword_parse_query("포트홀이랑 균열 찾아줘")
    assert cp.language == "ko"
    assert "pothole" in cp.concepts
    assert "crack" in cp.concepts


def test_pipeline_process_and_save():
    backend = MockVLM()
    with tempfile.TemporaryDirectory() as d:
        img = _tmp_image(Path(d), "road_clean_03.png")
        out_dir = Path(d) / "out"
        proc = VLMProcessor(backend, output_dir=out_dir)
        result = proc.process(img, query="포트홀 찾아줘")
        saved = proc.save_result(result)
        assert saved.exists()
        assert result.query is not None
        assert result.backend == "mock"
        meta = result.to_metadata()
        assert meta["source"] == "auto"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok - {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
