"""API 백엔드 테스트.

- 비용 계산은 키 없이 항상 검증.
- 실제 API 호출은 패키지 설치 + 키 존재 시에만 수행하고, 없으면 자동 skip.

  python -m pytest -q tests/test_api_backends.py
  python tests/test_api_backends.py   # pytest 없이도 실행(skip 사유 출력)
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from vlm.pricing import estimate_cost_usd  # noqa: E402
from vlm.schemas import ImageAnalysis  # noqa: E402


class SkipTest(Exception):
    pass


def _have(pkg: str) -> bool:
    try:
        return importlib.util.find_spec(pkg) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _tmp_image() -> str:
    d = tempfile.mkdtemp()
    p = Path(d) / "road_pothole.png"
    img = Image.new("RGB", (320, 240), (90, 90, 95))
    img.save(p)
    return str(p)


# --- 키 불필요 ----------------------------------------------------------------
def test_cost_estimation_known_model():
    # opus 4.8: $5 / $25 per MTok
    cost = estimate_cost_usd("claude-opus-4-8", 1_000_000, 1_000_000)
    assert cost == 30.0
    # 별칭/접두 매칭
    assert estimate_cost_usd("claude-haiku-4-5", 1_000_000, 0) == 1.0


def test_cost_estimation_unknown_model_is_none():
    assert estimate_cost_usd("some-unknown-model", 100, 100) is None


# --- 키 있을 때만 (없으면 skip) ----------------------------------------------
def test_anthropic_live_analyze():
    if not _have("anthropic"):
        raise SkipTest("anthropic 미설치")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SkipTest("ANTHROPIC_API_KEY 없음")

    from vlm.backends.anthropic_backend import AnthropicVLM

    backend = AnthropicVLM()
    analysis = backend.analyze_image(_tmp_image())
    assert isinstance(analysis, ImageAnalysis)
    assert backend.last_usage is not None
    assert backend.last_usage.input_tokens > 0


def test_openai_live_analyze():
    if not _have("openai"):
        raise SkipTest("openai 미설치")
    if not os.getenv("OPENAI_API_KEY"):
        raise SkipTest("OPENAI_API_KEY 없음")

    from vlm.backends.openai_backend import OpenAIVLM

    backend = OpenAIVLM()
    analysis = backend.analyze_image(_tmp_image())
    assert isinstance(analysis, ImageAnalysis)


def test_gemini_live_analyze():
    if not _have("google.genai"):
        raise SkipTest("google-genai 미설치")
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        raise SkipTest("GEMINI_API_KEY 없음")

    from vlm.backends.gemini_backend import GeminiVLM

    backend = GeminiVLM()
    analysis = backend.analyze_image(_tmp_image())
    assert isinstance(analysis, ImageAnalysis)
    assert backend.last_usage is not None


# pytest 없이 직접 실행할 때의 러너
def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = skipped = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   - {fn.__name__}")
            passed += 1
        except SkipTest as s:
            print(f"skip - {fn.__name__} ({s})")
            skipped += 1
    print(f"\n{passed} passed, {skipped} skipped")


if __name__ == "__main__":
    _run_all()
