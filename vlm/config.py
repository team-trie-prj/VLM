"""설정 로딩 + 백엔드 팩토리.

우선순위: 환경변수(VLM_BACKEND, VLM_MODEL) > config 파일 > 기본값.
백엔드는 지연 import 하므로, 선택하지 않은 백엔드의 선택 의존성(torch 등)이
없어도 다른 백엔드 사용에 지장이 없다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from .backends.base import VLMBackend

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"

_DEFAULTS: Dict[str, Any] = {
    "backend": "mock",
    "model": None,  # None이면 백엔드별 기본 모델 사용
    "max_tokens": 2048,
    "output_dir": "data/outputs",
}


def _load_dotenv() -> None:
    """.env가 있으면 로드(python-dotenv 설치 시). 없으면 조용히 통과."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    _load_dotenv()
    cfg = dict(_DEFAULTS)

    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if cfg_path.exists():
        try:
            import yaml

            with open(cfg_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            cfg.update({k: v for k, v in loaded.items() if v is not None})
        except ImportError:
            pass  # pyyaml 미설치 시 기본값 사용

    # 환경변수 오버라이드
    if os.getenv("VLM_BACKEND"):
        cfg["backend"] = os.environ["VLM_BACKEND"]
    if os.getenv("VLM_MODEL"):
        cfg["model"] = os.environ["VLM_MODEL"]

    return cfg


def build_backend(config: Optional[Dict[str, Any]] = None) -> VLMBackend:
    """config에 따라 백엔드 인스턴스를 생성한다."""
    cfg = config or load_config()
    backend = str(cfg.get("backend", "mock")).lower()
    model = cfg.get("model")
    max_tokens = int(cfg.get("max_tokens", 2048))

    if backend == "mock":
        from .backends.mock import MockVLM

        return MockVLM()

    if backend == "anthropic":
        from .backends.anthropic_backend import DEFAULT_MODEL, AnthropicVLM

        return AnthropicVLM(model=model or DEFAULT_MODEL, max_tokens=max_tokens)

    if backend == "openai":
        from .backends.openai_backend import DEFAULT_MODEL, OpenAIVLM

        return OpenAIVLM(model=model or DEFAULT_MODEL, max_tokens=max_tokens)

    if backend == "gemini":
        from .backends.gemini_backend import DEFAULT_MODEL, GeminiVLM

        return GeminiVLM(model=model or DEFAULT_MODEL, max_tokens=max_tokens)

    if backend == "qwen":
        from .backends.qwen_backend import DEFAULT_MODEL, QwenVLM

        return QwenVLM(model=model or DEFAULT_MODEL)

    raise ValueError(
        f"알 수 없는 backend: {backend!r} "
        "(사용 가능: mock, anthropic, openai, gemini, qwen)"
    )


def build_detector(config: Optional[Dict[str, Any]] = None):
    """config에 따라 탐지 백엔드(파이프라인 ③) 인스턴스를 생성한다."""
    cfg = config or load_config()
    detector = str(cfg.get("detector", "gemini")).lower()

    if detector == "mock":
        from .detectors.mock import MockDetector

        return MockDetector()

    if detector == "gemini":
        from .detectors.gemini import DEFAULT_MODEL, GeminiDetector

        return GeminiDetector(model=cfg.get("detector_model") or DEFAULT_MODEL)

    if detector == "yolo":
        from .detectors.yolo import DEFAULT_WEIGHTS, YoloDetector

        return YoloDetector(weights=cfg.get("yolo_weights") or DEFAULT_WEIGHTS,
                            conf=float(cfg.get("yolo_conf", 0.25)))

    raise ValueError(f"알 수 없는 detector: {detector!r} (사용 가능: mock, gemini, yolo)")
