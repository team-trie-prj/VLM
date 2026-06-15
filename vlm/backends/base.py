"""백엔드 공통 인터페이스 + 이미지 유틸."""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

from ..schemas import ConceptPrompt, ImageAnalysis

_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

_VALID_SUFFIXES = set(_MEDIA_TYPES)


def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in _VALID_SUFFIXES


def image_media_type(path: str | Path) -> str:
    return _MEDIA_TYPES.get(Path(path).suffix.lower(), "image/jpeg")


def encode_image_base64(path: str | Path) -> str:
    return base64.standard_b64encode(Path(path).read_bytes()).decode("utf-8")


def image_size(path: str | Path) -> Tuple[Optional[int], Optional[int]]:
    """(width, height) 반환. Pillow 미설치/실패 시 (None, None)."""
    try:
        from PIL import Image  # 지연 import: Pillow 없이도 모듈 로드 가능

        with Image.open(path) as im:
            return im.size
    except Exception:
        return (None, None)


def image_data_url(path: str | Path) -> str:
    """OpenAI 등에서 쓰는 data URL 형태."""
    return f"data:{image_media_type(path)};base64,{encode_image_base64(path)}"


class VLMBackend(ABC):
    """모든 VLM 백엔드가 구현하는 공통 인터페이스."""

    #: 사람이 읽는 백엔드 이름 (로그/결과 기록용)
    name: str = "base"
    #: 사용 모델 식별자
    model: str = "unknown"

    @abstractmethod
    def analyze_image(
        self, image_path: str, *, instruction: Optional[str] = None
    ) -> ImageAnalysis:
        """이미지 1장 → 구조화 분석(캡션 + 속성 + 손상 추정)."""

    @abstractmethod
    def vqa(self, image_path: str, question: str) -> str:
        """이미지에 대한 자연어 질의응답(VQA)."""

    @abstractmethod
    def parse_query(self, query: str) -> ConceptPrompt:
        """자연어 질의 → 탐지/분할용 개념 프롬프트."""
