"""탐지·분할 백엔드 공통 인터페이스."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..schemas import Detection


class DetectorBackend(ABC):
    """모든 탐지 백엔드가 구현하는 인터페이스."""

    name: str = "base"
    model: str = "unknown"

    @abstractmethod
    def detect(self, image_path: str, concepts: List[str]) -> List[Detection]:
        """이미지 + 개념 목록 -> 탐지 결과(bbox 픽셀좌표, 라벨, 신뢰도)."""
