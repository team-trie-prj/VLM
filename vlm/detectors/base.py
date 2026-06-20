"""탐지·분할 백엔드 공통 인터페이스."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..schemas import Detection


class DetectorBackend(ABC):
    """모든 탐지 백엔드가 구현하는 인터페이스."""

    name: str = "base"
    model: str = "unknown"

    @abstractmethod
    def detect(
        self, image_path: str, concepts: List[str], mask_dir: Optional[str] = None
    ) -> List[Detection]:
        """이미지 + 개념 목록 -> 탐지 결과(bbox 픽셀좌표, 라벨, 신뢰도).

        mask_dir 가 주어지고 백엔드가 분할을 지원하면, 인스턴스별 마스크 PNG를
        mask_dir 에 저장하고 Detection.mask_path 를 채운다.
        """
