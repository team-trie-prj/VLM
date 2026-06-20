"""Mock 탐지기 — 키/GPU 없이 파이프라인·오버레이를 검증하기 위한 결정론 더미."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional

from ..backends.base import image_size
from ..schemas import Detection
from .base import DetectorBackend


class MockDetector(DetectorBackend):
    name = "mock"
    model = "mock-detector-v1"

    def detect(
        self, image_path: str, concepts: List[str], mask_dir: Optional[str] = None
    ) -> List[Detection]:  # mask_dir 미지원(박스만)
        w, h = image_size(image_path)
        w = w or 640
        h = h or 480
        label = concepts[0] if concepts else "object"
        seed = int(hashlib.sha256(Path(image_path).name.encode()).hexdigest(), 16)
        n = 1 + (seed % 2)  # 1~2개
        dets: List[Detection] = []
        for i in range(n):
            cx = 0.3 + 0.25 * ((seed >> (i * 3)) % 3) / 2
            cy = 0.35 + 0.2 * ((seed >> (i * 5)) % 3) / 2
            bw, bh = 0.22 * w, 0.18 * h
            x0 = int(max(0, cx * w - bw / 2))
            y0 = int(max(0, cy * h - bh / 2))
            x1 = int(min(w, x0 + bw))
            y1 = int(min(h, y0 + bh))
            dets.append(
                Detection(label=label, box=[x0, y0, x1, y1],
                          confidence=round(0.5 + (seed % 40) / 100.0, 2))
            )
        return dets
