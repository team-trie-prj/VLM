"""파인튜닝한 YOLO 가중치로 탐지 (파이프라인 ⑥ 결과 활용).

Colab에서 파인튜닝한 best.pt 를 로컬 CPU로 돌려 우리 파이프라인에 끼운다.
이로써 `compare`로 "파인튜닝 YOLO vs Gemini" 비교가 가능하다.
필요 패키지: ultralytics (requirements-local.txt). GPU 없이 CPU 추론 가능(느림).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ..backends.base import image_size
from ..schemas import Detection
from .base import DetectorBackend

DEFAULT_WEIGHTS = "models/pothole_yolo.pt"


class YoloDetector(DetectorBackend):
    name = "yolo"

    def __init__(self, weights: str = DEFAULT_WEIGHTS, conf: float = 0.25):
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"YOLO 가중치가 없습니다: {weights}\n"
                "Colab 노트북(notebooks/finetune_yolo_colab.ipynb)에서 파인튜닝한 "
                "best.pt 를 이 경로에 두세요."
            )
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "ultralytics 패키지가 필요합니다: pip install ultralytics"
            ) from e

        self.model = weights
        self.conf = conf
        self._yolo = YOLO(weights)

    def detect(
        self, image_path: str, concepts: List[str], mask_dir: Optional[str] = None
    ) -> List[Detection]:
        w, h = image_size(image_path)
        res = self._yolo(image_path, conf=self.conf, verbose=False)[0]
        names = res.names
        dets: List[Detection] = []
        for b in res.boxes:
            cls = int(b.cls[0])
            label = str(names.get(cls, cls)) if isinstance(names, dict) else str(names[cls])
            x0, y0, x1, y1 = (round(v) for v in b.xyxy[0].tolist())
            dets.append(Detection(
                label=label, box=[x0, y0, x1, y1],
                confidence=round(float(b.conf[0]), 3),
            ))
        return dets
