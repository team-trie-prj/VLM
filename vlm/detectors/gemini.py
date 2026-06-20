"""Gemini 2.5 기반 탐지기 (파이프라인 ③, 무료·GPU 불필요).

Gemini는 box_2d([ymin,xmin,ymax,xmax] 0~1000 정규화) + label 을 JSON으로 반환한다.
이를 픽셀 좌표 Detection 으로 변환한다. GEMINI_API_KEY 필요.
(분할 mask는 후속 증분에서 추가 예정)
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import List

from pydantic import BaseModel

from ..backends.base import image_media_type, image_size
from ..schemas import Detection
from .base import DetectorBackend

DEFAULT_MODEL = "gemini-2.5-flash"

_PROMPT = (
    "Detect all instances of the following on this road image: {concepts}. "
    "Return a JSON list; each item has 'box_2d' as [ymin, xmin, ymax, xmax] "
    "normalized to 0-1000, a 'label' (one of: {concepts}), and a 'confidence' "
    "between 0 and 1. Return an empty list if none are visible."
)


class _RawDet(BaseModel):
    box_2d: List[int]  # [ymin, xmin, ymax, xmax] 0~1000
    label: str
    confidence: float = 0.5


class GeminiDetector(DetectorBackend):
    name = "gemini"

    def __init__(self, model: str = DEFAULT_MODEL):
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:  # pragma: no cover
            raise ImportError("google-genai 패키지가 필요합니다: pip install google-genai") from e

        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY(또는 GOOGLE_API_KEY) 환경변수가 필요합니다. "
                "https://aistudio.google.com 에서 무료 발급."
            )
        self.model = model
        self._types = types
        self._client = genai.Client(api_key=key)

    def _gen(self, contents, config):
        delay = 5.0
        for attempt in range(5):
            try:
                return self._client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
            except Exception as e:
                msg = str(e)
                if attempt < 4 and ("429" in msg or "RESOURCE_EXHAUSTED" in msg):
                    m = re.search(r"retry in (\d+(?:\.\d+)?)", msg)
                    time.sleep(min((float(m.group(1)) + 1.0) if m else delay, 60.0))
                    delay = min(delay * 2, 60.0)
                    continue
                raise

    def detect(self, image_path: str, concepts: List[str]) -> List[Detection]:
        w, h = image_size(image_path)
        w = w or 1
        h = h or 1
        concept_str = ", ".join(concepts) if concepts else "road damage"

        img = self._types.Part.from_bytes(
            data=Path(image_path).read_bytes(), mime_type=image_media_type(image_path)
        )
        cfg = self._types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[_RawDet],
            max_output_tokens=2048,
        )
        resp = self._gen([img, _PROMPT.format(concepts=concept_str)], cfg)

        raw = getattr(resp, "parsed", None)
        if raw is None:
            import json

            raw = [_RawDet(**d) for d in json.loads(resp.text or "[]")]

        dets: List[Detection] = []
        for r in raw:
            if len(r.box_2d) != 4:
                continue
            ymin, xmin, ymax, xmax = r.box_2d
            x0 = max(0, min(w, round(xmin / 1000 * w)))
            y0 = max(0, min(h, round(ymin / 1000 * h)))
            x1 = max(0, min(w, round(xmax / 1000 * w)))
            y1 = max(0, min(h, round(ymax / 1000 * h)))
            if x1 <= x0 or y1 <= y0:  # 비정상 박스 스킵
                continue
            dets.append(
                Detection(label=r.label, box=[x0, y0, x1, y1],
                          confidence=round(float(r.confidence), 3))
            )
        return dets
