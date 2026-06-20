"""Gemini 2.5 기반 탐지기 (파이프라인 ③, 무료·GPU 불필요).

Gemini는 box_2d([ymin,xmin,ymax,xmax] 0~1000 정규화) + label 을 JSON으로 반환한다.
이를 픽셀 좌표 Detection 으로 변환한다. GEMINI_API_KEY 필요.
(분할 mask는 후속 증분에서 추가 예정)
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from pathlib import Path
from typing import List, Optional

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

_MASK_PROMPT = (
    "Give the segmentation masks for all instances of: {concepts} on this road "
    "image. Output a JSON list of segmentation masks where each entry contains "
    "the 2D bounding box in the key 'box_2d' (0-1000), the segmentation mask in "
    "key 'mask' as a base64-encoded PNG, the text label in key 'label', and a "
    "'confidence' (0-1). Empty list if none."
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

    @staticmethod
    def _box_px(box_2d, w, h):
        if len(box_2d) != 4:
            return None
        ymin, xmin, ymax, xmax = box_2d
        x0 = max(0, min(w, round(xmin / 1000 * w)))
        y0 = max(0, min(h, round(ymin / 1000 * h)))
        x1 = max(0, min(w, round(xmax / 1000 * w)))
        y1 = max(0, min(h, round(ymax / 1000 * h)))
        return (x0, y0, x1, y1) if (x1 > x0 and y1 > y0) else None

    @staticmethod
    def _save_mask(mask_field, box, w, h, out_png) -> Optional[str]:
        from PIL import Image

        try:
            s = mask_field.split(",", 1)[-1] if "," in mask_field else mask_field
            m = Image.open(io.BytesIO(base64.b64decode(s))).convert("L")
        except Exception:
            return None
        x0, y0, x1, y1 = box
        m = m.resize((x1 - x0, y1 - y0)).point(lambda p: 255 if p >= 127 else 0)
        full = Image.new("L", (w, h), 0)
        full.paste(m, (x0, y0))
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        full.save(out_png)
        return str(out_png)

    def _detect_boxes(self, img, concept_str, w, h) -> List[Detection]:
        cfg = self._types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[_RawDet],
            max_output_tokens=2048,
        )
        resp = self._gen([img, _PROMPT.format(concepts=concept_str)], cfg)
        raw = getattr(resp, "parsed", None)
        if raw is None:
            raw = [_RawDet(**d) for d in json.loads(resp.text or "[]")]
        dets: List[Detection] = []
        for r in raw:
            box = self._box_px(r.box_2d, w, h)
            if box:
                dets.append(Detection(label=r.label, box=list(box),
                                      confidence=round(float(r.confidence), 3)))
        return dets

    def detect(
        self, image_path: str, concepts: List[str], mask_dir: Optional[str] = None
    ) -> List[Detection]:
        w, h = image_size(image_path)
        w = w or 1
        h = h or 1
        concept_str = ", ".join(concepts) if concepts else "road damage"
        img = self._types.Part.from_bytes(
            data=Path(image_path).read_bytes(), mime_type=image_media_type(image_path)
        )

        if not mask_dir:
            return self._detect_boxes(img, concept_str, w, h)

        # 박스 + 분할 mask (base64 PNG 파싱). mask는 매우 길어 토큰 여유를 크게.
        # 참고: 무료 gemini-2.5-flash는 유효 PNG mask 반환이 불안정 → mask 실패 시
        #       박스 탐지로 폴백한다(박스는 항상 확보).
        cfg = self._types.GenerateContentConfig(
            response_mime_type="application/json", max_output_tokens=32768
        )
        resp = self._gen([img, _MASK_PROMPT.format(concepts=concept_str)], cfg)
        text = resp.text or "[]"
        m = re.search(r"```(?:json)?\s*(\[.*\])\s*```", text, re.DOTALL)
        try:
            items = json.loads(m.group(1) if m else text)
        except Exception:
            items = []

        stem = Path(image_path).stem
        dets: List[Detection] = []
        for i, it in enumerate(items):
            box = self._box_px(it.get("box_2d", []), w, h)
            if not box:
                continue
            mask_path = None
            if it.get("mask"):
                mask_path = self._save_mask(
                    it["mask"], box, w, h, Path(mask_dir) / f"{stem}_{i}.png"
                )
            dets.append(Detection(
                label=it.get("label", concept_str), box=list(box),
                confidence=round(float(it.get("confidence", 0.5)), 3),
                mask_path=mask_path,
            ))

        if not dets:  # mask 응답이 비정상이면 박스만이라도 확보
            return self._detect_boxes(img, concept_str, w, h)
        return dets
