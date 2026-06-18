"""Google Gemini 비전 백엔드 — 무료 티어 사용 가능.

google-genai SDK 사용. GEMINI_API_KEY(또는 GOOGLE_API_KEY) 환경변수 필요.
무료 키 발급(결제수단 불필요): https://aistudio.google.com
구조화 출력(response_schema=Pydantic)으로 ImageAnalysis 스키마를 강제한다.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional

from ..keywords import keyword_parse_query
from ..pricing import estimate_cost_usd
from ..prompts import (
    ANALYZE_INSTRUCTION,
    ANALYZE_SYSTEM,
    PARSE_QUERY_INSTRUCTION,
    PARSE_QUERY_SYSTEM,
    VQA_SYSTEM,
)
from ..schemas import ConceptPrompt, ImageAnalysis, Usage
from .base import VLMBackend, image_media_type

DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiVLM(VLMBackend):
    name = "gemini"

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 2048):
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "google-genai 패키지가 필요합니다: pip install google-genai"
            ) from e

        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY(또는 GOOGLE_API_KEY) 환경변수가 필요합니다. "
                "https://aistudio.google.com 에서 무료로 발급하세요."
            )

        self.model = model
        self.max_tokens = max_tokens
        self._types = types
        self._client = genai.Client(api_key=key)

    # --- 내부 유틸 ----------------------------------------------------------
    def _image_part(self, image_path: str):
        return self._types.Part.from_bytes(
            data=Path(image_path).read_bytes(),
            mime_type=image_media_type(image_path),
        )

    def _gen(self, contents, config):
        """generate_content + 무료 티어 429(분당 한도 초과) 재시도(backoff)."""
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
                    wait = (float(m.group(1)) + 1.0) if m else delay
                    time.sleep(min(wait, 60.0))
                    delay = min(delay * 2, 60.0)
                    continue
                raise

    def _record_usage(self, resp) -> None:
        u = getattr(resp, "usage_metadata", None)
        if u is None:
            self.last_usage = None
            return
        inp = getattr(u, "prompt_token_count", 0) or 0
        out = getattr(u, "candidates_token_count", 0) or 0
        self.last_usage = Usage(
            input_tokens=inp,
            output_tokens=out,
            estimated_cost_usd=estimate_cost_usd(self.model, inp, out),
        )

    # --- 공개 API ----------------------------------------------------------
    def analyze_image(
        self, image_path: str, *, instruction: Optional[str] = None
    ) -> ImageAnalysis:
        cfg = self._types.GenerateContentConfig(
            system_instruction=ANALYZE_SYSTEM,
            response_mime_type="application/json",
            response_schema=ImageAnalysis,
            max_output_tokens=self.max_tokens,
        )
        resp = self._gen(
            [self._image_part(image_path), instruction or ANALYZE_INSTRUCTION], cfg
        )
        self._record_usage(resp)
        parsed = getattr(resp, "parsed", None)
        if isinstance(parsed, ImageAnalysis):
            return parsed
        return ImageAnalysis.model_validate_json(resp.text)  # 폴백

    def vqa(self, image_path: str, question: str) -> str:
        cfg = self._types.GenerateContentConfig(
            system_instruction=VQA_SYSTEM, max_output_tokens=1024
        )
        resp = self._gen([self._image_part(image_path), question], cfg)
        self._record_usage(resp)
        return (resp.text or "").strip()

    def parse_query(self, query: str) -> ConceptPrompt:
        try:
            cfg = self._types.GenerateContentConfig(
                system_instruction=PARSE_QUERY_SYSTEM,
                response_mime_type="application/json",
                response_schema=ConceptPrompt,
                max_output_tokens=512,
            )
            resp = self._gen([PARSE_QUERY_INSTRUCTION.format(query=query)], cfg)
            self._record_usage(resp)
            parsed = getattr(resp, "parsed", None)
            if not isinstance(parsed, ConceptPrompt):
                parsed = ConceptPrompt.model_validate_json(resp.text)
            parsed.original_query = query
            return parsed
        except Exception:
            return keyword_parse_query(query)

    def health_check(self) -> str:
        self._gen(["ping"], self._types.GenerateContentConfig(max_output_tokens=8))
        return f"gemini OK - model={self.model}, 키 인증/연결 정상 (무료 티어)"
