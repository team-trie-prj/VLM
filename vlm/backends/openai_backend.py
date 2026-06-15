"""OpenAI GPT-4o 비전 백엔드 (대안 API).

OpenAI SDK 버전 차를 피하기 위해 JSON 모드 + Pydantic 검증 방식을 사용한다.
OPENAI_API_KEY 환경변수가 필요하다.
"""

from __future__ import annotations

import json
from typing import Optional

from pydantic import ValidationError

from ..keywords import keyword_parse_query
from ..pricing import estimate_cost_usd
from ..prompts import (
    ANALYZE_INSTRUCTION,
    ANALYZE_SYSTEM,
    PARSE_QUERY_SYSTEM,
    VQA_SYSTEM,
)
from ..schemas import ConceptPrompt, ImageAnalysis, Usage
from .base import VLMBackend, image_data_url

DEFAULT_MODEL = "gpt-4o"


def _schema_hint(model_cls) -> str:
    return (
        "Respond ONLY with a JSON object matching this JSON Schema "
        "(no markdown, no commentary):\n"
        + json.dumps(model_cls.model_json_schema(), ensure_ascii=False)
    )


class OpenAIVLM(VLMBackend):
    name = "openai"

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 2048):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError("openai 패키지가 필요합니다: pip install openai") from e

        self.model = model
        self.max_tokens = max_tokens
        self._client = OpenAI()  # OPENAI_API_KEY 환경변수에서 자동 인증

    def _image_content(self, image_path: str) -> dict:
        return {"type": "image_url", "image_url": {"url": image_data_url(image_path)}}

    def _record_usage(self, resp) -> None:
        u = getattr(resp, "usage", None)
        if u is None:
            self.last_usage = None
            return
        inp = getattr(u, "prompt_tokens", 0) or 0
        out = getattr(u, "completion_tokens", 0) or 0
        self.last_usage = Usage(
            input_tokens=inp,
            output_tokens=out,
            estimated_cost_usd=estimate_cost_usd(self.model, inp, out),
        )

    def _json_call(self, system: str, user_content: list) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        )
        self._record_usage(resp)
        return resp.choices[0].message.content or "{}"

    def analyze_image(
        self, image_path: str, *, instruction: Optional[str] = None
    ) -> ImageAnalysis:
        system = ANALYZE_SYSTEM + "\n\n" + _schema_hint(ImageAnalysis)
        user_content = [
            {"type": "text", "text": instruction or ANALYZE_INSTRUCTION},
            self._image_content(image_path),
        ]
        raw = self._json_call(system, user_content)
        try:
            return ImageAnalysis.model_validate_json(raw)
        except ValidationError as e:
            raise RuntimeError(f"OpenAI 구조화 출력 검증 실패: {e}\nraw={raw[:500]}")

    def vqa(self, image_path: str, question: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": VQA_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        self._image_content(image_path),
                    ],
                },
            ],
        )
        self._record_usage(resp)
        return (resp.choices[0].message.content or "").strip()

    def parse_query(self, query: str) -> ConceptPrompt:
        try:
            system = PARSE_QUERY_SYSTEM + "\n\n" + _schema_hint(ConceptPrompt)
            raw = self._json_call(system, [{"type": "text", "text": query}])
            result = ConceptPrompt.model_validate_json(raw)
            result.original_query = query
            return result
        except Exception:
            return keyword_parse_query(query)

    def health_check(self) -> str:
        self._client.chat.completions.create(
            model=self.model,
            max_tokens=8,
            messages=[{"role": "user", "content": "ping"}],
        )
        return f"openai OK - model={self.model}, 키 인증/연결 정상"
