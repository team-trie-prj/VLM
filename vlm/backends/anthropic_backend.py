"""Anthropic Claude 비전 백엔드 — 1순위 실행 백엔드.

구조화 출력(messages.parse + Pydantic)으로 ImageAnalysis 스키마를 강제한다.
ANTHROPIC_API_KEY 환경변수가 필요하다.
"""

from __future__ import annotations

from typing import Optional

from ..keywords import keyword_parse_query
from ..prompts import (
    ANALYZE_INSTRUCTION,
    ANALYZE_SYSTEM,
    PARSE_QUERY_INSTRUCTION,
    PARSE_QUERY_SYSTEM,
    VQA_SYSTEM,
)
from ..schemas import ConceptPrompt, ImageAnalysis
from .base import VLMBackend, encode_image_base64, image_media_type

# skill: claude-api — 모델 ID는 레퍼런스 표의 정확한 문자열만 사용(날짜 접미사 금지).
DEFAULT_MODEL = "claude-opus-4-8"


class AnthropicVLM(VLMBackend):
    name = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 2048):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "anthropic 패키지가 필요합니다: pip install anthropic"
            ) from e

        self.model = model
        self.max_tokens = max_tokens
        # ANTHROPIC_API_KEY 환경변수에서 자동 인증
        self._client = anthropic.Anthropic()

    def _image_block(self, image_path: str) -> dict:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_media_type(image_path),
                "data": encode_image_base64(image_path),
            },
        }

    def analyze_image(
        self, image_path: str, *, instruction: Optional[str] = None
    ) -> ImageAnalysis:
        resp = self._client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=ANALYZE_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": [
                        self._image_block(image_path),
                        {"type": "text", "text": instruction or ANALYZE_INSTRUCTION},
                    ],
                }
            ],
            output_format=ImageAnalysis,
        )
        result = resp.parsed_output
        if result is None:
            raise RuntimeError(
                f"구조화 출력 실패 (stop_reason={resp.stop_reason}). "
                "안전 거부 또는 max_tokens 초과일 수 있습니다."
            )
        return result

    def vqa(self, image_path: str, question: str) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=VQA_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": [
                        self._image_block(image_path),
                        {"type": "text", "text": question},
                    ],
                }
            ],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    def parse_query(self, query: str) -> ConceptPrompt:
        try:
            resp = self._client.messages.parse(
                model=self.model,
                max_tokens=512,
                system=PARSE_QUERY_SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": PARSE_QUERY_INSTRUCTION.format(query=query),
                    }
                ],
                output_format=ConceptPrompt,
            )
            result = resp.parsed_output
            if result is not None:
                # 원문은 항상 사용자가 입력한 값으로 고정
                result.original_query = query
                return result
        except Exception:
            pass
        # 실패 시 규칙 기반 폴백
        return keyword_parse_query(query)
