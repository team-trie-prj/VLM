"""Anthropic Claude 비전 백엔드 — 1순위 실행 백엔드.

구조화 출력(messages.parse + Pydantic)으로 ImageAnalysis 스키마를 강제하고,
호출마다 토큰 사용량/예상 비용을 기록한다.
ANTHROPIC_API_KEY 환경변수가 필요하다.
"""

from __future__ import annotations

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
from .base import VLMBackend, encode_image_base64, image_media_type

# skill: claude-api — 모델 ID는 레퍼런스 표의 정확한 문자열만 사용(날짜 접미사 금지).
DEFAULT_MODEL = "claude-opus-4-8"


class AnthropicVLM(VLMBackend):
    name = "anthropic"

    def __init__(
        self, model: str = DEFAULT_MODEL, max_tokens: int = 2048, timeout: float = 60.0
    ):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "anthropic 패키지가 필요합니다: pip install anthropic"
            ) from e

        self.model = model
        self.max_tokens = max_tokens
        self._anthropic = anthropic
        # ANTHROPIC_API_KEY 환경변수에서 자동 인증. SDK가 429/5xx 자동 재시도.
        self._client = anthropic.Anthropic(timeout=timeout, max_retries=3)

    # --- 내부 유틸 ----------------------------------------------------------
    def _image_block(self, image_path: str) -> dict:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_media_type(image_path),
                "data": encode_image_base64(image_path),
            },
        }

    def _record_usage(self, resp) -> None:
        u = getattr(resp, "usage", None)
        if u is None:
            self.last_usage = None
            return
        inp = getattr(u, "input_tokens", 0) or 0
        out = getattr(u, "output_tokens", 0) or 0
        self.last_usage = Usage(
            input_tokens=inp,
            output_tokens=out,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            estimated_cost_usd=estimate_cost_usd(self.model, inp, out),
        )

    def _on_error(self, e: Exception) -> Exception:
        """SDK 예외를 사용자 친화 메시지로 변환."""
        if isinstance(e, self._anthropic.AuthenticationError):
            return RuntimeError(
                "ANTHROPIC_API_KEY 인증 실패 - 키 값을 확인하세요(.env 또는 환경변수)."
            )
        if isinstance(e, self._anthropic.RateLimitError):
            return RuntimeError("요청 한도 초과(rate limit). 잠시 후 다시 시도하세요.")
        if isinstance(e, self._anthropic.APIConnectionError):
            return RuntimeError("Anthropic API 연결 실패 - 네트워크를 확인하세요.")
        return e

    # --- 공개 API ----------------------------------------------------------
    def analyze_image(
        self, image_path: str, *, instruction: Optional[str] = None
    ) -> ImageAnalysis:
        try:
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
        except Exception as e:
            raise self._on_error(e) from e

        self._record_usage(resp)
        result = resp.parsed_output
        if result is None:
            raise RuntimeError(
                f"구조화 출력 실패 (stop_reason={resp.stop_reason}). "
                "안전 거부 또는 max_tokens 초과일 수 있습니다."
            )
        return result

    def vqa(self, image_path: str, question: str) -> str:
        try:
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
        except Exception as e:
            raise self._on_error(e) from e
        self._record_usage(resp)
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
            self._record_usage(resp)
            result = resp.parsed_output
            if result is not None:
                result.original_query = query  # 원문은 항상 입력값으로 고정
                return result
        except Exception:
            pass
        return keyword_parse_query(query)  # 실패 시 규칙 기반 폴백

    def health_check(self) -> str:
        try:
            self._client.messages.create(
                model=self.model,
                max_tokens=8,
                messages=[{"role": "user", "content": "ping"}],
            )
        except Exception as e:
            raise self._on_error(e) from e
        return f"anthropic OK - model={self.model}, 키 인증/연결 정상"
