"""모델별 단가표 + 비용 추정.

제안서의 "기대효과(비용 절감)" 정량화를 위해 호출 비용을 추정한다.
단가는 백만 토큰(MTok)당 USD (입력, 출력). 2026-06 기준 — 실제 청구 전
공식 가격 페이지로 재확인 권장. (캐시 읽기/쓰기 할인은 단순화를 위해 미반영)
"""

from __future__ import annotations

from typing import Optional

# Anthropic — claude-api 스킬 레퍼런스 표 기준 (권위 있는 값)
ANTHROPIC_PRICING = {
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

# OpenAI — 대략치. 정확한 단가는 OpenAI 공식 페이지 확인 필요.
OPENAI_PRICING = {
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
}

# Gemini — 무료 티어 한도 내 사용 기준 0. (유료 전환 시 공식 단가 참조)
GEMINI_PRICING = {
    "gemini-2.5-flash": (0.0, 0.0),
    "gemini-2.0-flash": (0.0, 0.0),
}

_TABLE = {**ANTHROPIC_PRICING, **OPENAI_PRICING, **GEMINI_PRICING}


def estimate_cost_usd(
    model: str, input_tokens: int, output_tokens: int
) -> Optional[float]:
    """모델 단가표에 있으면 예상 비용(USD), 없으면 None."""
    key = next((k for k in _TABLE if model.startswith(k)), None)
    if key is None:
        return None
    c_in, c_out = _TABLE[key]
    cost = input_tokens / 1_000_000 * c_in + output_tokens / 1_000_000 * c_out
    return round(cost, 6)
