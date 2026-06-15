"""한글/영문 키워드 → 탐지 개념 매핑 + 규칙 기반 질의 파서.

LLM 없이도 자연어 질의를 개념 프롬프트로 바꾸기 위한 가벼운 폴백.
Mock 백엔드와 API 백엔드의 폴백 경로에서 공유한다.
"""

from __future__ import annotations

import re

from .schemas import ConceptPrompt

# 한글/영문 표현 -> 표준 영문 개념 키워드
KEYWORD_CONCEPTS = {
    "포트홀": "pothole",
    "pothole": "pothole",
    "구멍": "pothole",
    "균열": "crack",
    "크랙": "crack",
    "갈라짐": "crack",
    "crack": "crack",
    "거북등": "alligator_crack",
    "거북등균열": "alligator_crack",
    "alligator": "alligator_crack",
    "패임": "rutting",
    "바퀴자국": "rutting",
    "rutting": "rutting",
    "rut": "rutting",
    "땜빵": "patch",
    "패치": "patch",
    "보수": "patch",
    "patch": "patch",
    "맨홀": "manhole",
    "manhole": "manhole",
    "낙하물": "debris",
    "장애물": "debris",
    "debris": "debris",
    "차선": "lane_marking",
    "차로": "lane_marking",
    "lane": "lane_marking",
}


def detect_language(text: str) -> str:
    if re.search(r"[가-힣]", text):
        return "ko"
    if re.search(r"[a-zA-Z]", text):
        return "en"
    return "unknown"


def keyword_parse_query(query: str) -> ConceptPrompt:
    """규칙 기반 질의 파싱. 매칭되는 키워드를 개념으로 수집."""
    found: list[str] = []
    for term, concept in KEYWORD_CONCEPTS.items():
        if term in query and concept not in found:
            found.append(concept)
    return ConceptPrompt(
        original_query=query,
        language=detect_language(query),
        concepts=found,
        normalized=", ".join(found),
    )
