"""데이터 구조 정의 (Pydantic v2).

제안서의 "데이터 구조 설계안"과 직결:
  - ImageAnalysis : VLM이 채우는 구조화 분석 결과(캡션 + 속성 + 손상 추정)
  - VLMResult     : 이미지 1장에 대해 영속화하는 레코드(메타데이터 + 분석 + 출처)
  - ConceptPrompt : 자연어 질의를 탐지/분할 단계(SAM3/YOLOE)로 넘기기 위한 개념 프롬프트
  - PromptLog     : 재현성/평가용 프롬프트 로그
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# --- 도메인 enum (도로 손상 도메인) --------------------------------------------
Weather = Literal["clear", "cloudy", "rain", "snow", "fog", "unknown"]
TimeOfDay = Literal["day", "night", "dawn_dusk", "unknown"]
SceneType = Literal[
    "highway",
    "urban_road",
    "alley",
    "rural_road",
    "parking_lot",
    "sidewalk",
    "other",
    "unknown",
]
SurfaceMaterial = Literal["asphalt", "concrete", "unpaved", "brick", "other", "unknown"]
Condition = Literal["good", "fair", "poor", "severe"]
Severity = Literal["low", "medium", "high"]


class DamageInstance(BaseModel):
    """추정된 손상 1건. (정확한 bbox/mask는 탐지·분할 단계의 책임)"""

    type: str = Field(description="손상 유형 영문 키워드 (pothole, crack, rutting, patch, ...)")
    severity: Severity = Field(description="심각도 추정")
    confidence: float = Field(description="0.0~1.0 신뢰도")
    description: str = Field(default="", description="짧은 설명(선택)")


class ImageAnalysis(BaseModel):
    """VLM이 한 번의 호출로 채우는 구조화 분석 결과.

    파이프라인 ①(이미지 이해)의 핵심 산출물. 구조화 출력(JSON schema)으로 강제된다.
    """

    caption: str = Field(description="이미지 1~2문장 캡션")
    scene_type: SceneType = Field(description="도로/장면 종류 추정")
    surface_material: SurfaceMaterial = Field(description="노면 재질 추정")
    weather: Weather = Field(description="날씨 추정")
    time_of_day: TimeOfDay = Field(description="촬영 시간대 추정")
    damage_present: bool = Field(description="손상 존재 여부")
    damages: List[DamageInstance] = Field(
        default_factory=list, description="추정된 손상 목록 (없으면 빈 배열)"
    )
    overall_condition: Condition = Field(description="노면 전반 상태")
    notes: str = Field(default="", description="검수자에게 유용한 추가 메모")


class ConceptPrompt(BaseModel):
    """자연어 질의 → 탐지/분할용 개념 프롬프트.

    파이프라인 ②. 예: "포트홀 찾아줘" -> concepts=["pothole"]
    이 결과가 SAM3/YOLOE/Grounded-SAM 어댑터의 입력이 된다.
    """

    original_query: str = Field(description="사용자가 입력한 원문 질의")
    language: str = Field(default="unknown", description="감지된 언어 (ko/en/...)")
    concepts: List[str] = Field(
        default_factory=list, description="탐지/분할용 영문 개념 명사구 목록"
    )
    normalized: str = Field(default="", description="정규화/번역된 질의(선택)")


class VLMResult(BaseModel):
    """이미지 1장에 대해 저장하는 최종 레코드."""

    image_id: str
    image_path: str
    width: Optional[int] = None
    height: Optional[int] = None
    analysis: ImageAnalysis
    backend: str
    model: str
    latency_ms: Optional[float] = None
    created_at: str
    query: Optional[ConceptPrompt] = None

    def to_metadata(self) -> dict:
        """제안서 데이터 구조의 `metadata` 슬롯 형태로 변환."""
        a = self.analysis
        return {
            "image_id": self.image_id,
            "scene_type": a.scene_type,
            "surface_material": a.surface_material,
            "weather": a.weather,
            "time_of_day": a.time_of_day,
            "overall_condition": a.overall_condition,
            "damage_present": a.damage_present,
            "damage_types": sorted({d.type for d in a.damages}),
            "source": "auto",
            "model": self.model,
            "backend": self.backend,
            "created_at": self.created_at,
        }


class PromptLog(BaseModel):
    """재현성·평가용 프롬프트 로그 (제안서 prompt_log)."""

    timestamp: str
    backend: str
    model: str
    kind: str  # analyze | vqa | parse_query
    image_id: Optional[str] = None
    query: Optional[str] = None
    response_summary: Optional[str] = None
    latency_ms: Optional[float] = None
