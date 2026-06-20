"""멀티모달 VLM 이미지 이해 모듈.

자연어 질의 + 이미지를 입력받아 캡션/속성/손상유형을 추정하고
구조화된 메타데이터(JSON)를 생성한다.

백엔드는 교체 가능(Anthropic / OpenAI / Qwen2.5-VL / Mock)하며,
지엔소프트 프로젝트 제안서의 "VLM 이미지 이해" 부문(파이프라인 ①·②)에 해당한다.
"""

from .schemas import (
    ConceptPrompt,
    DamageInstance,
    Detection,
    DetectionResult,
    ImageAnalysis,
    PromptLog,
    Usage,
    VLMResult,
)

__all__ = [
    "ConceptPrompt",
    "DamageInstance",
    "Detection",
    "DetectionResult",
    "ImageAnalysis",
    "PromptLog",
    "Usage",
    "VLMResult",
]

__version__ = "0.1.0"
