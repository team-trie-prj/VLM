"""교체 가능한 VLM 백엔드들.

공통 인터페이스 `VLMBackend`를 구현하며, config로 골라 끼운다.
  - mock      : 키/GPU 불필요. 파이프라인 동작 검증용 (오늘 바로 실행)
  - anthropic : Claude 비전 API (구조화 출력) — 1순위 실행 백엔드
  - openai    : GPT-4o 비전 API
  - qwen      : 로컬 Qwen2.5-VL (GPU/CPU) — GPU 확보 시 활성화
"""

from .base import VLMBackend

__all__ = ["VLMBackend"]
