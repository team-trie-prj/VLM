"""교체 가능한 탐지·분할 백엔드 (파이프라인 ③).

자연어 질의의 개념 프롬프트(②)를 받아 bbox(+mask)를 산출한다.
  - mock   : 키/GPU 불필요. 파이프라인/오버레이 검증용 결정론 더미
  - gemini : Gemini 2.5 의 box_2d 탐지 (무료, GPU 불필요)
나중에 Roboflow 호스팅, 로컬 YOLOE/SAM 등을 같은 인터페이스로 추가 가능.
"""

from .base import DetectorBackend

__all__ = ["DetectorBackend"]
