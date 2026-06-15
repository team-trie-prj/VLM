# VLM 이미지 이해 모듈

지엔소프트 프로젝트 제안서 중 **"VLM 이미지 이해"** 부문(파이프라인 ①·②) 구현.

> 자연어 질의 + 도로 이미지 → **캡션·속성(도로종류/날씨/손상유형) 추정 + 구조화 메타데이터(JSON) 생성**
> → 자연어 질의를 탐지·분할 단계(SAM3/YOLOE)로 넘길 **개념 프롬프트**로 변환

## 핵심 설계: 교체 가능한 백엔드

`VLMBackend` 공통 인터페이스 아래 백엔드를 **갈아끼웁니다**. 제안서의
"모델 조합 비교·검증" 취지 그대로, 같은 파이프라인에서 백엔드만 바꿔 비교합니다.

| backend | 설명 | 필요 조건 |
|---|---|---|
| `mock` | 키/GPU 없이 동작하는 결정론적 더미 (**기본값**) | 없음 |
| `anthropic` | Claude 비전 API, 구조화 출력 (1순위 실행) | `ANTHROPIC_API_KEY` + `pip install anthropic` |
| `openai` | GPT-4o 비전 API | `OPENAI_API_KEY` + `pip install openai` |
| `qwen` | 로컬 Qwen2.5-VL (**GPU 확보 시 활성화**) | `requirements-local.txt`, NVIDIA GPU 권장 |

현재 이 PC에는 NVIDIA GPU가 없어 로컬 Qwen은 느립니다. **`mock`으로 구조를
검증**하고, API 키가 생기면 `anthropic` 으로 전환하는 것을 권장합니다. Qwen 코드는
이미 들어 있으므로 GPU 서버에서 `backend: qwen` 한 줄로 켤 수 있습니다.

## 빠른 시작 (키 없이 지금 바로)

```powershell
# 1) 핵심 의존성 설치 (GPU 불필요)
pip install -r requirements.txt

# 2) 합성 샘플 이미지 생성
python scripts/make_sample_images.py

# 3) 이미지 1장 분석 → 메타데이터 JSON
python -m vlm analyze data/images/road_pothole_01.png

# 4) 디렉터리 일괄 처리 + 처리시간 측정 (라벨링 시간 단축률 근거)
python -m vlm batch data/images --query "포트홀 찾아줘"

# 5) 자연어 질의 → 탐지 개념 프롬프트 (SAM3/YOLOE 입력)
python -m vlm query "도로 균열이랑 패임 표시해줘"

# 6) 설정/백엔드 확인
python -m vlm info
```

## API 키가 생기면 (실제 추론으로 전환)

키 발급: https://console.anthropic.com → Billing 크레딧 충전 → API Keys → Create Key.

```powershell
pip install anthropic
copy .env.example .env          # .env 에 ANTHROPIC_API_KEY 입력

# 1) 환경 진단 + 키 검증 (실제 1회 호출)
python -m vlm --backend anthropic doctor --ping

# 2) 실제 추론
python -m vlm --backend anthropic analyze data/images/road_pothole_01.png
python -m vlm --backend anthropic batch data/images   # 토큰·예상 비용 합계 출력
```

또는 `config/default.yaml` 의 `backend: anthropic` 으로 영구 전환.

- **비용 추적**: 각 결과 JSON에 `usage`(토큰)와 `estimated_cost_usd`가 기록되고,
  배치는 합계를 출력한다 (제안서의 비용/효율 지표 근거). 단가표는 `vlm/pricing.py`.
- **`doctor`**: 패키지/키/백엔드 진단. `--ping` 없이는 호출 없이 점검만, `--ping`은 실제 키 검증.
- **비용 절감**: 대량 라벨링은 `--model claude-haiku-4-5` 또는 `claude-sonnet-4-6` 권장.

## 출력 예시 (메타데이터 JSON)

```jsonc
{
  "image_id": "road_pothole_01",
  "analysis": {
    "caption": "...",
    "scene_type": "urban_road",
    "surface_material": "asphalt",
    "weather": "clear",
    "time_of_day": "day",
    "damage_present": true,
    "damages": [{"type": "pothole", "severity": "high", "confidence": 0.82}],
    "overall_condition": "poor"
  },
  "backend": "anthropic", "model": "claude-opus-4-8",
  "latency_ms": 1840.5, "created_at": "2026-06-15T..."
}
```

## 구조

```
vlm/
  schemas.py     # 데이터 구조 (ImageAnalysis / VLMResult / ConceptPrompt / PromptLog)
  prompts.py     # 도메인 프롬프트 (도로 손상)
  keywords.py    # 한글→영문 개념 매핑 + 규칙 기반 질의 파서
  config.py      # 설정 로딩 + 백엔드 팩토리
  pipeline.py    # 오케스트레이터 (단건/배치, 시간측정, prompt_log)
  cli.py         # CLI
  backends/      # mock / anthropic / openai / qwen
config/default.yaml
scripts/make_sample_images.py
tests/test_mock.py
```

## 테스트

```powershell
python tests/test_mock.py     # 또는: python -m pytest -q
```

## 다음 단계 (파이프라인 ③ 이후)

이 모듈의 `ConceptPrompt` 출력이 탐지·분할 단계(SAM3 / YOLOE-26 / Grounded-SAM)의
입력이 됩니다. 같은 백엔드 추상화 패턴으로 `DetectorBackend` 를 추가하면 제안서의
전체 반자동 라벨링 파이프라인으로 확장됩니다.

```
[이미지+질의] → (VLM: 이 모듈) → meta.json + concepts
              → (탐지/분할: SAM3/YOLOE) → bbox+mask
              → COCO/YOLO 변환 → CVAT 검수 → 파인튜닝
```
