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
| `gemini` | Google Gemini 비전 API, 구조화 출력 (**무료 티어**) | `GEMINI_API_KEY`(무료) + `pip install google-genai` |
| `anthropic` | Claude 비전 API, 구조화 출력 (유료, 고품질) | `ANTHROPIC_API_KEY`(유료) + `pip install anthropic` |
| `openai` | GPT-4o 비전 API (유료) | `OPENAI_API_KEY` + `pip install openai` |
| `qwen` | 로컬 Qwen2.5-VL (**GPU 확보 시 활성화**) | `requirements-local.txt`, NVIDIA GPU 권장 |

현재 이 PC에는 NVIDIA GPU가 없어 로컬 Qwen은 느립니다. **`mock`으로 구조를
검증**하고, 실제 추론은 **무료 티어인 `gemini`** 로 전환하는 것을 권장합니다.
(`anthropic`/`openai`는 유료. Claude 구독과 API 과금은 별개입니다.) Qwen 코드는
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

## 데모 UI

### 커스텀 웹앱 (권장) — `python -m vlm web`
다크 테마·탐지/히스토리 2페이지·심각도 분포·COCO 내보내기를 갖춘 정식 웹 데모
(claude.ai/design 디자인 구현). 백엔드 FastAPI가 기존 탐지 파이프라인을 재사용.

```powershell
pip install fastapi uvicorn python-multipart
python -m vlm web            # http://127.0.0.1:8000
```
업로드/샘플 → 자연어 질의 → 모델(yolo/gemini) → 신뢰도 슬라이더 → 탐지 → 오버레이·표·심각도·COCO 다운로드. 히스토리는 브라우저(localStorage)에 저장.

### 간편 데모 — `python -m vlm demo`
```powershell
pip install gradio
python -m vlm demo            # Gradio, http://127.0.0.1:7860
```
탐지 모델: `yolo`(파인튜닝·로컬·무료) / `gemini`(무료 API·키 필요) / `mock`(키 없이).

## 실제 추론으로 전환

### 권장: Gemini (무료 티어)

키 발급(결제수단 불필요): https://aistudio.google.com → "Get API key".

```powershell
pip install google-genai
copy .env.example .env          # .env 에 GEMINI_API_KEY 입력

python -m vlm --backend gemini doctor --ping                 # 키/연결 검증
python -m vlm --backend gemini analyze data/images/road_pothole_01.png
python -m vlm --backend gemini batch data/images             # 토큰 합계 출력(무료=비용 0)
```

또는 `config/default.yaml` 의 `backend: gemini` 로 영구 전환.

### 유료 고품질: Anthropic Claude

> Claude **구독**(Pro/Max)과 Anthropic **API**는 과금이 별개입니다. 구독이 있어도
> API는 무료가 아니며, https://console.anthropic.com 에서 선불 크레딧 충전이 필요합니다.

```powershell
pip install anthropic
# .env 에 ANTHROPIC_API_KEY 입력
python -m vlm --backend anthropic doctor --ping
python -m vlm --backend anthropic batch data/images          # 토큰·예상 비용 합계 출력
```

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

## 탐지 + 분할 (detect, ③)

자연어 질의 → 개념 프롬프트(②) → **bbox 탐지 + 오버레이**. 제안서 앵커 시나리오
"포트홀 영역을 찾아줘"가 그대로 동작합니다. (Gemini 2.5 탐지, GPU 불필요)

```powershell
python -m vlm detect data/images/real/Pothole_000.jpg "포트홀 찾아줘" --detector gemini
#   --concepts pothole,crack  (개념 직접 지정), --detector mock (키 없이 테스트)
```
→ `data/outputs/<id>_detect.json`(박스·라벨·신뢰도) + `<id>_detected.png`(오버레이).
`--masks` 로 분할 mask도 시도(채색 오버레이). 단 **무료 gemini-2.5-flash는 mask 반환이 불안정**해
실패 시 박스로 폴백합니다 — 정밀 mask는 gemini-2.5-pro(유료)나 SAM 계열(후속 DetectorBackend) 권장.
탐지 백엔드는 `DetectorBackend` 추상화라 Roboflow/YOLOE/SAM으로 교체 가능.

## 자동 라벨 생성 (label, ④)

탐지 결과를 **COCO/YOLO** 포맷으로 변환 (confidence 필터 포함). 검수 도구(CVAT/Label Studio, ⑤)로 import.

```powershell
python -m vlm label data/images/real "포트홀 찾아줘" --detector gemini --min-conf 0.3 --delay 13
#   --format coco|yolo|both, --limit N
```
→ `data/outputs/labels_export/` 에 `annotations_coco.json` + `labels/*.txt` + `classes.txt`.

## 검수 (review, ⑤)

자동 라벨을 사람이 검수하는 루프. 외부 도구(CVAT/Label Studio)는 **COCO 라운드트립**으로 연결합니다.

```powershell
# 1) 트리아지: confidence로 자동확정/검수대상 분리 → 자동확정률
python -m vlm review prep --coco data/outputs/labels_export/annotations_coco.json --confirm-thr 0.5

# 2) CVAT(Docker) 또는 Label Studio(pip install label-studio)에 위 COCO를 import
#    → 사람이 수정/확정 → 다시 COCO로 export

# 3) 수정 전/후 비교 → 수정 비율·precision·recall (제안서 평가지표)
python -m vlm review report --auto <auto_coco.json> --reviewed <corrected_coco.json>
```
→ `review_manifest.json`(검수 상태/검수자/시각) + `review_report.json`(자동확정률·수정 비율).
확정 라벨은 ⑥ 파인튜닝 데이터로 환류됩니다.

## 파인튜닝 (⑥, Colab 무료 GPU)

Gemini는 닫힌 API라 학습 불가 → 오픈 모델(YOLO)을 **무료 Colab GPU**에서 파인튜닝하고
전/후 mAP를 비교합니다. 학습 데이터는 Roboflow 라벨 / 우리가 ④에서 만든 라벨.

1. [`notebooks/finetune_yolo_colab.ipynb`](notebooks/finetune_yolo_colab.ipynb) 를 Colab에 업로드 (런타임 = GPU)
2. Pothole zip 업로드 → 학습 → `mAP@0.5` → `best.pt` 다운로드
3. `models/pothole_yolo.pt` 로 저장 → 우리 파이프라인에 끼움:
```powershell
pip install ultralytics
python -m vlm detect <img> --detector yolo
python -m vlm compare data/images/real --backends gemini,yolo   # 파인튜닝 YOLO vs Gemini
```
→ `compare`로 모델 간/파인튜닝 전후 성능을 정량 비교 (제안서 "파인튜닝 전후 성능표").

## 탐지기 평가 (eval) — GT 대비 precision/recall

YOLO 데이터셋의 정답(GT) 라벨 기준으로 탐지기를 평가합니다(IoU≥0.5 매칭).

```powershell
python -m vlm eval data/roboflow/Pothole.v1-raw.yolov8/test --detector yolo
python -m vlm eval data/roboflow/Pothole.v1-raw.yolov8/test --detector gemini --limit 15 --delay 13
```

실측 결과(test 15장) — **파인튜닝 효과 증명**:

| 탐지기 | precision | recall |
|---|---|---|
| 파인튜닝 YOLO (mAP@0.5 0.79) | 59% | **74%** |
| Gemini zero-shot | 26% | 23% |
| COCO 사전학습 | ≈ 0 (pothole 클래스 없음) | — |

## 모델 비교 (compare)

같은 이미지를 여러 백엔드로 돌려 **정량 비교표**(일치율·속도·비용·탐지율)를 만듭니다.
제안서의 "모델 비교·검증 결과" 산출물.

```powershell
python -m vlm compare data/images/real --backends mock,gemini --delay 13
#   --limit N  (앞 N장만), --name <파일명>
```
→ 콘솔 요약표 + `data/outputs/comparison.json`(이미지별 상세 + 집계 + 백엔드 간 일치율).
전부 포트홀인 데이터에선 `dmg%`/`pot%` 가 사실상 탐지(recall) 지표가 됩니다.

실제 데이터 확보(Roboflow 등)는 [data/README.md](data/README.md) 참고.

## 구조

```
vlm/
  schemas.py     # 데이터 구조 (ImageAnalysis / VLMResult / ConceptPrompt / PromptLog)
  prompts.py     # 도메인 프롬프트 (도로 손상)
  keywords.py    # 한글→영문 개념 매핑 + 규칙 기반 질의 파서
  config.py      # 설정 로딩 + 백엔드 팩토리
  pipeline.py    # 오케스트레이터 (단건/배치, 시간측정, prompt_log)
  compare.py     # VLM 백엔드 비교 (정량 비교표)
  eval.py        # 탐지기 GT 평가 (precision/recall) — 모델 비교·검증
  labels.py      # 탐지 → COCO/YOLO 라벨 변환 (④)
  review.py      # 검수 트리아지/수정 비교 (⑤, 자동확정률·수정 비율)
  overlay.py     # 탐지 결과 이미지 오버레이 (박스 + 분할 mask)
  demo.py        # Gradio 간편 데모
  web.py         # 커스텀 웹앱 백엔드 (FastAPI) — 디자인 구현
  cli.py         # CLI (analyze/batch/query/vqa/detect/label/review/eval/compare/demo/web/doctor)
web/   index.html · style.css · app.js   # 커스텀 웹 프론트엔드(다크 테마, 탐지/히스토리)
  backends/      # VLM: mock / gemini / anthropic / openai / qwen
  detectors/     # 탐지: mock / gemini / yolo(파인튜닝 ⑥)
config/default.yaml
notebooks/finetune_yolo_colab.ipynb   # ⑥ Colab 파인튜닝
models/   # 파인튜닝 가중치(.pt, git 미추적)
scripts/  make_sample_images.py · fetch_roboflow.py
tests/  test_mock · test_compare · test_detect · test_labels · test_review · test_api_backends
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
