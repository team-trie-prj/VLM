# 데이터 가이드

## 폴더 구조
```
data/
  images/            # batch 처리 대상 (합성 샘플 .png 가 여기에)
  images/real/       # 실제 도로 이미지 (Roboflow 등에서 확보, git 미추적)
  roboflow/          # Roboflow 원본 다운로드 (git 미추적)
  outputs/           # 분석 결과 JSON / prompt_log (git 미추적)
```
> `data/images/*`, `data/roboflow/`, `data/outputs/*` 는 `.gitignore`로 커밋되지 않습니다.
> 코드/스크립트/문서만 추적하고 대용량 데이터는 각자 내려받습니다.

---

## 권장: Roboflow Universe (무료, 빠름)

수백~수천 장의 라벨링된 실제 도로 손상 이미지를 받을 수 있고, 일부는 mask(분할)까지 포함합니다.

**1) 무료 API 키 발급**
- https://roboflow.com 가입 → 우상단 계정 → **Settings → API Key** 복사
- `.env` 에 `ROBOFLOW_API_KEY=...` 입력

**2) 데이터셋 선택 + 좌표 확보**
- https://universe.roboflow.com/search?q=class:pothole 에서 데이터셋 선택
  (예: "road potholes and cracks"(분할 mask), "Pothole"(665장, ODbL), "crack and pothole")
- 데이터셋 페이지 → **Download Dataset** → 형식 선택(`yolov8` 권장) → 코드 스니펫의
  `rf.workspace("X").project("Y").version(N)` 에서 **X / Y / N** 확인

**3) 다운로드 실행**
```powershell
pip install roboflow
python scripts/fetch_roboflow.py --workspace X --project Y --version N
#   또는 데이터셋 URL로:
python scripts/fetch_roboflow.py --url https://universe.roboflow.com/X/Y/dataset/N
#   --sample 20  (real/ 로 복사할 장수, 0=전체)
```
→ `data/roboflow/Y/` 에 원본, `data/images/real/` 에 샘플 N장이 복사됩니다.

**이미 zip으로 받았다면** (Roboflow 사이트에서 직접 다운로드한 경우, 키 불필요):
```powershell
python scripts/fetch_roboflow.py --zip "C:\path\to\dataset.yolov8.zip" --sample 20
```

**4) VLM으로 검증**
```powershell
python -m vlm --backend gemini batch data/images/real --delay 13
```
> Gemini 무료 티어는 **분당 5회(5 RPM)** 제한이라, 여러 장 배치 시 `--delay 13`(이미지 간 13초)을
> 주면 한도를 넘지 않습니다. (백엔드에도 429 자동 재시도가 있어 일부는 그냥 기다렸다 처리됩니다.)

`.env` 에 `ROBOFLOW_WORKSPACE/PROJECT/VERSION` 을 넣어두면 인자 없이 실행할 수 있습니다.

---

## 대안

- **RDD2022** (제안서 표준 벤치마크, 6개국 47,420장): https://github.com/sekilab/RoadDamageDetector
  공식 다운로드 링크 상태를 확인 후 수동으로 받아 `data/images/real/` 에 풀어 넣으세요. 수 GB.
- **AI Hub** (한국 도메인 본격화 시): https://aihub.or.kr — 다운로드에 **IRB 심의 결과서·연구계획서·소속 증빙** 등
  승인 서류가 필요합니다. 지금 단계(솔로 프로토타입)에서는 보류, 추후 진행 권장.
- **직접 수집**: 스마트폰/거리뷰 이미지를 `data/images/real/` 에 직접 넣어도 됩니다.

---

## 라이선스 · 개인정보 주의
- 데이터셋별 라이선스를 준수하세요(예: Roboflow Pothole = ODbL v1.0 → 출처 표기/공유조건).
- 도로/CCTV 이미지의 **차량 번호판·얼굴**은 개인정보입니다. 공개·배포 시 **비식별화(블러)** 를 검토하세요.
