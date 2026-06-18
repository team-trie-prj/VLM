"""Roboflow Universe 공개 도로 손상 데이터셋 다운로드 -> data/images/real/ 샘플 적재.

준비:
  1) https://roboflow.com 무료 가입 -> Settings -> API Key 복사 -> .env 에 ROBOFLOW_API_KEY
  2) https://universe.roboflow.com 에서 도로손상/포트홀 데이터셋 선택
     -> "Download Dataset" -> 형식 선택 -> 코드 스니펫의 workspace/project/version 확인
     (또는 데이터셋 URL을 --url 로 그대로 넘겨도 workspace/project를 파싱)

사용 예:
  python scripts/fetch_roboflow.py --workspace <ws> --project <proj> --version <N>
  python scripts/fetch_roboflow.py --url https://universe.roboflow.com/<ws>/<proj>/dataset/3
  # .env 에 ROBOFLOW_WORKSPACE/PROJECT/VERSION 를 넣어두면 인자 없이 실행 가능

자세한 절차는 data/README.md 참고.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REAL_DIR = ROOT / "data" / "images" / "real"
RAW_DIR = ROOT / "data" / "roboflow"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def _safe_stdout() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except Exception:
            pass


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except Exception:
        pass


def parse_url(url: str):
    """Universe URL에서 (workspace, project, version) 추출. 없으면 (None, None, None)."""
    if not url:
        return None, None, None
    m = re.search(
        r"universe\.roboflow\.com/([^/]+)/([^/?#]+)(?:/(?:dataset|model)/(\d+))?", url
    )
    if not m:
        return None, None, None
    ver = int(m.group(3)) if m.group(3) else None
    return m.group(1), m.group(2), ver


def _populate(src: Path, name: str, sample: int) -> int:
    """src(추출/다운로드 폴더)에서 이미지를 모아 data/images/real/ 로 복사."""
    imgs = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not imgs:
        print("[경고] 폴더에서 이미지를 찾지 못했습니다.")
        return 1
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    pick = imgs if sample <= 0 else imgs[:sample]
    for i, p in enumerate(pick):
        shutil.copy2(p, REAL_DIR / f"{name}_{i:03d}{p.suffix.lower()}")
    print(f"real/ 에 {len(pick)}장 복사 (전체 {len(imgs)}장): {REAL_DIR}")
    print("다음: python -m vlm --backend gemini batch data/images/real")
    return 0


def main() -> int:
    _safe_stdout()
    _load_env()

    ap = argparse.ArgumentParser(description="Roboflow 공개 데이터셋 다운로드")
    ap.add_argument("--zip", help="로컬 Roboflow zip 경로 (SDK 대신 직접 받은 zip 사용)")
    ap.add_argument("--url", help="Roboflow Universe 데이터셋 URL (workspace/project 파싱)")
    ap.add_argument("--workspace", default=os.getenv("ROBOFLOW_WORKSPACE"))
    ap.add_argument("--project", default=os.getenv("ROBOFLOW_PROJECT"))
    ap.add_argument("--version", type=int,
                    default=(int(os.getenv("ROBOFLOW_VERSION")) if os.getenv("ROBOFLOW_VERSION") else None))
    ap.add_argument("--format", default=os.getenv("ROBOFLOW_FORMAT", "yolov8"))
    ap.add_argument("--sample", type=int, default=20,
                    help="real/ 로 복사할 이미지 수 (0=전체)")
    args = ap.parse_args()

    # 로컬 zip 경로: SDK/키 없이 바로 처리
    if args.zip:
        import zipfile

        zp = Path(args.zip)
        if not zp.exists():
            print(f"[실패] zip 파일이 없습니다: {zp}")
            return 1
        dest = RAW_DIR / zp.stem
        dest.mkdir(parents=True, exist_ok=True)
        print(f"압축 해제: {zp.name} -> {dest}")
        with zipfile.ZipFile(zp) as z:
            z.extractall(dest)
        return _populate(dest, zp.stem.split(".")[0], args.sample)

    # URL이 주면 명시 인자보다 우선순위 낮게 보충
    u_ws, u_proj, u_ver = parse_url(args.url or "")
    workspace = args.workspace or u_ws
    project = args.project or u_proj
    version = args.version or u_ver

    key = os.getenv("ROBOFLOW_API_KEY")
    if not key:
        print("[실패] ROBOFLOW_API_KEY 없음.")
        print("  https://roboflow.com 가입 -> Settings -> API Key -> .env 에 ROBOFLOW_API_KEY 입력")
        return 1
    if not (workspace and project and version):
        print("[실패] 데이터셋 좌표가 필요합니다 (workspace/project/version).")
        print("  Roboflow Universe 데이터셋 -> Download Dataset -> 코드 스니펫에서")
        print('  rf.workspace("X").project("Y").version(N) 의 X/Y/N 을 사용하세요.')
        print("  예: python scripts/fetch_roboflow.py --workspace X --project Y --version N")
        print("  절차: data/README.md")
        return 1

    try:
        from roboflow import Roboflow
    except ImportError:
        print("[실패] roboflow 패키지 필요: pip install roboflow")
        return 1

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    loc = RAW_DIR / project

    print(f"다운로드: workspace={workspace} project={project} version={version} format={args.format}")
    rf = Roboflow(api_key=key)
    rf.workspace(workspace).project(project).version(version).download(
        args.format, location=str(loc)
    )
    print(f"원본 데이터셋 저장: {loc}")

    return _populate(loc, f"rf_{project}", args.sample)


if __name__ == "__main__":
    raise SystemExit(main())
