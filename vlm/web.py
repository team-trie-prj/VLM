"""커스텀 웹 데모 (FastAPI) — claude.ai/design '도로 손상 탐지 데모' 구현.

정적 프론트엔드(web/) + /api/detect(기존 탐지 파이프라인 재사용). GPU 불필요.
실행: python -m vlm web   (uvicorn, 기본 http://127.0.0.1:8000)
필요 패키지: fastapi, uvicorn, python-multipart
"""

from __future__ import annotations

import base64
import io
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
SEVERITY_COLOR = {"severe": (239, 68, 68), "moderate": (245, 158, 11), "minor": (20, 184, 166)}


def _severity(conf: float) -> str:
    return "severe" if conf >= 0.7 else "moderate" if conf >= 0.4 else "minor"


def _sample_dir() -> Path:
    for d in (ROOT / "data" / "images" / "real", ROOT / "data" / "images"):
        if d.is_dir() and any(p.suffix.lower() in _IMG_EXT for p in d.iterdir()):
            return d
    return ROOT / "data" / "images" / "real"


def _overlay_b64(image_path: str, dets: list[dict]) -> str:
    from PIL import Image, ImageDraw

    im = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(im)
    for det in dets:
        x0, y0, x1, y1 = det["box"]
        c = SEVERITY_COLOR.get(det["severity"], (59, 130, 246))
        d.rectangle([x0, y0, x1, y1], outline=c, width=3)
        cap = f"{det['label']} {det['confidence']:.2f}"
        d.text((x0 + 3, y0 - 12 if y0 >= 12 else y0 + 2), cap, fill=c)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_app():
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles

    from .backends.base import image_size
    from .config import build_detector, load_config
    from .keywords import keyword_parse_query
    from .labels import to_coco
    from .schemas import DetectionResult

    load_config()  # .env 로드 (GEMINI_API_KEY 등)
    app = FastAPI(title="도로 손상 탐지 데모")
    sample_dir = _sample_dir()

    @app.get("/api/samples")
    def samples():
        names = sorted(p.name for p in sample_dir.iterdir() if p.suffix.lower() in _IMG_EXT) \
            if sample_dir.is_dir() else []
        return {"samples": names}

    @app.post("/api/detect")
    async def detect(
        query: str = Form(""),
        detector: str = Form("yolo"),
        conf: float = Form(0.25),
        sample: str = Form(""),
        image: UploadFile = File(None),
    ):
        tmp_path = None
        if sample:
            cand = sample_dir / sample
            if not cand.exists():
                return JSONResponse({"error": "샘플을 찾을 수 없습니다."}, status_code=400)
            path = str(cand)
        elif image is not None:
            data = await image.read()
            suffix = Path(image.filename or "upload.png").suffix or ".png"
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tf.write(data)
            tf.close()
            path = tf.name
            tmp_path = path
        else:
            return JSONResponse({"error": "이미지를 올려주세요."}, status_code=400)

        try:
            concepts = ["pothole"]
            if query.strip():
                cp = keyword_parse_query(query)
                concepts = cp.concepts or [query.strip()]
            cfg = {"detector": detector}
            if detector == "yolo":
                cfg["yolo_conf"] = float(conf)
            det = build_detector(cfg)
            raw = [d for d in det.detect(path, concepts) if d.confidence >= float(conf)]

            dets = [{"label": d.label, "box": d.box, "confidence": round(d.confidence, 3),
                     "severity": _severity(d.confidence)} for d in raw]
            w, h = image_size(path)
            result = DetectionResult(
                image_id=Path(path).stem, image_path=path, width=w, height=h,
                concepts=concepts, detections=raw, backend=det.name, model=det.model,
                created_at=datetime.now(timezone.utc).isoformat())
            dist = {"severe": 0, "moderate": 0, "minor": 0}
            for d in dets:
                dist[d["severity"]] += 1
            avg = round(sum(d["confidence"] for d in dets) / len(dets), 3) if dets else 0.0
            return {
                "detections": dets,
                "overlay": _overlay_b64(path, dets) if dets else None,
                "stats": {"count": len(dets), "severe": dist["severe"],
                          "avg_conf": avg, "model": det.model, "backend": det.name},
                "severity_dist": dist, "concepts": concepts, "coco": to_coco([result]),
            }
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        finally:
            if tmp_path:
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass

    if sample_dir.is_dir():
        app.mount("/samples", StaticFiles(directory=str(sample_dir)), name="samples")
    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    return app


def main(host: str = "127.0.0.1", port: int = 8000) -> int:
    try:
        import uvicorn
    except ImportError:
        print("[실패] 웹 데모에는 fastapi/uvicorn 필요: pip install fastapi uvicorn python-multipart")
        return 1
    print(f"웹 데모 실행: http://{host}:{port}  (Ctrl+C 종료)")
    uvicorn.run(build_app(), host=host, port=port)
    return 0


if __name__ == "__main__":
    main()
