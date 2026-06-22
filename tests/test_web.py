"""웹 백엔드 테스트 — severity 매핑 + 앱 구성(오프라인)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlm.web import _severity  # noqa: E402 (stdlib만 의존, fastapi 불필요)


def test_severity_mapping():
    assert _severity(0.9) == "severe"
    assert _severity(0.7) == "severe"
    assert _severity(0.5) == "moderate"
    assert _severity(0.4) == "moderate"
    assert _severity(0.2) == "minor"


def test_build_app_if_fastapi():
    if importlib.util.find_spec("fastapi") is None:
        print("skip - fastapi 미설치")
        return
    from vlm.web import build_app
    app = build_app()
    routes = {getattr(r, "path", "") for r in app.routes}
    assert "/api/detect" in routes
    assert "/api/samples" in routes


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok - {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
