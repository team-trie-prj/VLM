"""모델(백엔드) 비교·검증 — 같은 이미지를 여러 백엔드로 분석해 정량 비교.

제안서의 "모델 비교·검증 결과" 산출물. 백엔드 간 일치율·속도·비용을 표/JSON으로 만든다.
도메인 사전지식(예: Pothole 데이터셋은 전부 포트홀 포함)을 ground truth로 보면
damage_present_rate / pothole_rate 는 사실상 탐지(recall) 지표가 된다.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .backends.base import is_image_file
from .config import build_backend


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jaccard(a: set, b: set) -> float:
    """두 집합의 자카드 유사도. 둘 다 비면 1.0(완전 일치)로 본다."""
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def compare(
    image_dir, backend_names: List[str], delay: float = 0.0, limit: int = 0
) -> dict:
    paths = sorted(p for p in Path(image_dir).iterdir() if is_image_file(p))
    if limit > 0:
        paths = paths[:limit]
    backends = {n: build_backend({"backend": n}) for n in backend_names}

    per_image = []
    agg = {
        n: {"model": backends[n].model, "ok": 0, "lat": [], "cost": 0.0,
            "dp": 0, "ndmg": [], "pothole": 0}
        for n in backend_names
    }

    for p in paths:
        rec = {"image_id": p.stem, "results": {}}
        for n in backend_names:
            b = backends[n]
            try:
                t0 = time.perf_counter()
                a = b.analyze_image(str(p))
                lat = (time.perf_counter() - t0) * 1000.0
            except Exception as e:
                rec["results"][n] = {"error": str(e)[:150]}
                continue
            u = getattr(b, "last_usage", None)
            types = sorted({d.type for d in a.damages})
            rec["results"][n] = {
                "damage_present": a.damage_present,
                "types": types,
                "scene": a.scene_type,
                "condition": a.overall_condition,
                "latency_ms": round(lat, 1),
                "cost_usd": (u.estimated_cost_usd if u else None),
            }
            s = agg[n]
            s["ok"] += 1
            s["lat"].append(lat)
            s["dp"] += int(a.damage_present)
            s["ndmg"].append(len(a.damages))
            if u and u.estimated_cost_usd:
                s["cost"] += u.estimated_cost_usd
            if "pothole" in types:
                s["pothole"] += 1
            if delay and n not in ("mock", "yolo"):  # 로컬 백엔드는 페이싱 불필요
                time.sleep(delay)
        per_image.append(rec)

    per_backend = {}
    for n, s in agg.items():
        ok = s["ok"]
        per_backend[n] = {
            "model": s["model"],
            "images_ok": ok,
            "avg_latency_ms": round(sum(s["lat"]) / ok, 1) if ok else None,
            "total_cost_usd": round(s["cost"], 6),
            "damage_present_rate": round(s["dp"] / ok, 3) if ok else None,
            "avg_damages": round(sum(s["ndmg"]) / ok, 2) if ok else None,
            "pothole_rate": round(s["pothole"] / ok, 3) if ok else None,
        }

    pairwise = []
    for i in range(len(backend_names)):
        for j in range(i + 1, len(backend_names)):
            a, b = backend_names[i], backend_names[j]
            dp = sc = both = 0
            jac = []
            for rec in per_image:
                ra, rb = rec["results"].get(a), rec["results"].get(b)
                if not ra or not rb or "error" in ra or "error" in rb:
                    continue
                both += 1
                dp += int(ra["damage_present"] == rb["damage_present"])
                sc += int(ra["scene"] == rb["scene"])
                jac.append(_jaccard(set(ra["types"]), set(rb["types"])))
            if both:
                pairwise.append({
                    "a": a, "b": b, "compared": both,
                    "damage_present_agreement": round(dp / both, 3),
                    "scene_agreement": round(sc / both, 3),
                    "mean_type_jaccard": round(sum(jac) / both, 3),
                })

    return {
        "created_at": _now_iso(),
        "image_dir": str(image_dir),
        "backends": backend_names,
        "count": len(paths),
        "per_backend": per_backend,
        "pairwise": pairwise,
        "per_image": per_image,
    }


def format_report(report: dict) -> str:
    """콘솔용 요약표 (cp949 안전: ASCII 구두점만)."""
    lines = [
        f"비교 대상: {report['count']}장  /  백엔드: {', '.join(report['backends'])}",
        "",
    ]
    hdr = (f"{'backend':<10}{'model':<24}{'imgs':>5}{'avg_ms':>9}"
           f"{'cost$':>9}{'dmg%':>7}{'pot%':>7}{'avg_d':>7}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for n in report["backends"]:
        s = report["per_backend"][n]
        lines.append(
            f"{n:<10}{str(s['model'])[:23]:<24}{s['images_ok']:>5}"
            f"{(s['avg_latency_ms'] or 0):>9.0f}"
            f"{(s['total_cost_usd'] or 0):>9.4f}"
            f"{(s['damage_present_rate'] or 0) * 100:>6.0f}%"
            f"{(s['pothole_rate'] or 0) * 100:>6.0f}%"
            f"{(s['avg_damages'] or 0):>7.2f}"
        )
    if report["pairwise"]:
        lines.append("")
        lines.append("백엔드 간 일치율 (damage_present / scene / type-Jaccard):")
        for pw in report["pairwise"]:
            lines.append(
                f"  {pw['a']} vs {pw['b']} (n={pw['compared']}): "
                f"dmg {pw['damage_present_agreement'] * 100:.0f}% / "
                f"scene {pw['scene_agreement'] * 100:.0f}% / "
                f"type {pw['mean_type_jaccard']:.2f}"
            )
    return "\n".join(lines)
