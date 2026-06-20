"""검수(human-in-the-loop, 파이프라인 ⑤).

- triage: auto COCO를 confidence로 분류(자동확정 vs 검수대상) → 자동확정률
- compare_coco: auto COCO와 사람이 고친 COCO를 IoU 매칭 비교 → 수정 비율 등

CVAT/Label Studio 연동은 COCO 라운드트립으로 한다(둘 다 COCO import/export 지원):
  label(④) → review prep → [CVAT/LS에서 사람 검수] → review report.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from .schemas import ReviewItem, ReviewManifest, ReviewReport


def triage(coco: dict, confirm_thr: float = 0.5) -> ReviewManifest:
    """score >= 임계값이면 자동확정(confirmed), 아니면 검수대상(pending)."""
    items: List[ReviewItem] = []
    auto = 0
    for a in coco.get("annotations", []):
        score = float(a.get("score", 0.0))
        confirmed = score >= confirm_thr
        auto += int(confirmed)
        bbox = [int(round(v)) for v in a.get("bbox", [0, 0, 0, 0])]
        items.append(ReviewItem(
            ann_id=a.get("id", 0), image_id=a.get("image_id", 0),
            category_id=a.get("category_id", 0), bbox=bbox, score=round(score, 3),
            status="confirmed" if confirmed else "pending",
        ))
    total = len(items)
    return ReviewManifest(
        confirm_threshold=confirm_thr, total=total, auto_confirmed=auto,
        needs_review=total - auto,
        auto_confirm_rate=round(auto / total, 3) if total else 0.0,
        items=items,
    )


def _iou(a: List[float], b: List[float]) -> float:
    """COCO bbox [x, y, w, h] IoU."""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _by_image(coco: dict) -> dict:
    m: dict = {}
    for a in coco.get("annotations", []):
        m.setdefault(a.get("image_id"), []).append(a)
    return m


def compare_coco(auto: dict, reviewed: dict, iou_thr: float = 0.5) -> ReviewReport:
    """auto와 사람-수정 COCO를 IoU로 매칭해 수정량을 집계.

    매칭됨=유지, auto에만=검수자가 삭제(오탐), reviewed에만=검수자가 추가(미탐).
    reviewed를 정답으로 보면 precision/recall도 산출된다.
    """
    A, R = _by_image(auto), _by_image(reviewed)
    n_auto = sum(len(v) for v in A.values())
    n_rev = sum(len(v) for v in R.values())
    matched = removed = 0
    used_total = 0
    for iid in set(A) | set(R):
        al, rl = A.get(iid, []), R.get(iid, [])
        used = set()
        for a in al:
            best, bestj = -1.0, -1
            for j, r in enumerate(rl):
                if j in used or r.get("category_id") != a.get("category_id"):
                    continue
                v = _iou(a["bbox"], r["bbox"])
                if v > best:
                    best, bestj = v, j
            if best >= iou_thr and bestj >= 0:
                matched += 1
                used.add(bestj)
            else:
                removed += 1
        used_total += len(used)
    added = n_rev - used_total
    corr = removed + added
    return ReviewReport(
        iou_threshold=iou_thr, auto_count=n_auto, reviewed_count=n_rev,
        matched=matched, removed_by_reviewer=removed, added_by_reviewer=added,
        correction_rate=round(corr / n_auto, 3) if n_auto else 0.0,
        precision_vs_human=round(matched / n_auto, 3) if n_auto else 0.0,
        recall_vs_human=round(matched / n_rev, 3) if n_rev else 0.0,
    )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
