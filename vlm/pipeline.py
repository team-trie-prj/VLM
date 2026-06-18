"""VLM 처리 오케스트레이터.

이미지(+선택적 자연어 질의) → 구조화 메타데이터(VLMResult) 생성.
배치 처리 시 처리시간을 측정해 제안서의 "라벨링 시간 단축률" 근거 자료를 만든다.
프롬프트 로그(prompt_log.jsonl)를 남겨 재현성을 확보한다.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .backends.base import VLMBackend, image_size, is_image_file
from .schemas import ConceptPrompt, PromptLog, VLMResult


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class VLMProcessor:
    def __init__(self, backend: VLMBackend, output_dir: str | Path = "data/outputs"):
        self.backend = backend
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.output_dir / "prompt_log.jsonl"

    # --- 단건 처리 -----------------------------------------------------------
    def process(self, image_path: str | Path, query: Optional[str] = None) -> VLMResult:
        image_path = str(image_path)
        image_id = Path(image_path).stem
        w, h = image_size(image_path)

        t0 = time.perf_counter()
        analysis = self.backend.analyze_image(image_path)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        # parse_query 가 last_usage 를 덮어쓰기 전에 analyze 사용량을 캡처
        usage = getattr(self.backend, "last_usage", None)

        concept: Optional[ConceptPrompt] = None
        if query:
            concept = self.backend.parse_query(query)

        result = VLMResult(
            image_id=image_id,
            image_path=image_path,
            width=w,
            height=h,
            analysis=analysis,
            backend=self.backend.name,
            model=self.backend.model,
            latency_ms=round(latency_ms, 1),
            usage=usage,
            created_at=_now_iso(),
            query=concept,
        )
        self._log(
            PromptLog(
                timestamp=result.created_at,
                backend=self.backend.name,
                model=self.backend.model,
                kind="analyze",
                image_id=image_id,
                query=query,
                response_summary=analysis.caption[:120],
                latency_ms=result.latency_ms,
                input_tokens=usage.input_tokens if usage else None,
                output_tokens=usage.output_tokens if usage else None,
            )
        )
        return result

    def vqa(self, image_path: str | Path, question: str) -> str:
        answer = self.backend.vqa(str(image_path), question)
        self._log(
            PromptLog(
                timestamp=_now_iso(),
                backend=self.backend.name,
                model=self.backend.model,
                kind="vqa",
                image_id=Path(image_path).stem,
                query=question,
                response_summary=answer[:120],
            )
        )
        return answer

    def parse_query(self, query: str) -> ConceptPrompt:
        concept = self.backend.parse_query(query)
        self._log(
            PromptLog(
                timestamp=_now_iso(),
                backend=self.backend.name,
                model=self.backend.model,
                kind="parse_query",
                query=query,
                response_summary=", ".join(concept.concepts),
            )
        )
        return concept

    # --- 배치 처리 -----------------------------------------------------------
    def process_dir(
        self, image_dir: str | Path, query: Optional[str] = None, delay: float = 0.0
    ) -> List[VLMResult]:
        paths = sorted(p for p in Path(image_dir).iterdir() if is_image_file(p))
        results: List[VLMResult] = []
        for i, p in enumerate(paths):
            if delay and i:  # API 분당 한도(예: Gemini 무료 5 RPM) 대비 페이싱
                time.sleep(delay)
            try:
                results.append(self.process(p, query=query))
            except Exception as e:  # 한 장 실패가 배치 전체를 막지 않도록
                print(f"[warn] {p.name} 처리 실패: {e}")
        return results

    # --- 저장 ---------------------------------------------------------------
    def save_result(self, result: VLMResult) -> Path:
        out = self.output_dir / f"{result.image_id}.json"
        out.write_text(
            json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return out

    def save_dataset(self, results: List[VLMResult], name: str = "dataset") -> Path:
        """전체 결과를 하나의 데이터셋 JSON으로 저장 (images/metadata 형태)."""
        out = self.output_dir / f"{name}.json"
        tin = sum(r.usage.input_tokens for r in results if r.usage)
        tout = sum(r.usage.output_tokens for r in results if r.usage)
        costs = [r.usage.estimated_cost_usd for r in results if r.usage and r.usage.estimated_cost_usd]
        payload = {
            "created_at": _now_iso(),
            "backend": self.backend.name,
            "model": self.backend.model,
            "count": len(results),
            "totals": {
                "input_tokens": tin,
                "output_tokens": tout,
                "estimated_cost_usd": round(sum(costs), 6) if costs else None,
            },
            "images": [r.model_dump() for r in results],
            "metadata": [r.to_metadata() for r in results],
        }
        out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return out

    def _log(self, entry: PromptLog) -> None:
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.model_dump(), ensure_ascii=False) + "\n")
