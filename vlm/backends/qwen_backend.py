"""로컬 Qwen2.5-VL 백엔드 — GPU 확보 시 활성화하는 백엔드.

지금은 코드만 준비해 두고, NVIDIA GPU 서버가 생기면 config에서
`backend: qwen` 한 줄로 켤 수 있다. (CPU에서도 동작은 하지만 매우 느림)

필요 패키지(requirements-local.txt):
  torch, torchvision, transformers, accelerate, qwen-vl-utils, pillow
"""

from __future__ import annotations

import json
import re
from typing import Optional

from ..keywords import keyword_parse_query
from ..prompts import (
    ANALYZE_INSTRUCTION,
    ANALYZE_SYSTEM,
    PARSE_QUERY_SYSTEM,
    VQA_SYSTEM,
)
from ..schemas import ConceptPrompt, ImageAnalysis
from .base import VLMBackend

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def _extract_json(text: str) -> str:
    """모델 출력에서 첫 번째 JSON 오브젝트를 추출."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _schema_hint(model_cls) -> str:
    return (
        "Return ONLY a JSON object matching this schema, no extra text:\n"
        + json.dumps(model_cls.model_json_schema(), ensure_ascii=False)
    )


class QwenVLM(VLMBackend):
    name = "qwen"

    def __init__(self, model: str = DEFAULT_MODEL, max_new_tokens: int = 1024):
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "로컬 Qwen 백엔드에는 추가 패키지가 필요합니다: "
                "pip install -r requirements-local.txt"
            ) from e

        self.model = model
        self.max_new_tokens = max_new_tokens
        self._torch = torch

        device_map = "auto" if torch.cuda.is_available() else None
        dtype = "auto"
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model, torch_dtype=dtype, device_map=device_map
        )
        if device_map is None:
            self._model = self._model.to("cpu")
        self._processor = AutoProcessor.from_pretrained(model)

    def _generate(self, messages: list) -> str:
        from qwen_vl_utils import process_vision_info

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with self._torch.no_grad():
            generated = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def analyze_image(
        self, image_path: str, *, instruction: Optional[str] = None
    ) -> ImageAnalysis:
        messages = [
            {"role": "system", "content": ANALYZE_SYSTEM + "\n\n" + _schema_hint(ImageAnalysis)},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": instruction or ANALYZE_INSTRUCTION},
                ],
            },
        ]
        raw = self._generate(messages)
        return ImageAnalysis.model_validate_json(_extract_json(raw))

    def vqa(self, image_path: str, question: str) -> str:
        messages = [
            {"role": "system", "content": VQA_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": question},
                ],
            },
        ]
        return self._generate(messages).strip()

    def parse_query(self, query: str) -> ConceptPrompt:
        try:
            messages = [
                {"role": "system", "content": PARSE_QUERY_SYSTEM + "\n\n" + _schema_hint(ConceptPrompt)},
                {"role": "user", "content": [{"type": "text", "text": query}]},
            ]
            raw = self._generate(messages)
            result = ConceptPrompt.model_validate_json(_extract_json(raw))
            result.original_query = query
            return result
        except Exception:
            return keyword_parse_query(query)
