from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .gpu import require_gpu
from .parsing import extract_container_numbers, extract_size_type_codes
from .types import Box, InferenceResult, OCRText, path_to_str


OCRFactory = Callable[[], Any]


class PaddleOCREngine:
    def __init__(
        self,
        *,
        device: str = "gpu:0",
        lang: str = "en",
        ocr_factory: OCRFactory | None = None,
        require_gpu_check: bool = True,
    ) -> None:
        self.device = device
        self.lang = lang
        self._ocr_factory = ocr_factory
        self._ocr: Any | None = None
        if require_gpu_check:
            require_gpu(device=device)

    def _create_ocr(self) -> Any:
        if self._ocr_factory is not None:
            return self._ocr_factory()

        module = import_module("paddleocr")
        paddle_ocr = getattr(module, "PaddleOCR")

        return paddle_ocr(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            engine="paddle",
            lang=self.lang,
        )

    @property
    def ocr(self) -> Any:
        if self._ocr is None:
            self._ocr = self._create_ocr()
        return self._ocr

    def recognize(self, image: str | Path | Image.Image) -> tuple[list[OCRText], Any]:
        payload: Any
        if isinstance(image, (str, Path)):
            payload = str(image)
        else:
            payload = np.asarray(image.convert("RGB"))

        if hasattr(self.ocr, "predict"):
            raw = self.ocr.predict(payload)
        elif hasattr(self.ocr, "ocr"):
            raw = self.ocr.ocr(payload, cls=True)
        else:
            raise RuntimeError("PaddleOCR backend does not expose predict() or ocr().")

        return normalize_ocr_result(raw), raw

    def infer(self, image: str | Path | Image.Image) -> InferenceResult:
        ocr_texts, raw = self.recognize(image)
        return InferenceResult(
            image_path=path_to_str(image) if isinstance(image, (str, Path)) else None,
            ocr_texts=ocr_texts,
            container_numbers=extract_container_numbers(ocr_texts),
            size_type_codes=extract_size_type_codes(ocr_texts),
            raw_result=raw,
        )


def _as_box(points: Any) -> Box | None:
    if points is None:
        return None
    box: Box = []
    for point in points:
        if len(point) < 2:
            return None
        box.append((float(point[0]), float(point[1])))
    return box or None


def _from_result_dict(item: dict[str, Any]) -> list[OCRText]:
    texts = item.get("rec_texts") or item.get("texts") or []
    scores = item.get("rec_scores") or item.get("scores") or []
    boxes = item.get("rec_polys") or item.get("dt_polys") or item.get("boxes") or []
    output: list[OCRText] = []
    for index, text in enumerate(texts):
        score = float(scores[index]) if index < len(scores) else None
        box = _as_box(boxes[index]) if index < len(boxes) else None
        output.append(OCRText(text=str(text), confidence=score, box=box))
    return output


def _to_plain_result(item: Any) -> Any:
    if isinstance(item, dict):
        return item
    json_attr = getattr(item, "json", None)
    if isinstance(json_attr, dict):
        return json_attr
    if callable(json_attr):
        return json_attr()
    return item


def _parse_v2_detection(item: Any) -> OCRText | None:
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return None
    box = _as_box(item[0])
    rec = item[1]
    if isinstance(rec, (list, tuple)) and rec:
        text = str(rec[0])
        confidence = float(rec[1]) if len(rec) > 1 else None
        return OCRText(text=text, confidence=confidence, box=box)
    return None


def normalize_ocr_result(raw: Any) -> list[OCRText]:
    plain = _to_plain_result(raw)
    if isinstance(plain, dict):
        return _from_result_dict(plain)

    output: list[OCRText] = []
    if not isinstance(plain, list):
        return output

    for page in plain:
        page_plain = _to_plain_result(page)
        if isinstance(page_plain, dict):
            output.extend(_from_result_dict(page_plain))
            continue
        if isinstance(page_plain, list):
            for item in page_plain:
                parsed = _parse_v2_detection(item)
                if parsed is not None:
                    output.append(parsed)
            parsed_page = _parse_v2_detection(page_plain)
            if parsed_page is not None:
                output.append(parsed_page)

    return output
