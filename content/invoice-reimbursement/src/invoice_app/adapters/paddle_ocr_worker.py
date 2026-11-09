from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Sequence, cast


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PaddleOCR in an isolated worker process.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="gpu:0")
    args = parser.parse_args(argv)

    try:
        with contextlib.redirect_stdout(sys.stderr):
            lines = recognize_lines(Path(args.image), str(args.device))
    except Exception as exc:
        print(f"PaddleOCR worker failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"lines": lines}, ensure_ascii=False))
    return 0


def recognize_lines(image_path: Path, device: str) -> list[str]:
    _hide_torch_discovery()
    paddle = importlib.import_module("paddle")
    normalized_device = device.strip() or "cpu"
    if _requires_cuda(normalized_device) and not bool(paddle.device.is_compiled_with_cuda()):
        raise RuntimeError(f"Paddle device {normalized_device} requires CUDA, but installed PaddlePaddle is CPU-only")
    paddle.device.set_device(normalized_device)
    paddleocr = importlib.import_module("paddleocr")
    ocr = getattr(paddleocr, "PaddleOCR")(
        lang="ch",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        enable_mkldnn=False,
        device=normalized_device,
    )
    raw_result = ocr.predict(str(image_path))
    return _extract_lines(raw_result)


def _hide_torch_discovery() -> None:
    original_find_spec = importlib.util.find_spec

    def find_spec_without_torch(name: str, package: str | None = None) -> Any:
        if name == "torch" or name.startswith("torch."):
            return None
        return original_find_spec(name, package)

    importlib.util.find_spec = find_spec_without_torch


def _requires_cuda(device: str) -> bool:
    normalized = device.strip().lower()
    return normalized not in {"", "cpu"}


def _extract_lines(raw_result: object) -> list[str]:
    lines: list[str] = []
    for page in cast(Any, raw_result) or []:
        get_value = getattr(page, "get", None)
        if not callable(get_value):
            continue
        rec_texts = cast(Any, get_value("rec_texts", [])) or []
        for text in rec_texts:
            lines.append(str(text))
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
