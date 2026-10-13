from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .annotation import annotate_image, save_annotation
from .gpu import GPUUnavailableError
from .ocr_engine import PaddleOCREngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GPU-only PaddleOCR on a container image.")
    parser.add_argument("image", type=Path, help="Image path")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--device", default="gpu:0", help="Paddle device, default: gpu:0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        engine = PaddleOCREngine(device=args.device)
        result = engine.infer(args.image)
        annotated = annotate_image(args.image, result.ocr_texts, result.container_numbers, result.size_type_codes)
        result.annotated_image_path = str(save_annotation(annotated, args.out, args.image.name))
    except GPUUnavailableError as exc:
        print(f"GPU unavailable: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
