from __future__ import annotations

import argparse
from pathlib import Path

from container_ocr.trudi import prepare_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Copy a deterministic subset of local TRUDI images.")
    parser.add_argument("--source", required=True, type=Path, help="Extracted TRUDI dataset directory")
    parser.add_argument("--limit", type=int, default=20, help="Number of images to copy")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic shuffle seed")
    parser.add_argument("--out", type=Path, default=Path("data/trudi_samples"), help="Output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    copied = prepare_samples(args.source, limit=args.limit, seed=args.seed, out_dir=args.out)
    for path in copied:
        print(path)
    print(f"Copied {len(copied)} image(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
