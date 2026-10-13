from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="智能发票识别与自动报销填单系统")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser("process", help="处理样例发票文本并生成报销结果")
    _ = process_parser.add_argument("--input-dir", type=Path, required=True, help="输入发票文本目录")
    _ = process_parser.add_argument("--applicant-file", type=Path, required=True, help="个人信息文件路径")
    _ = process_parser.add_argument("--output-dir", type=Path, required=True, help="输出目录")

    return parser


def main(argv: list[str] | None = None) -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        _ = reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(argv)

    command = cast(str, getattr(args, "command"))
    if command == "process":
        result = run_pipeline(
            input_dir=cast(Path, getattr(args, "input_dir")),
            applicant_file=cast(Path, getattr(args, "applicant_file")),
            output_dir=cast(Path, getattr(args, "output_dir")),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
