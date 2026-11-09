from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from .infer import run_segmentation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="本地 YOLO 实例分割演示")
    _ = parser.add_argument("--image", required=True, help="输入图片路径")
    _ = parser.add_argument("--output-dir", default="outputs", help="输出目录")
    _ = parser.add_argument("--model", default="yolo11n-seg.pt", help="模型名称")
    _ = parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值，0 到 1")
    _ = parser.add_argument("--device", default="cpu", help="推理设备，默认 cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    image = cast(str, args.image)
    output_dir = cast(str, args.output_dir)
    model = cast(str, args.model)
    conf = cast(float, args.conf)
    device = cast(str, args.device)

    try:
        output_path, metadata = run_segmentation(
            image_path=Path(image),
            output_dir=Path(output_dir),
            model_name=model,
            conf=conf,
            device=device,
        )
    except Exception as exc:
        print(f"推理失败：{exc}", file=sys.stderr)
        return 1

    print("分割完成")
    print(f"输出图片：{output_path}")
    print(f"实例数量：{metadata['num_instances']}")
    instances = metadata.get("instances", [])
    if not isinstance(instances, list):
        instances = []
    for idx, item in enumerate(instances, start=1):
        if not isinstance(item, Mapping):
            continue
        label = item.get("class_name") or "unknown"
        score = item.get("confidence")
        conf_text = f"{score:.3f}" if isinstance(score, float) else "n/a"
        print(f"  {idx}. 类别：{label}，置信度：{conf_text}")
    return 0

__all__ = ["build_parser", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
