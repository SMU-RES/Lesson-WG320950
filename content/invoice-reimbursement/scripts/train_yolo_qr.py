from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, cast
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO QR detector and report the generated best.pt path.")
    parser.add_argument("--data", required=True, help="Ultralytics data.yaml for the QR dataset.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO model, for example yolov8n.pt.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="Ultralytics device, for example 0, cuda:0, or cpu.")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--project", default="runtime/models/yolo/runs")
    parser.add_argument("--name", default="qr")
    return parser.parse_args(argv)


def train(args: argparse.Namespace) -> Path:
    try:
        yolo_class = getattr(importlib.import_module("ultralytics"), "YOLO")
    except ModuleNotFoundError as exc:
        raise RuntimeError("缺少 ultralytics，请先运行：uv sync --extra real") from exc

    model = yolo_class(args.model)
    result = cast(Any, model).train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
    )
    save_dir = Path(getattr(result, "save_dir", Path(args.project) / args.name))
    return save_dir / "weights" / "best.pt"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    best_path = train(args)
    print(f"YOLO QR 权重训练完成后应使用：{best_path}")
    print("把该路径写入 INVOICE_APP_YOLO_WEIGHTS，或复制到 runtime/models/yolo/best.pt。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
