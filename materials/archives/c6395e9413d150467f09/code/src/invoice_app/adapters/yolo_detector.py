# pyright: reportMissingTypeStubs=false, reportUnnecessaryCast=false
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, cast

from PIL import Image, ImageDraw

from invoice_app.models import QrDetection


class MissingDependencyError(RuntimeError):
    pass


class YoloQrDetector:
    def __init__(self, weights_path: Path, conf: float = 0.25, imgsz: int = 1600, device: str = "0") -> None:
        self.weights_path = weights_path
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def detect_and_crop(self, image_path: Path, artifact_dir: Path) -> QrDetection:
        try:
            yolo_class = getattr(importlib.import_module("ultralytics"), "YOLO")
        except ModuleNotFoundError as exc:
            raise MissingDependencyError("缺少 ultralytics，真实 YOLO 模式请先安装依赖：uv sync --extra real") from exc
        if not self.weights_path.exists():
            raise FileNotFoundError(f"YOLO 权重不存在：{self.weights_path}。请将权重放到该路径，或设置 INVOICE_APP_YOLO_WEIGHTS 指向训练好的 best.pt")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model = yolo_class(str(self.weights_path))
        results = list(model.predict(source=str(image_path), conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False))
        if not results:
            raise ValueError("YOLO 未返回检测结果")
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            raise ValueError("YOLO 未检测到二维码")
        xyxy_values = cast(Any, boxes.xyxy)
        conf_values = cast(Any, boxes.conf)
        best_index = 0
        best_conf = float(conf_values[0]) if len(conf_values) else 0.0
        for index, confidence in enumerate(conf_values):
            parsed = float(confidence)
            if parsed > best_conf:
                best_index = index
                best_conf = parsed
        raw_box = [int(round(float(value))) for value in xyxy_values[best_index]]
        return crop_and_annotate(image_path, raw_box, best_conf, artifact_dir)


def crop_and_annotate(image_path: Path, bbox_xyxy: list[int], confidence: float, artifact_dir: Path, padding: int = 8) -> QrDetection:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    left, top, right, bottom = bbox_xyxy
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(width, right + padding)
    bottom = min(height, bottom + padding)
    if right <= left or bottom <= top:
        raise ValueError("YOLO 二维码框无效，无法裁剪")
    crop_path = artifact_dir / "qr_crop.png"
    annotated_path = artifact_dir / "yolo_annotated.png"
    image.crop((left, top, right, bottom)).save(crop_path)
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    draw.rectangle((left, top, right, bottom), outline="red", width=4)
    draw.text((left, max(0, top - 18)), f"qr_code {confidence:.2f}", fill="red")
    annotated.save(annotated_path)
    return QrDetection((left, top, right, bottom), confidence, crop_path, annotated_path)
