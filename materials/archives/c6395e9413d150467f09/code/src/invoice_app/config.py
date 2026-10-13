from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    mode: str = "sample"
    upload_dir: Path = Path("runtime/uploads")
    artifact_dir: Path = Path("runtime/artifacts")
    database_path: Path = Path("runtime/db/invoices.sqlite3")
    yolo_weights: Path = Path("runtime/models/yolo/best.pt")
    yolo_conf: float = 0.25
    yolo_imgsz: int = 1600
    yolo_device: str = "0"
    paddle_device: str = "gpu:0"
    secret_key: str = "dev"

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            mode=os.getenv("INVOICE_APP_MODE", "sample"),
            upload_dir=Path(os.getenv("INVOICE_APP_UPLOAD_DIR", "runtime/uploads")),
            artifact_dir=Path(os.getenv("INVOICE_APP_ARTIFACT_DIR", "runtime/artifacts")),
            database_path=Path(os.getenv("INVOICE_APP_DATABASE_PATH", "runtime/db/invoices.sqlite3")),
            yolo_weights=Path(os.getenv("INVOICE_APP_YOLO_WEIGHTS", "runtime/models/yolo/best.pt")),
            yolo_conf=float(os.getenv("INVOICE_APP_YOLO_CONF", "0.25")),
            yolo_imgsz=int(os.getenv("INVOICE_APP_YOLO_IMGSZ", "1600")),
            yolo_device=os.getenv("INVOICE_APP_YOLO_DEVICE", "0"),
            paddle_device=os.getenv("INVOICE_APP_PADDLE_DEVICE", "gpu:0"),
            secret_key=os.getenv("INVOICE_APP_SECRET_KEY", "dev"),
        )
