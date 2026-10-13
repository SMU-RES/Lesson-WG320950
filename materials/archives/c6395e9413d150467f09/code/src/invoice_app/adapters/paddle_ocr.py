# pyright: reportMissingTypeStubs=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnnecessaryCast=false
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from invoice_app.models import OcrResult


class PaddleOcrEngine:
    def __init__(self, device: str = "gpu:0", timeout_seconds: int = 600) -> None:
        self.device = device
        self.timeout_seconds = timeout_seconds

    def recognize(self, image_path: Path) -> OcrResult:
        command = [
            sys.executable,
            "-m",
            "invoice_app.adapters.paddle_ocr_worker",
            "--image",
            str(image_path),
            "--device",
            self.device,
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"PaddleOCR GPU worker timed out after {self.timeout_seconds} seconds") from exc
        if completed.returncode != 0:
            detail = _safe_tail(completed.stderr) or _safe_tail(completed.stdout) or "no worker output"
            raise RuntimeError(f"PaddleOCR GPU worker failed with exit code {completed.returncode}: {detail}")
        try:
            payload: Any = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("PaddleOCR GPU worker returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("PaddleOCR GPU worker returned non-object JSON")
        lines_value = payload.get("lines")
        if not isinstance(lines_value, list) or not all(isinstance(line, str) for line in lines_value):
            raise RuntimeError("PaddleOCR GPU worker returned invalid OCR lines")
        lines = list(lines_value)
        return OcrResult("\n".join(lines), lines)


def _safe_tail(value: str, limit: int = 800) -> str:
    stripped = value.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[-limit:]
