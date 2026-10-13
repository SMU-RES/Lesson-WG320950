# pyright: reportMissingTypeStubs=false
from __future__ import annotations

import importlib
from decimal import Decimal
from pathlib import Path
from typing import Any

from invoice_app.models import InvoiceFields, QrDecodeResult


class OpenCvQrDecoder:
    def decode(self, crop_path: Path) -> QrDecodeResult:
        try:
            cv2 = importlib.import_module("cv2")
        except ModuleNotFoundError as exc:
            raise RuntimeError("缺少 opencv-python，真实二维码解析请安装：uv sync --extra real") from exc
        image = cv2.imread(str(crop_path))
        if image is None:
            raise ValueError(f"无法读取二维码裁剪图：{crop_path}")
        detector = cv2.QRCodeDetector()
        raw_text = _detect_from_crop_variants(cv2, detector, image)
        if not raw_text:
            raise ValueError("二维码裁剪图解析失败")
        return QrDecodeResult(raw_text, parse_invoice_qr(raw_text), crop_path)


def _detect_from_crop_variants(cv2: Any, detector: Any, image: object) -> str:
    raw_text, _points, _straight = detector.detectAndDecode(image)
    if raw_text:
        return str(raw_text)
    resize = getattr(cv2, "resize")
    interpolation = getattr(cv2, "INTER_NEAREST", 0)
    for scale in (2, 3, 4, 6, 8, 10, 12):
        scaled = resize(image, None, fx=scale, fy=scale, interpolation=interpolation)
        raw_text, _points, _straight = detector.detectAndDecode(scaled)
        if raw_text:
            return str(raw_text)
    return ""


def parse_invoice_qr(raw_text: str) -> InvoiceFields:
    parts = [part.strip() for part in raw_text.replace("，", ",").replace("|", ",").split(",")]
    if len(parts) >= 6 and parts[0] == "01":
        return InvoiceFields(invoice_code=parts[2], invoice_number=parts[3], issue_date=_format_qr_date(parts[5]), total_amount=_decimal_or_zero(parts[4]))
    code = parts[0] if len(parts) > 0 else ""
    number = parts[1] if len(parts) > 1 else ""
    return InvoiceFields(invoice_code=code, invoice_number=number)


def _decimal_or_zero(value: str) -> Decimal:
    try:
        return Decimal(value)
    except Exception:
        return Decimal("0.00")


def _format_qr_date(value: str) -> str:
    digits = "".join(char for char in value if char.isdigit())
    if len(digits) == 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return value
