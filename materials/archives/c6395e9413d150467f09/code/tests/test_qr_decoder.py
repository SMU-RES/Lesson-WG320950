# pyright: reportMissingTypeStubs=false
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path
from types import ModuleType

import pytest

from invoice_app.adapters.qr_decoder import OpenCvQrDecoder, parse_invoice_qr


REAL_QR_TEXT = "01,10,012001800311,33207675,46.62,20190508,76939056883466677916,E1BD,"


class FakeQrCodeDetector:
    def detectAndDecode(self, image: object) -> tuple[str, None, None]:
        _ = image
        return REAL_QR_TEXT, None, None


def test_parse_invoice_qr_uses_chinese_vat_qr_positions() -> None:
    fields = parse_invoice_qr(REAL_QR_TEXT)

    assert fields.invoice_code == "012001800311"
    assert fields.invoice_number == "33207675"
    assert fields.issue_date == "2019-05-08"
    assert fields.amount_without_tax == Decimal("46.62")
    assert fields.total_amount == Decimal("0.00")


def test_opencv_decoder_reads_only_crop_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = ModuleType("cv2")
    read_paths: list[str] = []

    def imread(path: str) -> object:
        read_paths.append(path)
        return object()

    setattr(module, "imread", imread)
    setattr(module, "QRCodeDetector", FakeQrCodeDetector)
    monkeypatch.setitem(sys.modules, "cv2", module)
    crop = tmp_path / "qr_crop.png"
    crop.write_bytes(b"fake image")

    result = OpenCvQrDecoder().decode(crop)

    assert read_paths == [str(crop)]
    assert result.decoded_from == crop
    assert result.fields.invoice_number == "33207675"


def test_opencv_decoder_upscales_crop_before_failing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = ModuleType("cv2")
    resize_calls: list[tuple[int, int]] = []

    class RetryQrCodeDetector:
        calls = 0

        def detectAndDecode(self, image: object) -> tuple[str, None, None]:
            _ = image
            self.calls += 1
            if self.calls == 1:
                return "", None, None
            return REAL_QR_TEXT, None, None

    def resize(image: object, _size: None, *, fx: int, fy: int, interpolation: int) -> object:
        _ = image
        _ = _size
        _ = interpolation
        resize_calls.append((fx, fy))
        return object()

    setattr(module, "imread", lambda path: object())
    setattr(module, "QRCodeDetector", RetryQrCodeDetector)
    setattr(module, "resize", resize)
    setattr(module, "INTER_NEAREST", 0)
    monkeypatch.setitem(sys.modules, "cv2", module)
    crop = tmp_path / "qr_crop.png"
    crop.write_bytes(b"fake image")

    result = OpenCvQrDecoder().decode(crop)

    assert resize_calls == [(2, 2)]
    assert result.fields.invoice_number == "33207675"
