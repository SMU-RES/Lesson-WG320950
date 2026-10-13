# pyright: reportMissingTypeStubs=false
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from PIL import Image

from invoice_app.models import InvoiceFields, OcrResult, QrDecodeResult, QrDetection
from invoice_app.pipeline import InvoicePipeline


class FakeDetector:
    def detect_and_crop(self, image_path: Path, artifact_dir: Path) -> QrDetection:
        _ = image_path
        crop_path = artifact_dir / "qr_crop.png"
        Image.new("RGB", (64, 64), "white").save(crop_path)
        return QrDetection((10, 10, 74, 74), 0.98, crop_path, artifact_dir / "annotated.png")


class RecordingDecoder:
    def __init__(self) -> None:
        self.paths: list[Path] = []

    def decode(self, crop_path: Path) -> QrDecodeResult:
        self.paths.append(crop_path)
        fields = _fields()
        return QrDecodeResult("QR|12345678|106.00", fields, crop_path)


class FakeOcr:
    def recognize(self, image_path: Path) -> OcrResult:
        _ = image_path
        return OcrResult("发票号码 12345678\n价税合计 106.00", ["发票号码 12345678", "价税合计 106.00"])


class FakeLlm:
    def extract(self, ocr_text: str) -> InvoiceFields:
        _ = ocr_text
        return _fields()


def _fields() -> InvoiceFields:
    return InvoiceFields(
        invoice_code="044032100111",
        invoice_number="12345678",
        issue_date="2026-05-20",
        buyer_name="上海海事大学",
        seller_name="测试科技有限公司",
        amount_without_tax=Decimal("100.00"),
        tax_amount=Decimal("6.00"),
        total_amount=Decimal("106.00"),
        invoice_type="增值税电子普通发票",
    )


def test_pipeline_decodes_only_cropped_qr_image(tmp_path: Path) -> None:
    source = tmp_path / "invoice.png"
    Image.new("RGB", (300, 200), "white").save(source)
    decoder = RecordingDecoder()
    pipeline = InvoicePipeline(FakeDetector(), decoder, FakeOcr(), FakeLlm(), tmp_path / "artifacts")

    result = pipeline.process(source)

    assert result.validation.status == "pass"
    assert decoder.paths == [result.qr_detection.crop_path], "QR decoder must receive only YOLO crop artifact"
    assert decoder.paths[0] != source, "QR decoder must never receive the original full invoice image"
