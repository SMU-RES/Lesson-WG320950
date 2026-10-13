from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from PIL import Image, ImageDraw

from invoice_app.models import InvoiceFields, OcrResult, QrDecodeResult, QrDetection


REAL_SAMPLE_QR_TEXT = "01,10,012001800311,33207675,46.62,20190508,76939056883466677916,E1BD,"


def sample_fields() -> InvoiceFields:
    return InvoiceFields(
        invoice_code="012001800311",
        invoice_number="33207675",
        issue_date="2019-05-08",
        buyer_name="个人",
        seller_name="天津瑞佳讯贸易有限公司",
        amount_without_tax=Decimal("46.62"),
        tax_amount=Decimal("6.08"),
        total_amount=Decimal("52.70"),
        invoice_type="增值税电子普通发票",
    )


class SampleQrDetector:
    def detect_and_crop(self, image_path: Path, artifact_dir: Path) -> QrDetection:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        crop_path = artifact_dir / "qr_crop.png"
        annotated_path = artifact_dir / "yolo_annotated.png"
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        left = max(0, int(width * 0.094))
        top = max(0, int(height * 0.078))
        right = min(width, int(width * 0.182))
        bottom = min(height, int(height * 0.216))
        image.crop((left, top, right, bottom)).save(crop_path)
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        draw.rectangle((left, top, right, bottom), outline=(151, 27, 27), width=4)
        draw.text((left, max(0, top - 18)), "qr_code 0.98", fill=(151, 27, 27))
        annotated.save(annotated_path)
        return QrDetection((left, top, right, bottom), 0.98, crop_path, annotated_path)


class SampleQrDecoder:
    def decode(self, crop_path: Path) -> QrDecodeResult:
        return QrDecodeResult(REAL_SAMPLE_QR_TEXT, sample_fields(), crop_path)


class SampleOcrEngine:
    def recognize(self, image_path: Path) -> OcrResult:
        _ = image_path
        lines = [
            "增值税电子普通发票",
            "发票代码 012001800311",
            "发票号码 33207675",
            "开票日期 2019-05-08",
            "购买方 个人",
            "销售方 天津瑞佳讯贸易有限公司",
            "金额 46.62",
            "税额 6.08",
            "价税合计 52.70",
        ]
        return OcrResult("\n".join(lines), lines)


class SampleLlmExtractor:
    def extract(self, ocr_text: str) -> InvoiceFields:
        _ = ocr_text
        return sample_fields()
