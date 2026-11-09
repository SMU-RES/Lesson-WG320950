from __future__ import annotations

from pathlib import Path
from typing import Protocol
from uuid import uuid4

from .models import ApplicantInfo, InvoiceFields, OcrResult, ProcessingResult, QrDecodeResult, QrDetection, ReimbursementForm
from .validation import build_unique_key, normalize_money, validate_consistency


class QrDetector(Protocol):
    def detect_and_crop(self, image_path: Path, artifact_dir: Path) -> QrDetection:
        ...


class QrDecoder(Protocol):
    def decode(self, crop_path: Path) -> QrDecodeResult:
        ...


class OcrEngine(Protocol):
    def recognize(self, image_path: Path) -> OcrResult:
        ...


class LlmExtractor(Protocol):
    def extract(self, ocr_text: str) -> InvoiceFields:
        ...


class InvoicePipeline:
    def __init__(self, detector: QrDetector, decoder: QrDecoder, ocr: OcrEngine, llm: LlmExtractor, artifact_dir: Path, applicant: ApplicantInfo | None = None) -> None:
        self.detector: QrDetector = detector
        self.decoder: QrDecoder = decoder
        self.ocr: OcrEngine = ocr
        self.llm: LlmExtractor = llm
        self.artifact_dir: Path = artifact_dir
        self.applicant: ApplicantInfo = applicant or ApplicantInfo()

    def process(self, source_image: Path) -> ProcessingResult:
        job_id = uuid4().hex
        job_dir = self.artifact_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        detection = self.detector.detect_and_crop(source_image, job_dir)
        qr_result = self.decoder.decode(detection.crop_path)
        ocr_result = self.ocr.recognize(source_image)
        llm_fields = self.llm.extract(ocr_result.text)
        validation = validate_consistency(qr_result.fields, llm_fields)
        form = ReimbursementForm(
            applicant=self.applicant,
            invoice=llm_fields,
            expense_type="办公费",
            purpose="课程项目发票报销",
            amount=normalize_money(llm_fields.total_amount),
            unique_key=build_unique_key(llm_fields),
        )
        return ProcessingResult(job_id, source_image, detection, qr_result, ocr_result, llm_fields, validation, form)
