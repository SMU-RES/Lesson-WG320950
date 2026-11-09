from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from .models import InvoiceRecord
from .parser import parse_invoice_file
from .personal_info import load_applicant_info
from .reimbursement import export_reimbursement_workbook


class ArtifactPaths(TypedDict):
    parsed_json: str
    workbook: str
    summary_json: str


class PipelineResult(TypedDict):
    summary: dict[str, object]
    artifacts: ArtifactPaths


def _sorted_invoice_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt")


def _write_json(file_path: Path, payload: object) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    _ = file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(input_dir: Path, applicant_file: Path, output_dir: Path) -> PipelineResult:
    invoice_files = _sorted_invoice_files(input_dir)
    if not invoice_files:
        raise ValueError(f"输入目录中没有可处理的 TXT 发票文本: {input_dir}")

    applicant = load_applicant_info(applicant_file)
    invoices: list[InvoiceRecord] = [parse_invoice_file(file_path) for file_path in invoice_files]

    parsed_json_path = output_dir / "parsed_invoices.json"
    workbook_path = output_dir / "reimbursement_form.xlsx"
    summary_json_path = output_dir / "run_summary.json"

    _write_json(
        parsed_json_path,
        {
            "applicant": applicant.to_dict(),
            "invoices": [invoice.to_dict() for invoice in invoices],
        },
    )

    summary = export_reimbursement_workbook(applicant=applicant, invoices=invoices, output_path=workbook_path)
    run_summary: PipelineResult = {
        "summary": summary.to_dict(),
        "artifacts": {
            "parsed_json": str(parsed_json_path),
            "workbook": str(workbook_path),
            "summary_json": str(summary_json_path),
        },
    }
    _write_json(summary_json_path, run_summary)
    return run_summary
