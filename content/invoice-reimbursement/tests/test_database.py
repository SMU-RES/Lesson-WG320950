# pyright: reportMissingTypeStubs=false
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from invoice_app.database import ReimbursementRepository
from invoice_app.models import ApplicantInfo, ConsistencyResult, InvoiceFields, ReimbursementForm
from invoice_app.validation import build_unique_key


def _form() -> ReimbursementForm:
    invoice = InvoiceFields(
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
    return ReimbursementForm(
        applicant=ApplicantInfo(),
        invoice=invoice,
        expense_type="办公费",
        purpose="课程项目测试",
        amount=Decimal("106.00"),
        unique_key=build_unique_key(invoice),
    )


def test_repository_blocks_duplicate_reimbursement(tmp_path: Path) -> None:
    repo = ReimbursementRepository(tmp_path / "invoice.sqlite3")
    repo.initialize()
    validation = ConsistencyResult(status="pass")

    first = repo.submit(_form(), validation)
    second = repo.submit(_form(), validation)

    assert first.accepted is True
    assert second.accepted is False, "second submit of same invoice key must be blocked"
    assert repo.reimbursement_count() == 1, "duplicate invoice must not create a second reimbursement row"
