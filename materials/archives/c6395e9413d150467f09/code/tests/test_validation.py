# pyright: reportCallInDefaultInitializer=false, reportMissingTypeStubs=false
from __future__ import annotations

from decimal import Decimal

from invoice_app.models import InvoiceFields
from invoice_app.validation import build_unique_key, normalize_money, validate_consistency


def _fields(
    invoice_code: str = "044032100111",
    invoice_number: str = "12345678",
    issue_date: str = "2026-05-20",
    buyer_name: str = "上海海事大学",
    seller_name: str = "测试科技有限公司",
    amount_without_tax: Decimal = Decimal("100.00"),
    tax_amount: Decimal = Decimal("6.00"),
    total_amount: Decimal = Decimal("106.00"),
    invoice_type: str = "增值税电子普通发票",
) -> InvoiceFields:
    return InvoiceFields(
        invoice_code=invoice_code,
        invoice_number=invoice_number,
        issue_date=issue_date,
        buyer_name=buyer_name,
        seller_name=seller_name,
        amount_without_tax=amount_without_tax,
        tax_amount=tax_amount,
        total_amount=total_amount,
        invoice_type=invoice_type,
    )


def test_normalize_money_rounds_to_two_places() -> None:
    assert normalize_money("10.235") == Decimal("10.24"), "money must be rounded half-up for invoice comparison"


def test_build_unique_key_uses_normalized_invoice_identity() -> None:
    fields = _fields(invoice_number=" 1234 5678 ", total_amount=Decimal("106"))
    assert build_unique_key(fields) == "044032100111|12345678|2026-05-20|106.00"


def test_validate_consistency_passes_when_qr_and_llm_match() -> None:
    result = validate_consistency(_fields(), _fields())
    assert result.status == "pass", "matching QR and LLM fields should pass"


def test_validate_consistency_warns_for_seller_text_difference() -> None:
    result = validate_consistency(_fields(), _fields(seller_name="测试科技有限公司上海分公司"))
    assert result.status == "warning", "seller/buyer text differences should require manual review, not hard fail"


def test_validate_consistency_fails_for_total_amount_difference() -> None:
    result = validate_consistency(_fields(), _fields(total_amount=Decimal("108.00")))
    assert result.status == "fail", "amount mismatch above tolerance must block reimbursement"
