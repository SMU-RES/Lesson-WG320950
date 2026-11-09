from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP

from .models import ConsistencyFinding, ConsistencyResult, InvoiceFields


TWOPLACES = Decimal("0.01")


def normalize_money(value: Decimal | int | str) -> Decimal:
    return Decimal(str(value).strip() or "0").quantize(TWOPLACES, rounding=ROUND_HALF_UP)


def normalize_date(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    digits = re.sub(r"\D", "", text)
    if len(digits) == 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    match = re.search(r"(\d{4})\D+(\d{1,2})\D+(\d{1,2})", text)
    if match:
        year, month, day = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    return text


def normalize_identifier(value: str) -> str:
    return re.sub(r"\s+", "", value).strip()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value).strip()


def build_unique_key(fields: InvoiceFields) -> str:
    code = normalize_identifier(fields.invoice_code)
    number = normalize_identifier(fields.invoice_number)
    date = normalize_date(fields.issue_date)
    amount = f"{normalize_money(fields.total_amount):.2f}"
    if code:
        return "|".join([code, number, date, amount])
    return "|".join([number, date, amount, normalize_text(fields.seller_name)])


def validate_consistency(qr_fields: InvoiceFields, llm_fields: InvoiceFields) -> ConsistencyResult:
    findings: list[ConsistencyFinding] = []

    def add(field: str, status: str, message: str) -> None:
        findings.append(ConsistencyFinding(field=field, status=status, message=message))

    critical_pairs = [
        ("invoice_number", normalize_identifier(qr_fields.invoice_number), normalize_identifier(llm_fields.invoice_number), "发票号码"),
        ("invoice_code", normalize_identifier(qr_fields.invoice_code), normalize_identifier(llm_fields.invoice_code), "发票代码"),
        ("issue_date", normalize_date(qr_fields.issue_date), normalize_date(llm_fields.issue_date), "开票日期"),
    ]
    for field, left, right, label in critical_pairs:
        if left and right and left != right:
            add(field, "fail", f"{label}不一致：QR={left}，LLM={right}")
        else:
            add(field, "pass", f"{label}一致")

    qr_total = normalize_money(qr_fields.total_amount)
    qr_amount_without_tax = normalize_money(qr_fields.amount_without_tax)
    if qr_total:
        llm_total = normalize_money(llm_fields.total_amount)
        if abs(qr_total - llm_total) > TWOPLACES:
            add("total_amount", "fail", f"价税合计不一致：QR={qr_total:.2f}，LLM={llm_total:.2f}")
        else:
            add("total_amount", "pass", "价税合计一致")
    elif qr_amount_without_tax:
        llm_amount_without_tax = normalize_money(llm_fields.amount_without_tax)
        if abs(qr_amount_without_tax - llm_amount_without_tax) > TWOPLACES:
            add("amount_without_tax", "fail", f"不含税金额不一致：QR={qr_amount_without_tax:.2f}，LLM={llm_amount_without_tax:.2f}")
        else:
            add("amount_without_tax", "pass", "不含税金额一致")
    else:
        add("amount", "warning", "二维码金额缺失，建议人工复核")

    for field, left, right, label in [
        ("buyer_name", qr_fields.buyer_name, llm_fields.buyer_name, "购买方"),
        ("seller_name", qr_fields.seller_name, llm_fields.seller_name, "销售方"),
    ]:
        left_text = normalize_text(left)
        right_text = normalize_text(right)
        if left_text and right_text and left_text != right_text:
            add(field, "warning", f"{label}名称不完全一致，建议人工复核")
        else:
            add(field, "pass", f"{label}一致")

    if any(finding.status == "fail" for finding in findings):
        status = "fail"
    elif any(finding.status == "warning" for finding in findings):
        status = "warning"
    else:
        status = "pass"
    return ConsistencyResult(status=status, findings=findings)
