from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from pathlib import Path

from .models import InvoiceRecord, quantize_money


DATE_PATTERNS = [
    re.compile(r"开票日期[:：]\s*(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?"),
    re.compile(r"日期[:：]\s*(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?"),
]

FIELD_PATTERNS = {
    "seller_name": [re.compile(r"销售方名称[:：]\s*(.+)"), re.compile(r"销\s*售\s*方[:：]\s*(.+)")],
    "invoice_code": [re.compile(r"发票代码[:：]\s*([0-9A-Z]+)")],
    "invoice_number": [re.compile(r"发票号码[:：]\s*([0-9A-Z]+)")],
    "title": [re.compile(r"项目名称[:：]\s*(.+)"), re.compile(r"货物或应税劳务、服务名称[:：]\s*(.+)")],
    "notes": [re.compile(r"备注[:：]\s*(.+)")],
}

TOTAL_PATTERNS = [
    re.compile(r"价税合计(?:\(小写\))?[:：¥￥\s]*([0-9]+(?:\.[0-9]{1,2})?)"),
    re.compile(r"合计金额[:：¥￥\s]*([0-9]+(?:\.[0-9]{1,2})?)"),
]

TAX_PATTERNS = [
    re.compile(r"税额[:：¥￥\s]*([0-9]+(?:\.[0-9]{1,2})?)"),
    re.compile(r"税\s*额\s*合计[:：¥￥\s]*([0-9]+(?:\.[0-9]{1,2})?)"),
]

CATEGORY_RULES = {
    "交通": ("高铁", "火车", "机票", "航班", "打车", "滴滴", "交通", "出行"),
    "住宿": ("酒店", "住宿", "宾馆", "客房"),
    "办公": ("办公", "耗材", "打印", "文具", "设备"),
    "餐饮": ("餐饮", "用餐", "午餐", "晚餐"),
    "会议": ("会议", "培训", "会务"),
}


def _extract_first(patterns: list[re.Pattern[str]], text: str, field_name: str) -> str:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    raise ValueError(f"未能从发票文本中提取字段: {field_name}")


def _extract_decimal(patterns: list[re.Pattern[str]], text: str, field_name: str) -> Decimal:
    value = _extract_first(patterns, text, field_name)
    normalized = value.replace(",", "").replace("¥", "").replace("￥", "")
    return quantize_money(Decimal(normalized))


def _extract_date(text: str) -> date:
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            year, month, day = (int(part) for part in match.groups())
            return date(year, month, day)
    raise ValueError("未能从发票文本中提取开票日期")


def infer_category(text: str, title: str) -> str:
    combined = f"{text}\n{title}"
    for category, keywords in CATEGORY_RULES.items():
        if any(keyword in combined for keyword in keywords):
            return category
    return "其他"


def parse_invoice_text(text: str, source_file: str) -> InvoiceRecord:
    seller_name = _extract_first(FIELD_PATTERNS["seller_name"], text, "seller_name")
    invoice_code = _extract_first(FIELD_PATTERNS["invoice_code"], text, "invoice_code")
    invoice_number = _extract_first(FIELD_PATTERNS["invoice_number"], text, "invoice_number")
    title = _extract_first(FIELD_PATTERNS["title"], text, "title")
    total_amount = _extract_decimal(TOTAL_PATTERNS, text, "total_amount")
    tax_amount = _extract_decimal(TAX_PATTERNS, text, "tax_amount")
    issue_date = _extract_date(text)

    notes = ""
    try:
        notes = _extract_first(FIELD_PATTERNS["notes"], text, "notes")
    except ValueError:
        notes = "课程设计演示样例"

    net_amount = quantize_money(total_amount - tax_amount)
    category = infer_category(text, title)

    return InvoiceRecord(
        source_file=source_file,
        seller_name=seller_name,
        invoice_code=invoice_code,
        invoice_number=invoice_number,
        issue_date=issue_date,
        total_amount=total_amount,
        tax_amount=tax_amount,
        net_amount=net_amount,
        category=category,
        title=title,
        notes=notes,
    )


def parse_invoice_file(file_path: Path) -> InvoiceRecord:
    from .ocr import extract_invoice_text

    text = extract_invoice_text(file_path)
    return parse_invoice_text(text=text, source_file=file_path.name)
