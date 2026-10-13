from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP


TWOPLACES = Decimal("0.01")


def quantize_money(value: Decimal) -> Decimal:
    return value.quantize(TWOPLACES, rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class ApplicantInfo:
    school: str
    major: str
    name: str
    student_id: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class InvoiceRecord:
    source_file: str
    seller_name: str
    invoice_code: str
    invoice_number: str
    issue_date: date
    total_amount: Decimal
    tax_amount: Decimal
    net_amount: Decimal
    category: str
    title: str
    notes: str

    def to_dict(self) -> dict[str, str]:
        payload = asdict(self)
        payload["issue_date"] = self.issue_date.isoformat()
        payload["total_amount"] = f"{quantize_money(self.total_amount):.2f}"
        payload["tax_amount"] = f"{quantize_money(self.tax_amount):.2f}"
        payload["net_amount"] = f"{quantize_money(self.net_amount):.2f}"
        return payload


@dataclass(frozen=True)
class ReimbursementSummary:
    reimbursement_id: str
    applicant: ApplicantInfo
    generated_on: date
    invoice_count: int
    total_amount: Decimal
    total_tax: Decimal
    total_net: Decimal

    def to_dict(self) -> dict[str, object]:
        return {
            "reimbursement_id": self.reimbursement_id,
            "applicant": self.applicant.to_dict(),
            "generated_on": self.generated_on.isoformat(),
            "invoice_count": self.invoice_count,
            "total_amount": f"{quantize_money(self.total_amount):.2f}",
            "total_tax": f"{quantize_money(self.total_tax):.2f}",
            "total_net": f"{quantize_money(self.total_net):.2f}",
        }
