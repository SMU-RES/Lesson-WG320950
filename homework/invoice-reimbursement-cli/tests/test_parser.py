from decimal import Decimal
from importlib import import_module
import sys
from pathlib import Path
from typing import Callable, Protocol, cast


class InvoiceLike(Protocol):
    category: str
    invoice_number: str
    total_amount: Decimal
    net_amount: Decimal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


parse_invoice_text = cast(
    Callable[..., InvoiceLike],
    getattr(import_module("invoice_reimbursement.parser"), "parse_invoice_text"),
)


def test_parse_invoice_text_extracts_core_fields() -> None:
    text = """
    增值税电子普通发票
    销售方名称：上海虹桥高铁服务有限公司
    发票代码：3100202310
    发票号码：08234567
    开票日期：2026-03-18
    项目名称：高铁车票
    价税合计(小写)：156.50
    税额：4.56
    备注：交通报销
    """
    invoice = parse_invoice_text(text=text, source_file="transport.txt")

    assert invoice.category == "交通"
    assert invoice.invoice_number == "08234567"
    assert invoice.total_amount == Decimal("156.50")
    assert invoice.net_amount == Decimal("151.94")
