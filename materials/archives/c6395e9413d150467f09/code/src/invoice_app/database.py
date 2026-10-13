# pyright: reportUnusedCallResult=false
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .models import ConsistencyResult, ReimbursementForm, SubmitResult


class ReimbursementRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS invoice (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    unique_key TEXT NOT NULL UNIQUE,
                    invoice_code TEXT NOT NULL,
                    invoice_number TEXT NOT NULL,
                    issue_date TEXT NOT NULL,
                    buyer_name TEXT NOT NULL,
                    seller_name TEXT NOT NULL,
                    amount_without_tax TEXT NOT NULL,
                    tax_amount TEXT NOT NULL,
                    total_amount TEXT NOT NULL,
                    invoice_type TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS validation_result (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    invoice_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    mismatch_detail TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(invoice_id) REFERENCES invoice(id)
                );

                CREATE TABLE IF NOT EXISTS reimbursement (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    invoice_id INTEGER NOT NULL,
                    applicant TEXT NOT NULL,
                    student_id TEXT NOT NULL,
                    expense_type TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    amount TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(invoice_id) REFERENCES invoice(id)
                );
                """
            )

    def submit(self, form: ReimbursementForm, validation: ConsistencyResult) -> SubmitResult:
        if validation.status == "fail":
            return SubmitResult(False, "校验失败，禁止提交报销")
        self.initialize()
        invoice = form.invoice
        mismatch_detail = json.dumps([finding.__dict__ for finding in validation.findings], ensure_ascii=False)
        with self._connect() as connection:
            try:
                cursor = connection.execute(
                    """
                    INSERT INTO invoice (
                        unique_key, invoice_code, invoice_number, issue_date, buyer_name, seller_name,
                        amount_without_tax, tax_amount, total_amount, invoice_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        form.unique_key,
                        invoice.invoice_code,
                        invoice.invoice_number,
                        invoice.issue_date,
                        invoice.buyer_name,
                        invoice.seller_name,
                        f"{invoice.amount_without_tax:.2f}",
                        f"{invoice.tax_amount:.2f}",
                        f"{invoice.total_amount:.2f}",
                        invoice.invoice_type,
                    ),
                )
            except sqlite3.IntegrityError:
                row = connection.execute("SELECT id FROM invoice WHERE unique_key = ?", (form.unique_key,)).fetchone()
                existing_id = int(row["id"]) if row is not None else None
                return SubmitResult(False, "重复报销：该发票已存在", existing_invoice_id=existing_id)

            if cursor.lastrowid is None:
                raise RuntimeError("发票记录写入失败，未返回记录 ID")
            invoice_id = int(cursor.lastrowid)
            connection.execute(
                "INSERT INTO validation_result (invoice_id, status, mismatch_detail) VALUES (?, ?, ?)",
                (invoice_id, validation.status, mismatch_detail),
            )
            reimbursement_cursor = connection.execute(
                """
                INSERT INTO reimbursement (invoice_id, applicant, student_id, expense_type, purpose, amount, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    invoice_id,
                    form.applicant.name,
                    form.applicant.student_id,
                    form.expense_type,
                    form.purpose,
                    f"{form.amount:.2f}",
                    "submitted",
                ),
            )
            if reimbursement_cursor.lastrowid is None:
                raise RuntimeError("报销记录写入失败，未返回记录 ID")
            return SubmitResult(True, "提交成功", reimbursement_id=int(reimbursement_cursor.lastrowid))

    def reimbursement_count(self) -> int:
        self.initialize()
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM reimbursement").fetchone()
            return int(row["count"])
