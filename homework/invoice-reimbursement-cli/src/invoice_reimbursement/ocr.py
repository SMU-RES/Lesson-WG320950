from __future__ import annotations

from pathlib import Path


SUPPORTED_SUFFIXES = {".txt"}


def extract_invoice_text(file_path: Path) -> str:
    """Return deterministic OCR-like text for supported local fixtures.

    The project keeps OCR optional on purpose. Stable `.txt` fixtures are the
    default demo path for local course presentation.
    """

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"暂不支持的输入格式: {suffix}")
    return file_path.read_text(encoding="utf-8")
