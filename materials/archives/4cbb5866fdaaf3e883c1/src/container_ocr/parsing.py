from __future__ import annotations

import re
from collections.abc import Iterable

from .types import ContainerNumber, OCRText, SizeTypeCode


LETTER_VALUES = {
    "A": 10,
    "B": 12,
    "C": 13,
    "D": 14,
    "E": 15,
    "F": 16,
    "G": 17,
    "H": 18,
    "I": 19,
    "J": 20,
    "K": 21,
    "L": 23,
    "M": 24,
    "N": 25,
    "O": 26,
    "P": 27,
    "Q": 28,
    "R": 29,
    "S": 30,
    "T": 31,
    "U": 32,
    "V": 34,
    "W": 35,
    "X": 36,
    "Y": 37,
    "Z": 38,
}

CONTAINER_PATTERN = re.compile(r"[A-Z]{3}[UJZ]\d{7}")
MARKETING_SIZE_PATTERN = re.compile(r"\b(?:20|40|45)(?:GP|HC|HQ|RF|OT|FR|TK)\b")
ISO_SIZE_PATTERN = re.compile(r"\b[1245L][0-9][GRHUPTVBS][0-9A-Z]\b")


def normalize_ocr_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def calculate_check_digit(code_without_digit: str) -> int:
    candidate = normalize_ocr_text(code_without_digit)
    if not re.fullmatch(r"[A-Z]{3}[UJZ]\d{6}", candidate):
        raise ValueError("ISO 6346 check-digit input must be 3 letters, category, and 6 digits")

    total = 0
    for index, char in enumerate(candidate):
        if char.isdigit():
            value = int(char)
        else:
            value = LETTER_VALUES[char]
        total += value * (2**index)

    remainder = total % 11
    return 0 if remainder == 10 else remainder


def validate_container_number(value: str, raw_text: str | None = None, text_index: int | None = None) -> ContainerNumber:
    normalized = normalize_ocr_text(value)
    if not re.fullmatch(r"[A-Z]{3}[UJZ]\d{7}", normalized):
        raise ValueError(f"Not an ISO 6346 candidate: {value}")

    expected = calculate_check_digit(normalized[:10])
    actual = int(normalized[-1])
    valid = expected == actual
    reason = "valid" if valid else f"invalid check digit: expected {expected}, got {actual}"
    return ContainerNumber(
        value=normalized,
        raw_text=raw_text if raw_text is not None else value,
        owner_code=normalized[:3],
        equipment_category=normalized[3],
        serial_number=normalized[4:10],
        check_digit=actual,
        calculated_check_digit=expected,
        is_valid=valid,
        reason=reason,
        text_index=text_index,
    )


def _iter_texts(source: str | Iterable[str] | Iterable[OCRText]) -> list[tuple[str, float | None, int | None]]:
    if isinstance(source, str):
        return [(source, None, 0)]

    items: list[tuple[str, float | None, int | None]] = []
    for index, item in enumerate(source):
        if isinstance(item, OCRText):
            items.append((item.text, item.confidence, index))
        else:
            items.append((str(item), None, index))
    return items


def extract_container_numbers(source: str | Iterable[str] | Iterable[OCRText]) -> list[ContainerNumber]:
    texts = _iter_texts(source)
    candidates: list[ContainerNumber] = []
    seen: set[str] = set()

    def add_from_text(raw: str, text_index: int | None) -> None:
        normalized = normalize_ocr_text(raw)
        for match in CONTAINER_PATTERN.finditer(normalized):
            value = match.group(0)
            if value in seen:
                continue
            seen.add(value)
            candidates.append(validate_container_number(value, raw_text=raw, text_index=text_index))

    for raw, _confidence, index in texts:
        add_from_text(raw, index)

    if len(texts) > 1:
        joined = " ".join(raw for raw, _confidence, _index in texts)
        add_from_text(joined, None)

    return candidates


def extract_size_type_codes(source: str | Iterable[str] | Iterable[OCRText]) -> list[SizeTypeCode]:
    results: list[SizeTypeCode] = []
    seen: set[str] = set()

    for raw, confidence, index in _iter_texts(source):
        text = raw.upper().replace("-", " ")
        for pattern in (MARKETING_SIZE_PATTERN, ISO_SIZE_PATTERN):
            for match in pattern.finditer(text):
                code = match.group(0)
                if code in seen:
                    continue
                seen.add(code)
                results.append(SizeTypeCode(code=code, raw_text=raw, confidence=confidence, text_index=index))

    return results
