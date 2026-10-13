from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


Point = tuple[float, float]
Box = list[Point]


@dataclass(slots=True)
class OCRText:
    text: str
    confidence: float | None = None
    box: Box | None = None


@dataclass(slots=True)
class ContainerNumber:
    value: str
    raw_text: str
    owner_code: str
    equipment_category: str
    serial_number: str
    check_digit: int
    calculated_check_digit: int
    is_valid: bool
    reason: str
    text_index: int | None = None


@dataclass(slots=True)
class SizeTypeCode:
    code: str
    raw_text: str
    confidence: float | None = None
    text_index: int | None = None


@dataclass(slots=True)
class GPUStatus:
    available: bool
    reason: str
    device: str | None = None
    device_count: int | None = None
    paddle_version: str | None = None


@dataclass(slots=True)
class InferenceResult:
    image_path: str | None
    ocr_texts: list[OCRText] = field(default_factory=list)
    container_numbers: list[ContainerNumber] = field(default_factory=list)
    size_type_codes: list[SizeTypeCode] = field(default_factory=list)
    annotated_image_path: str | None = None
    raw_result: Any = None


def path_to_str(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return str(path)
