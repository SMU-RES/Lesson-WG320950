from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .types import ContainerNumber, OCRText, SizeTypeCode


def load_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def _polygon(draw: ImageDraw.ImageDraw, box: list[tuple[float, float]], color: str, width: int) -> None:
    points = [(int(x), int(y)) for x, y in box]
    if len(points) >= 2:
        draw.line(points + [points[0]], fill=color, width=width)


def annotate_image(
    image: str | Path | Image.Image,
    ocr_texts: list[OCRText],
    container_numbers: list[ContainerNumber],
    size_type_codes: list[SizeTypeCode],
) -> Image.Image:
    annotated = load_image(image).copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    valid_indexes = {item.text_index for item in container_numbers if item.is_valid and item.text_index is not None}
    invalid_indexes = {item.text_index for item in container_numbers if not item.is_valid and item.text_index is not None}
    size_indexes = {item.text_index for item in size_type_codes if item.text_index is not None}

    for index, item in enumerate(ocr_texts):
        if item.box is None:
            continue
        color = "#3fb950"
        width = 3
        label = item.text
        if index in invalid_indexes:
            color = "#d29922"
        elif index in size_indexes and index not in valid_indexes:
            color = "#58a6ff"
            width = 2
        elif index not in valid_indexes:
            color = "#8b949e"
            width = 1
        _polygon(draw, item.box, color=color, width=width)
        x, y = int(item.box[0][0]), int(item.box[0][1])
        draw.text((x, max(0, y - 12)), label[:32], fill=color, font=font)

    summary_parts = []
    valid = [item.value for item in container_numbers if item.is_valid]
    invalid = [item.value for item in container_numbers if not item.is_valid]
    sizes = [item.code for item in size_type_codes]
    if valid:
        summary_parts.append("Container: " + ", ".join(valid))
    if invalid:
        summary_parts.append("Check digit failed: " + ", ".join(invalid))
    if sizes:
        summary_parts.append("Size/type: " + ", ".join(sizes))
    if summary_parts:
        draw.rectangle((0, 0, annotated.width, 24), fill="#0d1117")
        draw.text((8, 6), " | ".join(summary_parts)[:160], fill="#f0f6fc", font=font)

    return annotated


def save_annotation(image: Image.Image, out_dir: str | Path, source_name: str = "result") -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(source_name).stem or "result"
    target = out_path / f"{stem}_annotated.jpg"
    image.save(target, quality=95)
    return target
