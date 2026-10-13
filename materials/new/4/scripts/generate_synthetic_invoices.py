from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import qrcode  # pyright: ignore[reportMissingModuleSource]
from qrcode.constants import ERROR_CORRECT_M  # pyright: ignore[reportMissingModuleSource]
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data_working" / "synthetic"
DEFAULT_PREVIEW_DIR = PROJECT_ROOT / "evidence" / "label_previews"
TEMPLATE_FAMILIES = [
    "a4_blue_grid",
    "a4_green_box",
    "a4_red_stamp",
    "thermal_gray_dense",
    "thermal_blue_sparse",
    "thermal_beige_mixed",
    "ledger_mono_table",
    "ledger_teal_blocks",
    "ledger_purple_summary",
    "service_orange_note",
    "service_navy_lines",
    "service_olive_blocks",
]

TEMPLATE_SPLITS = {
    "a4_blue_grid": "train",
    "a4_green_box": "train",
    "a4_red_stamp": "train",
    "thermal_gray_dense": "train",
    "thermal_blue_sparse": "train",
    "thermal_beige_mixed": "train",
    "ledger_mono_table": "val",
    "ledger_teal_blocks": "val",
    "ledger_purple_summary": "val",
    "service_orange_note": "test",
    "service_navy_lines": "test",
    "service_olive_blocks": "test",
}

PAYMENT_WORDS = [
    "Invoice",
    "Receipt",
    "Tax Bill",
    "Customer Copy",
    "Statement",
    "Merchant Copy",
    "Payment Notice",
    "Billing Summary",
]

VENDORS = [
    "Northwind Supply",
    "Harbor Retail",
    "Blue Pine Labs",
    "Sunrise Market",
    "Metro Services",
    "Cloud Nine Cafe",
    "Silver Oak Telecom",
    "Acorn Utility",
]

STREET_NAMES = [
    "Jiefang Rd.",
    "Minsheng Ave.",
    "Renmin West St.",
    "Binjiang Blvd.",
    "Wenhua Rd.",
    "Dongfeng North Rd.",
]


@dataclass
class SampleRecord:
    sample_id: str
    template_family: str
    split: str
    qr_present: bool
    image_path: Path
    label_path: Path
    bbox_xyxy: tuple[int, int, int, int] | None
    qr_rotation: float
    qr_scale_ratio: float
    color_mode: str
    degradation: str
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate invoice-like synthetic QR detection samples with YOLO labels."
    )
    parser.add_argument("--count", type=int, default=180, help="Number of synthetic samples to generate.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory for images, labels, and metadata.")
    parser.add_argument("--seed", type=int, default=20260523, help="Deterministic seed for reproducible generation.")
    parser.add_argument(
        "--negative-count",
        type=int,
        default=None,
        help="Optional exact number of no-QR negative samples. Defaults to a safe ratio with a floor of 20 for larger runs.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=12,
        help="Number of samples to include in the preview grid.",
    )
    return parser.parse_args()


def ensure_dirs(out_dir: Path) -> dict[str, Path]:
    paths = {
        "root": out_dir,
        "images": out_dir / "images",
        "labels": out_dir / "labels",
        "preview": out_dir / "preview",
        "manifest": out_dir / "synthetic_manifest.csv",
    }
    for key in ("root", "images", "labels", "preview"):
        paths[key].mkdir(parents=True, exist_ok=True)
    DEFAULT_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    return paths


def pick_negative_count(count: int, requested: int | None) -> int:
    if requested is not None:
        return max(0, min(count, requested))
    if count >= 100:
        return min(count, max(20, round(count * 0.22)))
    return min(count, max(3, round(count * 0.2)))


def stable_choice(items: list[str], rng: random.Random) -> str:
    return items[rng.randrange(len(items))]


def build_sample_plan(count: int, negative_count: int, rng: random.Random) -> list[tuple[str, bool]]:
    family_order = TEMPLATE_FAMILIES.copy()
    rng.shuffle(family_order)
    positive_target = count - negative_count
    plan: list[tuple[str, bool]] = []
    family_index = 0
    positives_added = 0
    negatives_added = 0
    while len(plan) < count:
        family = family_order[family_index % len(family_order)]
        family_index += 1
        should_be_positive = positives_added < positive_target
        if should_be_positive:
            plan.append((family, True))
            positives_added += 1
            continue
        if negatives_added < negative_count:
            plan.append((family, False))
            negatives_added += 1
    rng.shuffle(plan)
    return plan


def page_size_for_family(family: str, rng: random.Random) -> tuple[int, int]:
    if family.startswith("thermal"):
        return rng.randint(760, 940), rng.randint(1450, 1820)
    if family.startswith("ledger"):
        return rng.randint(1120, 1440), rng.randint(880, 1120)
    if family.startswith("service"):
        return rng.randint(1000, 1180), rng.randint(1300, 1500)
    return rng.randint(1100, 1280), rng.randint(1460, 1720)


def background_palette(family: str, rng: random.Random) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    if family.startswith("thermal"):
        base = rng.choice([(245, 244, 238), (238, 238, 236), (248, 246, 240)])
        ink = rng.choice([(40, 40, 40), (55, 55, 55), (30, 60, 90)])
        accent = rng.choice([(120, 120, 120), (80, 110, 140), (160, 120, 90)])
        return base, ink, accent
    if family.startswith("ledger"):
        base = rng.choice([(240, 244, 246), (244, 245, 240), (248, 243, 247)])
        ink = rng.choice([(35, 40, 55), (50, 50, 50), (30, 70, 80)])
        accent = rng.choice([(30, 120, 130), (90, 90, 120), (105, 70, 130)])
        return base, ink, accent
    if family.startswith("service"):
        base = rng.choice([(247, 241, 232), (240, 246, 232), (236, 241, 248)])
        ink = rng.choice([(55, 48, 40), (28, 52, 88), (50, 70, 35)])
        accent = rng.choice([(210, 130, 65), (78, 96, 152), (110, 135, 78)])
        return base, ink, accent
    base = rng.choice([(248, 248, 246), (244, 247, 251), (251, 244, 244)])
    ink = rng.choice([(35, 35, 35), (40, 56, 90), (95, 36, 36)])
    accent = rng.choice([(30, 110, 180), (54, 146, 90), (190, 65, 70)])
    return base, ink, accent


def font(size: int) -> Any:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_invoice_background(image: Image.Image, family: str, rng: random.Random) -> None:
    width, height = image.size
    draw = ImageDraw.Draw(image)
    base, ink, accent = background_palette(family, rng)
    draw.rectangle([0, 0, width, height], fill=base)

    margin_x = rng.randint(40, 95)
    margin_top = rng.randint(32, 90)
    title_font = font(28 if width > 1000 else 22)
    body_font = font(16 if width > 1000 else 14)
    small_font = font(13 if width > 1000 else 11)

    title = stable_choice(PAYMENT_WORDS, rng)
    vendor = stable_choice(VENDORS, rng)
    address = f"{rng.randint(8, 688)} {stable_choice(STREET_NAMES, rng)}"
    ticket_id = f"INV-{rng.randint(20230000, 20269999)}"
    draw.text((margin_x, margin_top), vendor, fill=ink, font=title_font)
    draw.text((margin_x, margin_top + 38), address, fill=ink, font=body_font)
    draw.text((margin_x, margin_top + 64), f"Doc: {title}    No: {ticket_id}", fill=accent, font=body_font)

    if family.startswith("a4"):
        draw_a4_table(draw, width, height, margin_x, margin_top + 110, ink, accent, body_font, small_font, rng)
    elif family.startswith("thermal"):
        draw_thermal_receipt(draw, width, height, margin_x, margin_top + 110, ink, accent, body_font, small_font, rng)
    elif family.startswith("ledger"):
        draw_ledger_statement(draw, width, height, margin_x, margin_top + 110, ink, accent, body_font, small_font, rng)
    else:
        draw_service_notice(draw, width, height, margin_x, margin_top + 110, ink, accent, body_font, small_font, rng)

    add_background_distractors(draw, width, height, ink, accent, rng)


def draw_a4_table(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    margin_x: int,
    start_y: int,
    ink: tuple[int, int, int],
    accent: tuple[int, int, int],
    body_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    rng: random.Random,
) -> None:
    table_left = margin_x
    table_right = width - margin_x
    header_h = 38
    row_h = rng.randint(34, 42)
    col_positions = [table_left, table_left + int((table_right - table_left) * 0.44), table_left + int((table_right - table_left) * 0.62), table_left + int((table_right - table_left) * 0.79), table_right]
    draw.rounded_rectangle([table_left, start_y, table_right, start_y + header_h], radius=8, fill=accent)
    headers = ["Description", "Qty", "Unit", "Amount"]
    for idx, header in enumerate(headers):
        draw.text((col_positions[idx] + 10, start_y + 9), header, fill=(250, 250, 250), font=body_font)

    y = start_y + header_h
    row_count = rng.randint(8, 14)
    for row in range(row_count):
        bottom = y + row_h
        fill = (255, 255, 255) if row % 2 == 0 else (244, 245, 247)
        draw.rectangle([table_left, y, table_right, bottom], fill=fill, outline=(220, 220, 220))
        values = [
            f"Service item {row + 1}",
            str(rng.randint(1, 6)),
            f"{rng.randint(12, 899)}.{rng.randint(0, 99):02d}",
            f"{rng.randint(30, 999)}.{rng.randint(0, 99):02d}",
        ]
        for idx, value in enumerate(values):
            draw.text((col_positions[idx] + 10, y + 10), value, fill=ink, font=small_font)
        y = bottom

    subtotal_y = y + 18
    for label, offset in (("Subtotal", 0), ("Tax", 30), ("Total Due", 66)):
        draw.text((table_right - 220, subtotal_y + offset), label, fill=ink, font=body_font)
        draw.text((table_right - 90, subtotal_y + offset), f"{rng.randint(200, 3000)}.{rng.randint(0, 99):02d}", fill=ink, font=body_font)


def draw_thermal_receipt(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    margin_x: int,
    start_y: int,
    ink: tuple[int, int, int],
    accent: tuple[int, int, int],
    body_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    rng: random.Random,
) -> None:
    center_x = width // 2
    draw.line([(margin_x, start_y - 12), (width - margin_x, start_y - 12)], fill=accent, width=2)
    draw.text((center_x - 70, start_y), "POS RECEIPT", fill=ink, font=body_font)
    y = start_y + 36
    for idx in range(rng.randint(20, 30)):
        price = f"{rng.randint(2, 180)}.{rng.randint(0, 99):02d}"
        qty = rng.randint(1, 3)
        draw.text((margin_x, y), f"Item {idx + 1:02d} x{qty}", fill=ink, font=small_font)
        draw.text((width - margin_x - 80, y), price, fill=ink, font=small_font)
        if rng.random() < 0.25:
            draw.line([(margin_x, y + 18), (width - margin_x, y + 18)], fill=(170, 170, 170), width=1)
        y += rng.randint(24, 33)
    draw.text((margin_x, y + 18), "PAYMENT METHOD: QR / CARD / CASH", fill=accent, font=body_font)
    draw.text((margin_x, y + 52), f"TOTAL {rng.randint(100, 999)}.{rng.randint(0, 99):02d}", fill=ink, font=body_font)


def draw_ledger_statement(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    margin_x: int,
    start_y: int,
    ink: tuple[int, int, int],
    accent: tuple[int, int, int],
    body_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    rng: random.Random,
) -> None:
    header_h = 44
    draw.rounded_rectangle([margin_x, start_y, width - margin_x, start_y + header_h], radius=10, outline=accent, width=3)
    draw.text((margin_x + 14, start_y + 11), "ACCOUNT STATEMENT", fill=accent, font=body_font)
    col_x = [margin_x + 10, margin_x + 250, margin_x + 470, margin_x + 640, width - margin_x - 120]
    y = start_y + header_h + 18
    draw.text((col_x[0], y), "Date", fill=ink, font=body_font)
    draw.text((col_x[1], y), "Reference", fill=ink, font=body_font)
    draw.text((col_x[2], y), "Details", fill=ink, font=body_font)
    draw.text((col_x[4], y), "Amount", fill=ink, font=body_font)
    y += 30
    for idx in range(rng.randint(9, 15)):
        row_fill = (255, 255, 255) if idx % 2 == 0 else (235, 240, 244)
        draw.rectangle([margin_x, y - 4, width - margin_x, y + 24], fill=row_fill)
        draw.text((col_x[0], y), f"2026-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}", fill=ink, font=small_font)
        draw.text((col_x[1], y), f"REF{rng.randint(1000,9999)}", fill=ink, font=small_font)
        draw.text((col_x[2], y), f"Settlement batch {idx + 1}", fill=ink, font=small_font)
        draw.text((col_x[4], y), f"{rng.randint(50, 1500)}.{rng.randint(0,99):02d}", fill=ink, font=small_font)
        y += 31
    box_top = y + 18
    draw.rounded_rectangle([margin_x, box_top, width - margin_x, min(height - 60, box_top + 210)], radius=14, outline=accent, width=2)
    paragraph = "Please complete payment before the due date. Keep this statement for reimbursement and internal audit records."
    wrapped = textwrap.wrap(paragraph, width=58)
    for idx, line in enumerate(wrapped):
        draw.text((margin_x + 18, box_top + 18 + idx * 24), line, fill=ink, font=small_font)


def draw_service_notice(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    margin_x: int,
    start_y: int,
    ink: tuple[int, int, int],
    accent: tuple[int, int, int],
    body_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    rng: random.Random,
) -> None:
    draw.rounded_rectangle([margin_x, start_y, width - margin_x, start_y + 68], radius=12, fill=(255, 255, 255), outline=accent, width=2)
    draw.text((margin_x + 16, start_y + 18), "SERVICE CHARGE NOTICE", fill=accent, font=body_font)
    y = start_y + 92
    for block in range(rng.randint(4, 7)):
        block_h = rng.randint(80, 140)
        fill = (255, 255, 255) if block % 2 == 0 else (242, 242, 236)
        draw.rounded_rectangle([margin_x, y, width - margin_x, y + block_h], radius=10, fill=fill, outline=(225, 225, 225))
        draw.text((margin_x + 14, y + 12), f"Section {block + 1}", fill=ink, font=body_font)
        paragraph = "Customer charge summary and settlement instruction. Review items, retained amount, handling fee, and payment confirmation details."
        wrapped = textwrap.wrap(paragraph, width=max(34, (width - 2 * margin_x) // 18))
        for idx, line in enumerate(wrapped[:3]):
            draw.text((margin_x + 14, y + 40 + idx * 20), line, fill=ink, font=small_font)
        if rng.random() < 0.7:
            draw.rectangle([width - margin_x - 170, y + 14, width - margin_x - 24, y + 48], fill=accent)
            draw.text((width - margin_x - 150, y + 22), f"Due {rng.randint(80, 999)}", fill=(250, 250, 250), font=small_font)
        y += block_h + 22


def add_background_distractors(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    ink: tuple[int, int, int],
    accent: tuple[int, int, int],
    rng: random.Random,
) -> None:
    for _ in range(rng.randint(2, 5)):
        x = rng.randint(20, max(20, width - 220))
        y = rng.randint(20, max(20, height - 120))
        w = rng.randint(80, 180)
        h = rng.randint(12, 28)
        if rng.random() < 0.5:
            draw_barcode(draw, x, y, w, h, ink, rng)
        else:
            draw_stamp(draw, x, y, w, h * 3, accent, rng)


def draw_barcode(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, height: int, ink: tuple[int, int, int], rng: random.Random) -> None:
    current = x
    while current < x + width:
        bar_w = rng.randint(1, 4)
        if rng.random() < 0.75:
            draw.rectangle([current, y, min(x + width, current + bar_w), y + height], fill=ink)
        current += bar_w + rng.randint(1, 3)


def draw_stamp(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, height: int, accent: tuple[int, int, int], rng: random.Random) -> None:
    color = tuple(min(255, channel + rng.randint(-20, 20)) for channel in accent)
    draw.ellipse([x, y, x + height, y + height], outline=color, width=3)
    draw.text((x + height // 5, y + height // 3), rng.choice(["PAID", "OK", "CHK"]), fill=color, font=font(14))
    if rng.random() < 0.4:
        draw.line([(x + height + 10, y + height // 2), (x + width, y + height // 2)], fill=color, width=2)


def generate_qr_rgba(rng: random.Random, qr_side: int) -> Image.Image:
    qr = qrcode.QRCode(
        version=rng.randint(2, 5),
        error_correction=ERROR_CORRECT_M,
        box_size=max(4, qr_side // 56),
        border=rng.randint(2, 4),
    )
    payload = f"PAY://{rng.randint(100000000, 999999999)}-{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"
    qr.add_data(payload)
    qr.make(fit=True)
    dark = rng.choice([(15, 15, 15), (30, 30, 30), (20, 45, 60), (55, 25, 25)])
    light = rng.choice([(252, 252, 252), (247, 247, 242), (245, 248, 250)])
    qr_image = qr.make_image(fill_color=dark, back_color=light)
    qr_image = qr_image.get_image().convert("RGBA")
    qr_image = qr_image.resize((qr_side, qr_side), Image.Resampling.NEAREST)

    if rng.random() < 0.45:
        pad = max(6, qr_side // 18)
        qr_image = ImageOps.expand(qr_image, border=pad, fill=light + (255,))
    if rng.random() < 0.35:
        overlay = Image.new("RGBA", qr_image.size, (255, 255, 255, 0))
        odraw = ImageDraw.Draw(overlay)
        for _ in range(rng.randint(2, 4)):
            x0 = rng.randint(0, max(1, qr_image.width - 12))
            y0 = rng.randint(0, max(1, qr_image.height - 12))
            x1 = min(qr_image.width, x0 + rng.randint(qr_image.width // 8, qr_image.width // 4))
            y1 = min(qr_image.height, y0 + rng.randint(6, qr_image.height // 8))
            odraw.rectangle([x0, y0, x1, y1], fill=(245, 245, 245, rng.randint(120, 220)))
        qr_image = Image.alpha_composite(qr_image, overlay)

    rotation = rng.uniform(-9.0, 9.0)
    return qr_image.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255, 0))


def place_qr(image: Image.Image, family: str, rng: random.Random) -> tuple[tuple[int, int, int, int], float, float]:
    width, height = image.size
    short_edge = min(width, height)
    qr_side = rng.randint(int(short_edge * 0.11), int(short_edge * 0.22))
    qr_layer = generate_qr_rgba(rng, qr_side)
    alpha = qr_layer.getchannel("A")
    qr_box = alpha.getbbox()
    if qr_box is None:
        raise RuntimeError("Generated QR mask is empty.")

    region_candidates = candidate_regions(width, height, family)
    region = rng.choice(region_candidates)
    available_x0, available_y0, available_x1, available_y1 = region
    max_x = max(available_x0, available_x1 - qr_layer.width)
    max_y = max(available_y0, available_y1 - qr_layer.height)
    paste_x = rng.randint(available_x0, max_x)
    paste_y = rng.randint(available_y0, max_y)

    if rng.random() < 0.35:
        shadow = Image.new("RGBA", qr_layer.size, (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(shadow)
        sdraw.rectangle([0, 0, qr_layer.width - 1, qr_layer.height - 1], fill=(0, 0, 0, 48))
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=5))
        image.alpha_composite(shadow, dest=(paste_x + 6, paste_y + 6))

    image.alpha_composite(qr_layer, dest=(paste_x, paste_y))
    x0 = paste_x + qr_box[0]
    y0 = paste_y + qr_box[1]
    x1 = paste_x + qr_box[2]
    y1 = paste_y + qr_box[3]
    side_ratio = qr_side / short_edge
    return (x0, y0, x1, y1), float(extract_rotation_hint(qr_layer)), side_ratio


def candidate_regions(width: int, height: int, family: str) -> list[tuple[int, int, int, int]]:
    if family.startswith("thermal"):
        return [
            (40, int(height * 0.55), width - 40, height - 80),
            (40, int(height * 0.18), width - 40, int(height * 0.5)),
        ]
    if family.startswith("ledger"):
        return [
            (int(width * 0.72), int(height * 0.1), width - 40, int(height * 0.42)),
            (int(width * 0.68), int(height * 0.54), width - 50, height - 60),
            (50, int(height * 0.6), int(width * 0.35), height - 60),
        ]
    return [
        (int(width * 0.65), int(height * 0.12), width - 50, int(height * 0.36)),
        (int(width * 0.66), int(height * 0.62), width - 50, height - 70),
        (50, int(height * 0.64), int(width * 0.34), height - 70),
    ]


def extract_rotation_hint(qr_layer: Image.Image) -> float:
    alpha = qr_layer.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        return 0.0
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if width == height:
        return 0.0
    return math.degrees(math.atan2(height - width, max(width, 1)))


def apply_global_degradation(image: Image.Image, rng: random.Random) -> tuple[Image.Image, str, str]:
    degradation_flags: list[str] = []
    color_mode = "color"
    if rng.random() < 0.35:
        image = ImageOps.grayscale(image).convert("RGB")
        color_mode = "grayscale"
        degradation_flags.append("grayscale")
    elif rng.random() < 0.2:
        image = ImageOps.colorize(ImageOps.grayscale(image), black=(45, 45, 45), white=(235, 230, 210))
        color_mode = "duotone"
        degradation_flags.append("duotone")

    if rng.random() < 0.5:
        blur_radius = rng.uniform(0.3, 1.6)
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        degradation_flags.append(f"blur:{blur_radius:.2f}")

    array = np.asarray(image).astype(np.int16)
    if rng.random() < 0.7:
        noise_sigma = rng.uniform(2.0, 13.0)
        noise = np.random.default_rng(rng.randint(0, 10_000_000)).normal(0, noise_sigma, array.shape)
        array = np.clip(array + noise, 0, 255)
        degradation_flags.append(f"noise:{noise_sigma:.1f}")

    image = Image.fromarray(array.astype(np.uint8))

    if rng.random() < 0.6:
        quality = rng.randint(35, 82)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
        degradation_flags.append(f"jpeg:{quality}")

    return image, color_mode, "|".join(degradation_flags) if degradation_flags else "clean"


def yolo_line(bbox_xyxy: tuple[int, int, int, int], image_size: tuple[int, int]) -> str:
    width, height = image_size
    x0, y0, x1, y1 = bbox_xyxy
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    x_center = (x0 + x1) / 2 / width
    y_center = (y0 + y1) / 2 / height
    norm_w = box_w / width
    norm_h = box_h / height
    return f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def record_rows(out_dir: Path, rows: list[SampleRecord]) -> None:
    manifest_path = out_dir / "synthetic_manifest.csv"
    fieldnames = [
        "id",
        "source_name",
        "source_url",
        "license",
        "real_or_synthetic",
        "split",
        "image_path",
        "label_path",
        "qr_present",
        "notes",
        "include_status",
        "template_family",
        "split_guardrail_group",
        "qr_rotation_hint",
        "qr_scale_ratio",
        "color_mode",
        "degradation",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row.sample_id,
                    "source_name": "Project synthetic invoice QR generation",
                    "source_url": "scripts/generate_synthetic_invoices.py",
                    "license": "internal project generated",
                    "real_or_synthetic": "synthetic",
                    "split": row.split,
                    "image_path": relative_path(row.image_path),
                    "label_path": relative_path(row.label_path),
                    "qr_present": str(row.qr_present).lower(),
                    "notes": row.notes,
                    "include_status": "pending_review",
                    "template_family": row.template_family,
                    "split_guardrail_group": row.template_family,
                    "qr_rotation_hint": f"{row.qr_rotation:.3f}",
                    "qr_scale_ratio": f"{row.qr_scale_ratio:.4f}",
                    "color_mode": row.color_mode,
                    "degradation": row.degradation,
                }
            )


def render_preview(rows: list[SampleRecord], out_dir: Path, preview_count: int) -> Path:
    chosen = rows[: min(preview_count, len(rows))]
    if not chosen:
        raise RuntimeError("No rows available for preview rendering.")
    thumbs: list[Image.Image] = []
    for row in chosen:
        image = Image.open(row.image_path).convert("RGB")
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        if row.bbox_xyxy is not None:
            draw.rectangle(row.bbox_xyxy, outline=(220, 40, 40), width=5)
            draw.text((row.bbox_xyxy[0] + 5, max(6, row.bbox_xyxy[1] - 24)), row.template_family, fill=(220, 40, 40), font=font(16))
        else:
            draw.text((12, 12), f"NEGATIVE | {row.template_family}", fill=(220, 40, 40), font=font(16))
        overlay.thumbnail((420, 420))
        thumbs.append(overlay)

    cols = 3
    rows_needed = math.ceil(len(thumbs) / cols)
    canvas = Image.new("RGB", (cols * 430, rows_needed * 430), (245, 245, 245))
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * 430 + 5
        y = (idx // cols) * 430 + 5
        canvas.paste(thumb, (x, y))

    local_path = out_dir / "preview" / "preview_grid.png"
    evidence_path = DEFAULT_PREVIEW_DIR / f"{out_dir.name}_preview_grid.png"
    canvas.save(local_path)
    canvas.save(evidence_path)
    return evidence_path


def sample_hash(sample_id: str) -> str:
    return hashlib.md5(sample_id.encode("utf-8")).hexdigest()[:10]


def generate_dataset(count: int, out_dir: Path, seed: int, negative_count: int, preview_count: int) -> tuple[list[SampleRecord], Path]:
    rng = random.Random(seed)
    paths = ensure_dirs(out_dir)
    plan = build_sample_plan(count, negative_count, rng)
    records: list[SampleRecord] = []

    for index, (family, qr_present) in enumerate(plan):
        local_seed = seed * 1000 + index * 17 + sum(ord(ch) for ch in family)
        sample_rng = random.Random(local_seed)
        size = page_size_for_family(family, sample_rng)
        base, _, _ = background_palette(family, sample_rng)
        base_image = Image.new("RGB", size, base)
        draw_invoice_background(base_image, family, sample_rng)

        rgba = base_image.convert("RGBA")
        bbox = None
        qr_rotation = 0.0
        qr_scale_ratio = 0.0
        if qr_present:
            bbox, qr_rotation, qr_scale_ratio = place_qr(rgba, family, sample_rng)
        final_image, color_mode, degradation = apply_global_degradation(rgba.convert("RGB"), sample_rng)

        sample_id = f"syn-{family}-{index:04d}-{sample_hash(family + str(seed) + str(index))}"
        image_path = paths["images"] / f"{sample_id}.png"
        label_path = paths["labels"] / f"{sample_id}.txt"
        final_image.save(image_path)
        if bbox is not None:
            label_path.write_text(yolo_line(bbox, final_image.size) + "\n", encoding="utf-8")
        else:
            label_path.write_text("", encoding="utf-8")

        split = TEMPLATE_SPLITS[family]
        note_bits = [
            f"template_family={family}",
            f"split_guardrail={family}->{split}",
            f"qr_present={str(qr_present).lower()}",
        ]
        if not qr_present:
            note_bits.append("negative_sample=true")
        records.append(
            SampleRecord(
                sample_id=sample_id,
                template_family=family,
                split=split,
                qr_present=qr_present,
                image_path=image_path,
                label_path=label_path,
                bbox_xyxy=bbox,
                qr_rotation=qr_rotation,
                qr_scale_ratio=qr_scale_ratio,
                color_mode=color_mode,
                degradation=degradation,
                notes="; ".join(note_bits),
            )
        )

    records.sort(key=lambda item: item.sample_id)
    record_rows(out_dir, records)
    preview_path = render_preview(records, out_dir, preview_count)
    return records, preview_path


def print_summary(records: list[SampleRecord], preview_path: Path, out_dir: Path, seed: int) -> None:
    positive_count = sum(1 for row in records if row.qr_present)
    negative_count = len(records) - positive_count
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for row in records:
        split_counts[row.split] = split_counts.get(row.split, 0) + 1
    print(f"Generated {len(records)} samples in {relative_path(out_dir)}")
    print(f"Seed: {seed}")
    print(f"Positives: {positive_count} | Negatives: {negative_count}")
    print(f"Split counts (family-guarded): {split_counts}")
    print(f"Preview grid: {relative_path(preview_path)}")
    print(f"Companion manifest: {relative_path(out_dir / 'synthetic_manifest.csv')}")


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be > 0")
    negative_count = pick_negative_count(args.count, args.negative_count)
    records, preview_path = generate_dataset(
        count=args.count,
        out_dir=args.out.resolve(),
        seed=args.seed,
        negative_count=negative_count,
        preview_count=max(1, args.preview_count),
    )
    print_summary(records, preview_path, args.out.resolve(), args.seed)


if __name__ == "__main__":
    main()
