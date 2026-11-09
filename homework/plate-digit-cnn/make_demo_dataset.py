from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from plate_cnn.data import CLASS_NAMES
from plate_cnn.utils import ensure_dir

IMAGE_SIZE = (32, 32)
SPLITS = {"train": 120, "val": 30, "test": 30}
SEGMENTS: dict[str, tuple[str, ...]] = {
    "0": ("a", "b", "c", "d", "e", "f"),
    "1": ("b", "c"),
    "2": ("a", "b", "g", "e", "d"),
    "3": ("a", "b", "g", "c", "d"),
    "4": ("f", "g", "b", "c"),
    "5": ("a", "f", "g", "c", "d"),
    "6": ("a", "f", "g", "e", "c", "d"),
    "7": ("a", "b", "c"),
    "8": ("a", "b", "c", "d", "e", "f", "g"),
    "9": ("a", "b", "c", "d", "f", "g"),
}


def segment_boxes(offset_x: int, offset_y: int, thickness: int) -> dict[str, tuple[int, int, int, int]]:
    left = offset_x + 6
    right = offset_x + 24
    top = offset_y + 4
    mid = offset_y + 14
    bottom = offset_y + 24

    return {
        "a": (left, top, right, top + thickness),
        "g": (left, mid, right, mid + thickness),
        "d": (left, bottom, right, bottom + thickness),
        "f": (left, top, left + thickness, mid),
        "b": (right - thickness, top, right, mid),
        "e": (left, mid, left + thickness, bottom),
        "c": (right - thickness, mid, right, bottom),
    }


def create_digit_image(digit: str, seed: int) -> Image.Image:
    rng = random.Random(seed)
    image = Image.new("L", IMAGE_SIZE, color=rng.randint(215, 255))
    draw = ImageDraw.Draw(image)
    thickness = rng.randint(3, 5)
    boxes = segment_boxes(rng.randint(-2, 2), rng.randint(-2, 2), thickness)
    ink = rng.randint(0, 40)

    for segment in SEGMENTS[digit]:
        draw.rounded_rectangle(boxes[segment], radius=1, fill=ink)

    for _ in range(rng.randint(2, 6)):
        x = rng.randint(0, IMAGE_SIZE[0] - 1)
        y = rng.randint(0, IMAGE_SIZE[1] - 1)
        image.putpixel((x, y), rng.randint(80, 200))

    if rng.random() < 0.7:
        image = image.rotate(rng.uniform(-8, 8), fillcolor=rng.randint(220, 255))
    if rng.random() < 0.6:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.8)))

    return image


def main() -> None:
    root = Path("data/demo_digits")
    for split, count in SPLITS.items():
        for digit in CLASS_NAMES:
            target_dir = ensure_dir(root / split / digit)
            for index in range(count):
                image = create_digit_image(digit, seed=hash((split, digit, index)) & 0xFFFFFFFF)
                image.save(target_dir / f"{digit}_{index:03d}.png")
    print(f"Demo dataset created at: {root.resolve()}")


if __name__ == "__main__":
    main()
