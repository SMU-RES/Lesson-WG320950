from __future__ import annotations

import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_images(source: str | Path) -> list[Path]:
    root = Path(source)
    if not root.exists():
        raise FileNotFoundError(f"TRUDI source directory does not exist: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def select_images(images: list[Path], limit: int, seed: int) -> list[Path]:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:limit])


def copy_images(images: list[Path], out_dir: str | Path) -> list[Path]:
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for index, image in enumerate(images, start=1):
        target = target_dir / image.name
        if target.exists():
            target = target_dir / f"{index:03d}_{image.name}"
        shutil.copy2(image, target)
        copied.append(target)
    return copied


def prepare_samples(source: str | Path, limit: int, seed: int, out_dir: str | Path) -> list[Path]:
    images = find_images(source)
    if not images:
        raise FileNotFoundError(f"No supported images found under {source}")
    return copy_images(select_images(images, limit=limit, seed=seed), out_dir)
