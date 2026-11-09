from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

from plate_cnn.data import CLASS_NAMES
from plate_cnn.utils import ensure_dir

DATASET_URL = "https://raw.githubusercontent.com/pragatiunna/License-Plate-Number-Detection/main/data.zip"
ARCHIVE_NAME = "license_plate_characters.zip"
SOURCE_DIR_NAME = "source_extract"
TARGET_DIR_NAME = "public_digits"


def download_archive(archive_path: Path) -> None:
    if archive_path.exists():
        print(f"Using cached archive: {archive_path.resolve()}")
        return

    _ = ensure_dir(archive_path.parent)
    print(f"Downloading public dataset from: {DATASET_URL}")
    _ = urllib.request.urlretrieve(DATASET_URL, archive_path)
    print(f"Archive saved to: {archive_path.resolve()}")


def extract_archive(archive_path: Path, extract_root: Path) -> Path:
    source_root = extract_root / SOURCE_DIR_NAME
    if source_root.exists():
        shutil.rmtree(source_root)
    _ = ensure_dir(source_root)

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(source_root)

    extracted_data_dir = source_root / "data"
    if not extracted_data_dir.exists():
        raise FileNotFoundError(f"Expected extracted folder not found: {extracted_data_dir}")
    return extracted_data_dir


def copy_split_files(source_root: Path, target_root: Path) -> None:
    if target_root.exists():
        shutil.rmtree(target_root)

    split_mapping = {
        "train": "train",
        "val": "val",
    }

    for source_split, target_split in split_mapping.items():
        for digit in CLASS_NAMES:
            _ = ensure_dir(target_root / target_split / digit)
            class_dir = source_root / source_split / f"class_{digit}"
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing source class directory: {class_dir}")

            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file():
                    _ = shutil.copy2(image_path, target_root / target_split / digit / image_path.name)


def create_test_split(target_root: Path) -> None:
    for digit in CLASS_NAMES:
        val_dir = target_root / "val" / digit
        test_dir = ensure_dir(target_root / "test" / digit)
        val_images = sorted(path for path in val_dir.iterdir() if path.is_file())
        if len(val_images) < 2:
            raise ValueError(f"Not enough validation images to create test split for class {digit}")

        test_count = max(1, len(val_images) // 2)
        for image_path in val_images[-test_count:]:
            _ = shutil.move(str(image_path), test_dir / image_path.name)


def summarize_dataset(target_root: Path) -> None:
    for split in ("train", "val", "test"):
        counts: list[str] = []
        for digit in CLASS_NAMES:
            class_dir = target_root / split / digit
            count = sum(1 for path in class_dir.iterdir() if path.is_file())
            counts.append(f"{digit}:{count}")
        print(f"{split}: {' '.join(counts)}")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    raw_root = ensure_dir(project_root / "data" / "raw_public")
    archive_path = raw_root / ARCHIVE_NAME
    target_root = project_root / "data" / TARGET_DIR_NAME

    download_archive(archive_path)
    extracted_data_dir = extract_archive(archive_path, raw_root)
    copy_split_files(extracted_data_dir, target_root)
    create_test_split(target_root)

    print(f"Prepared public dataset at: {target_root.resolve()}")
    summarize_dataset(target_root)


if __name__ == "__main__":
    main()
