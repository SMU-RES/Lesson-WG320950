from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any, Sequence


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
EXPECTED_NAMES = {0: "qr"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a YOLO QR dataset before training.")
    parser.add_argument("--data", default="runtime/datasets/processed/qr_yolo_v2/data.yaml", help="Path to Ultralytics data.yaml.")
    parser.add_argument("--allow-empty-labels", action="store_true", help="Allow empty .txt label files for negative images.")
    return parser.parse_args(argv)


def load_data_yaml(data_yaml: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError:
        return parse_simple_data_yaml(data_yaml)
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{data_yaml} must contain a YAML mapping")
    return data


def parse_simple_data_yaml(data_yaml: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    names: dict[int, str] = {}
    in_names = False
    for raw_line in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line.startswith((" ", "\t")) and in_names and ":" in stripped:
            key, value = stripped.split(":", 1)
            names[int(key.strip())] = value.strip().strip("'\"")
            continue
        in_names = False
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "names":
            in_names = True
            if value.startswith("["):
                parsed_names = ast.literal_eval(value)
                if isinstance(parsed_names, list):
                    names.update({index: str(name) for index, name in enumerate(parsed_names)})
            continue
        data[key] = value.strip("'\"")
    if names:
        data["names"] = names
    return data


def normalize_names(raw_names: object) -> dict[int, str]:
    if isinstance(raw_names, list):
        return {index: str(name) for index, name in enumerate(raw_names)}
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    return {}


def dataset_root(data_yaml: Path, config: dict[str, Any]) -> Path:
    root_value = config.get("path")
    if isinstance(root_value, str) and root_value:
        root = Path(root_value)
        return root if root.is_absolute() else data_yaml.parent / root
    return data_yaml.parent


def label_dir_for_split(root: Path, split_name: str, split_path: Path) -> Path:
    parts = split_path.parts
    if parts and parts[0] == "images":
        return root.joinpath("labels", *parts[1:])
    return root / "labels" / split_name


def iter_images(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def validate_label_file(label_path: Path, allow_empty_labels: bool) -> list[str]:
    failures: list[str] = []
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines and not allow_empty_labels:
        return [f"{label_path}: empty label file"]
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 5:
            failures.append(f"{label_path}:{line_number}: expected 5 YOLO fields, got {len(fields)}")
            continue
        class_id = fields[0]
        if class_id != "0":
            failures.append(f"{label_path}:{line_number}: expected class id 0, got {class_id}")
        try:
            x_center, y_center, width, height = (float(value) for value in fields[1:])
        except ValueError:
            failures.append(f"{label_path}:{line_number}: bbox fields must be numeric")
            continue
        for name, value in (("x_center", x_center), ("y_center", y_center), ("width", width), ("height", height)):
            if value < 0 or value > 1:
                failures.append(f"{label_path}:{line_number}: {name} must be normalized to [0, 1], got {value}")
        if width <= 0 or height <= 0:
            failures.append(f"{label_path}:{line_number}: width and height must be positive")
    return failures


def validate_dataset(data_yaml: Path, allow_empty_labels: bool = False) -> list[str]:
    failures: list[str] = []
    data_yaml = data_yaml.resolve()
    if not data_yaml.exists():
        return [f"data.yaml not found: {data_yaml}"]
    config = load_data_yaml(data_yaml)
    names = normalize_names(config.get("names"))
    if names != EXPECTED_NAMES:
        failures.append(f"names must be {EXPECTED_NAMES}, got {names}")
    root = dataset_root(data_yaml, config).resolve()
    for split_name in ("train", "val", "test"):
        split_value = config.get(split_name)
        if not isinstance(split_value, str) or not split_value:
            failures.append(f"missing split path: {split_name}")
            continue
        image_dir = (root / split_value).resolve()
        label_dir = label_dir_for_split(root, split_name, Path(split_value)).resolve()
        if not image_dir.exists():
            failures.append(f"{split_name} image directory not found: {image_dir}")
            continue
        if not label_dir.exists():
            failures.append(f"{split_name} label directory not found: {label_dir}")
            continue
        images = iter_images(image_dir)
        if not images:
            failures.append(f"{split_name} has no images: {image_dir}")
        for image_path in images:
            relative_image = image_path.relative_to(image_dir)
            label_path = label_dir / relative_image.with_suffix(".txt")
            if not label_path.exists():
                failures.append(f"missing label for {image_path}: expected {label_path}")
                continue
            failures.extend(validate_label_file(label_path, allow_empty_labels))
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    failures = validate_dataset(Path(args.data), allow_empty_labels=bool(args.allow_empty_labels))
    if failures:
        print("YOLO dataset validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("YOLO dataset validation passed: class 0=qr, image/label pairs, and normalized boxes are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
