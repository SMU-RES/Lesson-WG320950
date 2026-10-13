from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias, cast

import yaml
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXPECTED_NAMES = {0: "qr_code"}
DEFAULT_PREVIEW_COUNT = 12
YamlData: TypeAlias = dict[str, object]


@dataclass
class CliArgs:
    data: Path
    out: Path
    preview_dir: Path
    preview_count: int


@dataclass
class LabelBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    line_number: int


@dataclass
class SampleRecord:
    split: str
    image_path: Path
    label_path: Path
    width: int = 0
    height: int = 0
    boxes: list[LabelBox] = field(default_factory=list)
    invalid_label_lines: int = 0
    nonempty_label_lines: int = 0
    explicit_negative: bool = False
    image_hash: str = ""
    manifest_split: str | None = None
    manifest_family: str | None = None
    manifest_group: str | None = None

    @property
    def is_positive(self) -> bool:
        return bool(self.boxes)


@dataclass
class SplitStats:
    image_count: int = 0
    label_count: int = 0
    missing_label_count: int = 0
    extra_label_count: int = 0
    unreadable_image_count: int = 0
    positive_image_count: int = 0
    negative_image_count: int = 0
    positive_label_lines: int = 0
    invalid_label_lines: int = 0


@dataclass
class ManifestInfo:
    manifest_paths: list[Path] = field(default_factory=list)
    rows_by_image_name: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    group_expected_splits: dict[str, set[str]] = field(default_factory=dict)


@dataclass
class ValidationResult:
    data_path: Path
    dataset_root: Path
    out_path: Path
    preview_dir: Path
    split_stats: dict[str, SplitStats] = field(default_factory=dict)
    samples: list[SampleRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    preview_paths: list[Path] = field(default_factory=list)
    duplicate_filename_conflicts: int = 0
    duplicate_hash_conflicts: int = 0
    internal_duplicate_hash_warnings: int = 0
    manifest_images_matched: int = 0
    manifest_images_unmatched: int = 0
    leakage_conflicts: int = 0
    manifest_count: int = 0

    @property
    def passed(self) -> bool:
        return not self.errors

    @property
    def total_images(self) -> int:
        return sum(stats.image_count for stats in self.split_stats.values())

    @property
    def total_labels(self) -> int:
        return sum(stats.label_count for stats in self.split_stats.values())

    @property
    def total_positive_images(self) -> int:
        return sum(stats.positive_image_count for stats in self.split_stats.values())

    @property
    def total_negative_images(self) -> int:
        return sum(stats.negative_image_count for stats in self.split_stats.values())

    @property
    def total_positive_label_lines(self) -> int:
        return sum(stats.positive_label_lines for stats in self.split_stats.values())

    @property
    def total_invalid_label_lines(self) -> int:
        return sum(stats.invalid_label_lines for stats in self.split_stats.values())


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Validate an Ultralytics YOLO QR-code dataset and write evidence artifacts."
    )
    _ = parser.add_argument(
        "--data",
        type=Path,
        default=Path("dataset/data.yaml"),
        help="Path to Ultralytics dataset YAML.",
    )
    _ = parser.add_argument(
        "--out",
        type=Path,
        default=Path("evidence/logs/dataset_validation.txt"),
        help="Validation report output path.",
    )
    _ = parser.add_argument(
        "--preview-dir",
        type=Path,
        default=Path("evidence/label_previews"),
        help="Directory for labeled validation preview images.",
    )
    _ = parser.add_argument(
        "--preview-count",
        type=int,
        default=DEFAULT_PREVIEW_COUNT,
        help="Minimum number of validation preview images to create.",
    )
    namespace = parser.parse_args()
    return CliArgs(
        data=cast(Path, namespace.data),
        out=cast(Path, namespace.out),
        preview_dir=cast(Path, namespace.preview_dir),
        preview_count=cast(int, namespace.preview_count),
    )


def resolve_cli_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists() or cwd_candidate.parent.exists():
        return cwd_candidate
    return (PROJECT_ROOT / path).resolve()


def rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def load_yaml(data_path: Path, errors: list[str]) -> YamlData:
    if not data_path.exists():
        errors.append(f"[DATA_YAML] Missing dataset YAML: {rel_path(data_path)}")
        return {}
    try:
        loaded = cast(object, yaml.safe_load(data_path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError) as exc:
        errors.append(f"[DATA_YAML] Could not read/parse {rel_path(data_path)}: {exc}")
        return {}
    if not isinstance(loaded, dict):
        errors.append(f"[DATA_YAML] Expected mapping in {rel_path(data_path)}, got {type(loaded).__name__}")
        return {}
    return cast(YamlData, loaded)


def split_path_exists(dataset_root: Path, data: YamlData, split: str) -> bool:
    split_value = data.get(split)
    if not isinstance(split_value, str):
        return False
    split_path = Path(split_value)
    image_dir = split_path if split_path.is_absolute() else dataset_root / split_path
    return image_dir.exists()


def resolve_dataset_root(data_path: Path, data: YamlData) -> Path:
    raw_path = data.get("path")
    candidates: list[Path] = []
    if isinstance(raw_path, str) and raw_path.strip():
        raw = Path(raw_path)
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.extend([Path.cwd() / raw, PROJECT_ROOT / raw, data_path.parent / raw])
    else:
        candidates.append(data_path.parent)

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique_candidates.append(resolved)
            seen.add(resolved)

    for candidate in unique_candidates:
        if candidate.exists() and all(split_path_exists(candidate, data, split) for split in SPLITS):
            return candidate
    return unique_candidates[0] if unique_candidates else data_path.parent.resolve()


def normalize_names(raw_names: object) -> dict[int, str]:
    normalized: dict[int, str] = {}
    if isinstance(raw_names, list):
        names_list = cast(list[object], raw_names)
        for index, name in enumerate(names_list):
            normalized[index] = str(name)
    elif isinstance(raw_names, dict):
        names_dict = cast(dict[object, object], raw_names)
        for key, name in names_dict.items():
            try:
                class_id = int(str(key))
            except (TypeError, ValueError):
                continue
            normalized[class_id] = str(name)
    return normalized


def validate_classes(dataset_root: Path, data: YamlData, result: ValidationResult) -> None:
    names = normalize_names(data.get("names"))
    if names != EXPECTED_NAMES:
        result.errors.append(f"[CLASSES] Expected names {EXPECTED_NAMES}, found {names!r}")

    nc_value = data.get("nc")
    if nc_value is not None:
        try:
            nc = int(str(nc_value))
        except (TypeError, ValueError):
            result.errors.append(f"[CLASSES] nc must be integer 1 when present, found {nc_value!r}")
        else:
            if nc != 1:
                result.errors.append(f"[CLASSES] Expected nc=1 when present, found {nc}")

    classes_path = dataset_root / "classes.txt"
    if not classes_path.exists():
        result.errors.append(f"[CLASSES] Missing class list: {rel_path(classes_path)}")
        return
    try:
        class_lines = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines()]
    except OSError as exc:
        result.errors.append(f"[CLASSES] Could not read {rel_path(classes_path)}: {exc}")
        return
    nonempty_lines = [line for line in class_lines if line]
    if nonempty_lines != ["qr_code"]:
        result.errors.append(f"[CLASSES] Expected classes.txt to contain only qr_code, found {nonempty_lines!r}")


def resolve_split_dirs(dataset_root: Path, data: YamlData, split: str, result: ValidationResult) -> tuple[Path, Path] | None:
    split_value = data.get(split)
    if not isinstance(split_value, str) or not split_value.strip():
        result.errors.append(f"[SPLIT_PATH] Split {split!r} must be a non-empty directory path string")
        return None

    image_rel = Path(split_value)
    image_dir = image_rel if image_rel.is_absolute() else dataset_root / image_rel
    image_dir = image_dir.resolve()

    parts = list(image_rel.parts)
    label_dir: Path | None = None
    if parts and parts[0] == "images":
        label_dir = dataset_root.joinpath("labels", *parts[1:]).resolve()
    elif "images" in parts:
        index = parts.index("images")
        label_parts = parts.copy()
        label_parts[index] = "labels"
        label_dir = (dataset_root / Path(*label_parts)).resolve()
    else:
        result.errors.append(
            f"[SPLIT_PATH] Cannot derive label directory for split {split!r} from {split_value!r}"
        )
        return None

    if not image_dir.exists() or not image_dir.is_dir():
        result.errors.append(f"[SPLIT_PATH] Missing image directory for {split}: {rel_path(image_dir)}")
    if not label_dir.exists() or not label_dir.is_dir():
        result.errors.append(f"[SPLIT_PATH] Missing label directory for {split}: {rel_path(label_dir)}")
    return image_dir, label_dir


def list_image_files(image_dir: Path) -> list[Path]:
    if not image_dir.exists() or not image_dir.is_dir():
        return []
    return sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def list_label_files(label_dir: Path) -> list[Path]:
    if not label_dir.exists() or not label_dir.is_dir():
        return []
    return sorted(path for path in label_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as opened_image:
        image = cast(Image.Image, opened_image)
        image.verify()
        return image.size


def parse_label_file(label_path: Path, sample_name: str, errors: list[str], warnings: list[str]) -> tuple[list[LabelBox], int, int, bool]:
    try:
        content = label_path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"[LABEL_READ] Could not read label file {rel_path(label_path)} for {sample_name}: {exc}")
        return [], 1, 0, False

    if not content.strip():
        return [], 0, 0, True

    boxes: list[LabelBox] = []
    invalid_lines = 0
    nonempty_lines = 0
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            warnings.append(f"[LABEL_BLANK_LINE] Ignored blank label line in {rel_path(label_path)}:{line_number}")
            continue
        nonempty_lines += 1
        fields = line.split()
        line_errors: list[str] = []
        if len(fields) != 5:
            line_errors.append(f"expected 5 fields, found {len(fields)}")
        if fields and fields[0] != "0":
            line_errors.append(f"class id must be 0, found {fields[0]!r}")

        values: list[float] = []
        if len(fields) == 5:
            for field_name, field_value in zip(("x_center", "y_center", "width", "height"), fields[1:]):
                try:
                    parsed = float(field_value)
                except ValueError:
                    line_errors.append(f"{field_name} is not a float: {field_value!r}")
                    values.append(float("nan"))
                    continue
                if not math.isfinite(parsed):
                    line_errors.append(f"{field_name} must be finite, found {field_value!r}")
                values.append(parsed)

        if len(values) == 4:
            x_center, y_center, width, height = values
            for field_name, parsed in zip(("x_center", "y_center", "width", "height"), values):
                if math.isfinite(parsed) and not 0.0 <= parsed <= 1.0:
                    line_errors.append(f"{field_name} must be normalized to [0, 1], found {parsed}")
            if math.isfinite(width) and width <= 0.0:
                line_errors.append(f"width must be > 0, found {width}")
            if math.isfinite(height) and height <= 0.0:
                line_errors.append(f"height must be > 0, found {height}")
        else:
            x_center = y_center = width = height = float("nan")

        if line_errors:
            invalid_lines += 1
            errors.append(
                f"[LABEL_FORMAT] {rel_path(label_path)}:{line_number} for {sample_name}: "
                + "; ".join(line_errors)
            )
            continue

        boxes.append(
            LabelBox(
                class_id=0,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
                line_number=line_number,
            )
        )

    return boxes, invalid_lines, nonempty_lines, False


def validate_split(dataset_root: Path, data: YamlData, split: str, result: ValidationResult) -> None:
    stats = SplitStats()
    result.split_stats[split] = stats
    dirs = resolve_split_dirs(dataset_root, data, split, result)
    if dirs is None:
        return
    image_dir, label_dir = dirs
    image_files = list_image_files(image_dir)
    label_files = list_label_files(label_dir)
    stats.image_count = len(image_files)
    stats.label_count = len(label_files)

    image_stems = {image_path.stem for image_path in image_files}
    label_stems = {label_path.stem for label_path in label_files}

    for label_path in label_files:
        if label_path.stem not in image_stems:
            stats.extra_label_count += 1
            result.errors.append(f"[PAIRING] Extra label without image in {split}: {rel_path(label_path)}")

    for image_path in image_files:
        label_path = label_dir / f"{image_path.stem}.txt"
        sample = SampleRecord(split=split, image_path=image_path, label_path=label_path)
        result.samples.append(sample)

        try:
            sample.width, sample.height = read_image_size(image_path)
        except (OSError, UnidentifiedImageError) as exc:
            stats.unreadable_image_count += 1
            result.errors.append(f"[IMAGE_READ] Unreadable image in {split}: {rel_path(image_path)} ({exc})")

        try:
            sample.image_hash = sha256_file(image_path)
        except OSError as exc:
            result.errors.append(f"[IMAGE_HASH] Could not hash image in {split}: {rel_path(image_path)} ({exc})")

        if image_path.stem not in label_stems:
            stats.missing_label_count += 1
            result.errors.append(f"[PAIRING] Missing label for image in {split}: {rel_path(image_path)}")
            continue

        boxes, invalid_lines, nonempty_lines, explicit_negative = parse_label_file(
            label_path, image_path.name, result.errors, result.warnings
        )
        sample.boxes = boxes
        sample.invalid_label_lines = invalid_lines
        sample.nonempty_label_lines = nonempty_lines
        sample.explicit_negative = explicit_negative
        stats.invalid_label_lines += invalid_lines

        if explicit_negative:
            stats.negative_image_count += 1
        else:
            stats.positive_label_lines += len(boxes)
            if boxes:
                stats.positive_image_count += 1
            elif invalid_lines:
                result.errors.append(
                    f"[LABEL_FORMAT] Non-empty label file has no valid boxes: {rel_path(label_path)}"
                )


def check_duplicates(result: ValidationResult) -> None:
    by_filename: dict[str, list[SampleRecord]] = defaultdict(list)
    by_hash: dict[str, list[SampleRecord]] = defaultdict(list)
    for sample in result.samples:
        by_filename[sample.image_path.name].append(sample)
        if sample.image_hash:
            by_hash[sample.image_hash].append(sample)

    for filename, records in sorted(by_filename.items()):
        split_names = sorted({record.split for record in records})
        if len(split_names) > 1:
            result.duplicate_filename_conflicts += 1
            locations = ", ".join(f"{record.split}:{rel_path(record.image_path)}" for record in records)
            result.errors.append(
                f"[DUPLICATE_FILENAME] Image filename {filename!r} appears across splits {split_names}: {locations}"
            )

    for image_hash, records in sorted(by_hash.items()):
        if len(records) < 2:
            continue
        split_names = sorted({record.split for record in records})
        locations = ", ".join(f"{record.split}:{rel_path(record.image_path)}" for record in records)
        if len(split_names) > 1:
            result.duplicate_hash_conflicts += 1
            result.errors.append(
                f"[DUPLICATE_HASH] Image content hash {image_hash[:16]} appears across splits {split_names}: {locations}"
            )
        else:
            result.internal_duplicate_hash_warnings += 1
            result.warnings.append(
                f"[DUPLICATE_HASH_INTERNAL] Image content hash {image_hash[:16]} repeats within split {split_names[0]}: {locations}"
            )


def load_manifest_info() -> ManifestInfo:
    manifest_paths = [
        PROJECT_ROOT / "data_working" / "synthetic" / "synthetic_manifest.csv",
        PROJECT_ROOT / "data_working" / "synthetic_extra_task6" / "synthetic_manifest.csv",
    ]
    rows_by_image_name: dict[str, list[dict[str, str]]] = defaultdict(list)
    group_expected_splits: dict[str, set[str]] = defaultdict(set)
    existing_paths: list[Path] = []

    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            continue
        existing_paths.append(manifest_path)
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                normalized = {str(key): str(value) for key, value in row.items() if key is not None}
                image_name = Path(normalized.get("image_path", "")).name
                if image_name:
                    normalized["_manifest_path"] = str(manifest_path)
                    rows_by_image_name[image_name].append(normalized)
                group = normalized.get("split_guardrail_group") or normalized.get("template_family")
                split = normalized.get("split")
                if group and split:
                    group_expected_splits[group].add(split)

    return ManifestInfo(
        manifest_paths=existing_paths,
        rows_by_image_name=dict(rows_by_image_name),
        group_expected_splits=dict(group_expected_splits),
    )


def check_synthetic_leakage(result: ValidationResult) -> None:
    try:
        manifest_info = load_manifest_info()
    except (OSError, csv.Error) as exc:
        result.errors.append(f"[MANIFEST] Could not read synthetic manifests: {exc}")
        return

    result.manifest_count = len(manifest_info.manifest_paths)
    if not manifest_info.manifest_paths:
        result.warnings.append("[MANIFEST] No synthetic manifests found; synthetic leakage check skipped")
        return

    final_group_splits: dict[str, set[str]] = defaultdict(set)
    final_group_examples: dict[str, list[str]] = defaultdict(list)

    for sample in result.samples:
        manifest_rows = manifest_info.rows_by_image_name.get(sample.image_path.name, [])
        if not manifest_rows:
            result.manifest_images_unmatched += 1
            if sample.image_path.name.startswith("syn-"):
                result.errors.append(f"[MANIFEST] Synthetic-looking image missing from manifests: {rel_path(sample.image_path)}")
            else:
                result.warnings.append(f"[MANIFEST] No manifest metadata for {rel_path(sample.image_path)}")
            continue

        result.manifest_images_matched += 1
        groups = {
            row.get("split_guardrail_group") or row.get("template_family") or ""
            for row in manifest_rows
        } - {""}
        families = {row.get("template_family") or "" for row in manifest_rows} - {""}
        expected_splits = {row.get("split") or "" for row in manifest_rows} - {""}

        if len(groups) != 1:
            result.leakage_conflicts += 1
            result.errors.append(
                f"[LEAKAGE] Manifest group mismatch for {rel_path(sample.image_path)}: {sorted(groups)!r}"
            )
            continue
        if len(expected_splits) != 1:
            result.leakage_conflicts += 1
            result.errors.append(
                f"[LEAKAGE] Manifest split mismatch for {rel_path(sample.image_path)}: {sorted(expected_splits)!r}"
            )
            continue

        group = next(iter(groups))
        expected_split = next(iter(expected_splits))
        sample.manifest_group = group
        sample.manifest_split = expected_split
        sample.manifest_family = next(iter(families)) if families else None
        final_group_splits[group].add(sample.split)
        if len(final_group_examples[group]) < 4:
            final_group_examples[group].append(f"{sample.split}:{sample.image_path.name}")

        if expected_split != sample.split:
            result.leakage_conflicts += 1
            message = (
                f"[LEAKAGE] Final split {sample.split!r} disagrees with manifest split {expected_split!r} "
                + f"for {rel_path(sample.image_path)}"
            )
            result.errors.append(message)

    for group, splits in sorted(final_group_splits.items()):
        if len(splits) > 1:
            result.leakage_conflicts += 1
            examples = ", ".join(final_group_examples[group])
            message = (
                f"[LEAKAGE] split_guardrail_group/template_family {group!r} appears in multiple final "
                + f"splits {sorted(splits)}; examples: {examples}"
            )
            result.errors.append(message)

    for group, expected_splits in sorted(manifest_info.group_expected_splits.items()):
        if len(expected_splits) > 1:
            result.leakage_conflicts += 1
            result.errors.append(
                f"[LEAKAGE] Manifest group {group!r} maps to multiple source splits {sorted(expected_splits)}"
            )


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def select_preview_samples(samples: list[SampleRecord], minimum_count: int) -> list[SampleRecord]:
    selected: list[SampleRecord] = []
    selected_keys: set[tuple[str, str]] = set()

    def add(sample: SampleRecord) -> None:
        key = (sample.split, sample.image_path.name)
        if key not in selected_keys:
            selected.append(sample)
            selected_keys.add(key)

    for split in SPLITS:
        split_samples = sorted(
            [sample for sample in samples if sample.split == split and sample.width > 0 and sample.height > 0],
            key=lambda sample: sample.image_path.name,
        )
        negatives = [sample for sample in split_samples if sample.explicit_negative]
        positives = [sample for sample in split_samples if sample.is_positive]
        if negatives:
            add(negatives[0])
        if positives:
            add(positives[0])
        target_for_split = max(1, minimum_count // len(SPLITS))
        for sample in split_samples:
            if len([item for item in selected if item.split == split]) >= target_for_split:
                break
            add(sample)

    for sample in sorted(samples, key=lambda item: (item.split, item.image_path.name)):
        if len(selected) >= minimum_count:
            break
        if sample.width > 0 and sample.height > 0:
            add(sample)

    return selected[:minimum_count]


def draw_preview(sample: SampleRecord, preview_path: Path) -> None:
    with Image.open(sample.image_path) as image:
        canvas = image.convert("RGB")

    max_width = 900
    max_height = 900
    scale = min(1.0, max_width / canvas.width, max_height / canvas.height)
    if scale < 1.0:
        new_size = (max(1, round(canvas.width * scale)), max(1, round(canvas.height * scale)))
        canvas = canvas.resize(new_size, Image.Resampling.LANCZOS)

    draw = ImageDraw.Draw(canvas)
    font = get_font(18)
    small_font = get_font(14)
    x_scale = canvas.width / max(sample.width, 1)
    y_scale = canvas.height / max(sample.height, 1)

    if sample.boxes:
        for box in sample.boxes:
            x_center = box.x_center * sample.width * x_scale
            y_center = box.y_center * sample.height * y_scale
            width = box.width * sample.width * x_scale
            height = box.height * sample.height * y_scale
            x1 = max(0.0, x_center - width / 2.0)
            y1 = max(0.0, y_center - height / 2.0)
            x2 = min(float(canvas.width - 1), x_center + width / 2.0)
            y2 = min(float(canvas.height - 1), y_center + height / 2.0)
            draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 60), width=4)
            label_y = max(0.0, y1 - 24)
            draw.rectangle([x1, label_y, x1 + 112, label_y + 22], fill=(0, 120, 40))
            draw.text((x1 + 4, label_y + 2), "qr_code", fill=(255, 255, 255), font=small_font)
    else:
        draw.rectangle([0, 0, canvas.width - 1, canvas.height - 1], outline=(230, 40, 40), width=5)

    status = "POSITIVE" if sample.boxes else "NEGATIVE - explicit empty label"
    banner = f"{sample.split.upper()} | {status} | {sample.image_path.name}"
    draw.rectangle([0, 0, canvas.width, 34], fill=(20, 20, 20))
    draw.text((10, 8), banner, fill=(255, 255, 255), font=font)

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(preview_path)


def generate_previews(result: ValidationResult, minimum_count: int) -> None:
    result.preview_dir.mkdir(parents=True, exist_ok=True)
    chosen = select_preview_samples(result.samples, max(DEFAULT_PREVIEW_COUNT, minimum_count))
    for index, sample in enumerate(chosen, start=1):
        safe_stem = sample.image_path.stem[:80]
        preview_path = result.preview_dir / f"validation_{index:02d}_{sample.split}_{safe_stem}.png"
        try:
            draw_preview(sample, preview_path)
        except (OSError, UnidentifiedImageError) as exc:
            result.errors.append(f"[PREVIEW] Could not create preview for {rel_path(sample.image_path)}: {exc}")
            continue
        result.preview_paths.append(preview_path)

    represented_splits = {path.name.split("_")[2] for path in result.preview_paths if len(path.name.split("_")) > 2}
    if len(result.preview_paths) < DEFAULT_PREVIEW_COUNT:
        result.errors.append(
            f"[PREVIEW] Expected at least {DEFAULT_PREVIEW_COUNT} validation previews, created {len(result.preview_paths)}"
        )
    for split in SPLITS:
        if split not in represented_splits:
            result.errors.append(f"[PREVIEW] No validation preview generated for split {split}")


def status_from_count(conflict_count: int) -> str:
    return "PASS" if conflict_count == 0 else "FAIL"


def build_report_lines(result: ValidationResult) -> list[str]:
    lines = [
        "Dataset validation report",
        f"data_yaml={rel_path(result.data_path)}",
        f"dataset_root={rel_path(result.dataset_root)}",
        "",
        "Split counts:",
    ]
    for split in SPLITS:
        stats = result.split_stats.get(split, SplitStats())
        lines.extend(
            [
                f"{split}_images={stats.image_count}",
                f"{split}_labels={stats.label_count}",
                f"{split}_positive_images={stats.positive_image_count}",
                f"{split}_negative_images={stats.negative_image_count}",
                f"{split}_positive_label_lines={stats.positive_label_lines}",
                f"{split}_invalid_label_lines={stats.invalid_label_lines}",
                f"{split}_missing_labels={stats.missing_label_count}",
                f"{split}_extra_labels={stats.extra_label_count}",
                f"{split}_unreadable_images={stats.unreadable_image_count}",
            ]
        )
    lines.extend(
        [
            "",
            "Totals:",
            f"total_images={result.total_images}",
            f"total_labels={result.total_labels}",
            f"positive_images={result.total_positive_images}",
            f"negative_images={result.total_negative_images}",
            f"positive_label_lines={result.total_positive_label_lines}",
            f"invalid_label_count={result.total_invalid_label_lines}",
            "",
            "Duplicate and leakage checks:",
            f"duplicate_filename_conflicts={result.duplicate_filename_conflicts}",
            f"duplicate_filename_status={status_from_count(result.duplicate_filename_conflicts)}",
            f"duplicate_content_hash_conflicts={result.duplicate_hash_conflicts}",
            f"duplicate_content_hash_status={status_from_count(result.duplicate_hash_conflicts)}",
            f"internal_duplicate_hash_warnings={result.internal_duplicate_hash_warnings}",
            f"synthetic_manifest_count={result.manifest_count}",
            f"synthetic_manifest_images_matched={result.manifest_images_matched}",
            f"synthetic_manifest_images_unmatched={result.manifest_images_unmatched}",
            f"synthetic_leakage_conflicts={result.leakage_conflicts}",
            f"synthetic_leakage_status={status_from_count(result.leakage_conflicts)}",
            "",
            "Preview images:",
            f"preview_count={len(result.preview_paths)}",
        ]
    )
    for preview_path in result.preview_paths:
        lines.append(f"preview={rel_path(preview_path)}")
    lines.extend(
        [
            "",
            "Messages:",
            f"critical_error_count={len(result.errors)}",
            f"warning_count={len(result.warnings)}",
        ]
    )
    if result.errors:
        lines.append("Critical errors:")
        lines.extend(f"- {message}" for message in result.errors)
    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {message}" for message in result.warnings)
    final_status = "PASS" if result.passed else "FAIL"
    lines.extend(["", f"FINAL_STATUS: {final_status}", final_status])
    return lines


def write_text_file(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_task_validation_evidence(result: ValidationResult, args: CliArgs) -> None:
    final_status = "PASS" if result.passed else "FAIL"
    exit_code = 0 if result.passed else 1
    command = f".venv\\Scripts\\python.exe scripts\\validate_dataset.py --data {args.data} --out {args.out}"
    lines = [
        "Task 7 dataset validation evidence",
        f"command={command}",
        f"exit_code={exit_code}",
        f"data_yaml={rel_path(result.data_path)}",
        f"report={rel_path(result.out_path)}",
        f"train_images={result.split_stats.get('train', SplitStats()).image_count}",
        f"val_images={result.split_stats.get('val', SplitStats()).image_count}",
        f"test_images={result.split_stats.get('test', SplitStats()).image_count}",
        f"total_images={result.total_images}",
        f"total_labels={result.total_labels}",
        f"positive_images={result.total_positive_images}",
        f"negative_images={result.total_negative_images}",
        f"positive_label_lines={result.total_positive_label_lines}",
        f"invalid_label_count={result.total_invalid_label_lines}",
        f"duplicate_filename_conflicts={result.duplicate_filename_conflicts}",
        f"duplicate_content_hash_conflicts={result.duplicate_hash_conflicts}",
        f"synthetic_leakage_conflicts={result.leakage_conflicts}",
        f"preview_count={len(result.preview_paths)}",
        f"critical_error_count={len(result.errors)}",
        f"warning_count={len(result.warnings)}",
        f"final_status={final_status}",
    ]
    write_text_file(PROJECT_ROOT / ".sisyphus" / "evidence" / "task-7-validation.txt", lines)


def write_task_preview_evidence(result: ValidationResult) -> None:
    split_counts: dict[str, int] = defaultdict(int)
    negative_previews = 0
    positive_previews = 0
    preview_by_name = {path.name: path for path in result.preview_paths}
    for sample in result.samples:
        for preview_name in preview_by_name:
            if f"_{sample.split}_{sample.image_path.stem}" in preview_name:
                split_counts[sample.split] += 1
                if sample.explicit_negative:
                    negative_previews += 1
                elif sample.is_positive:
                    positive_previews += 1
                break

    lines = [
        "Task 7 validation preview evidence",
        f"preview_count={len(result.preview_paths)}",
        f"positive_preview_count={positive_previews}",
        f"negative_preview_count={negative_previews}",
    ]
    for split in SPLITS:
        lines.append(f"{split}_preview_count={split_counts.get(split, 0)}")
    lines.append("representative_preview_paths=")
    for preview_path in result.preview_paths[:24]:
        lines.append(f"- {rel_path(preview_path)}")
    write_text_file(PROJECT_ROOT / ".sisyphus" / "evidence" / "task-7-previews.txt", lines)


def validate_dataset(args: CliArgs) -> ValidationResult:
    data_path = resolve_cli_path(args.data)
    out_path = resolve_cli_path(args.out)
    preview_dir = resolve_cli_path(args.preview_dir)
    preliminary_errors: list[str] = []
    data = load_yaml(data_path, preliminary_errors)
    dataset_root = resolve_dataset_root(data_path, data) if data else data_path.parent.resolve()
    result = ValidationResult(
        data_path=data_path,
        dataset_root=dataset_root,
        out_path=out_path,
        preview_dir=preview_dir,
        errors=preliminary_errors,
    )

    if data:
        validate_classes(dataset_root, data, result)
        for split in SPLITS:
            validate_split(dataset_root, data, split, result)
        check_duplicates(result)
        check_synthetic_leakage(result)
        generate_previews(result, args.preview_count)

    write_text_file(out_path, build_report_lines(result))
    write_task_validation_evidence(result, args)
    write_task_preview_evidence(result)
    return result


def main() -> int:
    args = parse_args()
    if args.preview_count < DEFAULT_PREVIEW_COUNT:
        args.preview_count = DEFAULT_PREVIEW_COUNT
    result = validate_dataset(args)
    print(f"Dataset validation report: {rel_path(result.out_path)}")
    print(f"Preview images created: {len(result.preview_paths)}")
    print(f"Critical errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    print("PASS" if result.passed else "FAIL")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
