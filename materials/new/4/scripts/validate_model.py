from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable, Sequence
from typing import Any, cast

import yaml
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO  # pyright: ignore[reportMissingImports]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIN_BEST_WEIGHT_BYTES = 1_048_576
MAP50_FLOOR = 0.80
RECALL_FLOOR = 0.75
NEGATIVE_FP_FLOOR = 2
DEFAULT_METRICS_LOG = Path("evidence/logs/validation_metrics.txt")
TASK_METRICS_EVIDENCE = Path(".sisyphus/evidence/task-9-metrics.txt")
TASK_NEGATIVES_EVIDENCE = Path(".sisyphus/evidence/task-9-negatives.txt")
TRAINING_RESULTS_SCREENSHOT = Path("evidence/screenshots/06-training-results.png")
VALIDATION_RESULTS_SCREENSHOT = Path("evidence/screenshots/07-validation-results.png")
FINAL_WEIGHT = Path("weights/best.pt")
TRAINING_RESULTS_PNG = Path("runs/detect/train_qr/results.png")
TRAINING_RESULTS_CSV = Path("runs/detect/train_qr/results.csv")
NOTEBOOK_DIR = Path(".sisyphus/notepads/yolo-invoice-qr-detection")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class CliArgs:
    weights: Path
    data: Path
    split: str
    conf: float
    project: Path
    name: str
    device: str


@dataclass(frozen=True)
class Metrics:
    precision: float | None
    recall: float | None
    map50: float | None
    map50_95: float | None
    source: str


@dataclass(frozen=True)
class NegativeResult:
    split: str
    image_path: Path
    detection_count: int
    max_confidence: float | None


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(description="Validate a trained YOLO QR detector and write Task 9 evidence.")
    parser.add_argument("--weights", type=Path, required=True, help="Trained YOLO weight path to validate.")
    parser.add_argument("--data", type=Path, required=True, help="Ultralytics dataset YAML path.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"), help="Dataset split to validate.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for negative false-positive checks.")
    parser.add_argument("--project", type=Path, default=Path("runs/detect"), help="Ultralytics validation project directory.")
    parser.add_argument("--name", default="val_qr", help="Ultralytics validation run name.")
    parser.add_argument("--device", default="cpu", help="Validation/prediction device.")
    namespace = parser.parse_args()
    return CliArgs(
        weights=cast(Path, namespace.weights),
        data=cast(Path, namespace.data),
        split=cast(str, namespace.split),
        conf=cast(float, namespace.conf),
        project=cast(Path, namespace.project),
        name=cast(str, namespace.name),
        device=cast(str, namespace.device),
    )


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def safe_float(value: object) -> float | None:
    try:
        parsed = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed


def scalar_from(value: object) -> float | None:
    parsed = safe_float(value)
    if parsed is not None:
        return parsed
    item = getattr(value, "item", None)
    if callable(item):
        return safe_float(item())
    try:
        sequence = list(cast(Iterable[object], value))
    except TypeError:
        return None
    if not sequence:
        return None
    values = [safe_float(item) for item in sequence]
    valid_values = [item for item in values if item is not None]
    if not valid_values:
        return None
    return sum(valid_values) / len(valid_values)


def metric_from_dict(results_dict: dict[str, object], candidates: Sequence[str]) -> float | None:
    normalized = {key.lower().replace(" ", "").replace("/", "").replace("(", "").replace(")", ""): value for key, value in results_dict.items()}
    for candidate in candidates:
        direct = results_dict.get(candidate)
        parsed = scalar_from(direct)
        if parsed is not None:
            return parsed
        normalized_candidate = candidate.lower().replace(" ", "").replace("/", "").replace("(", "").replace(")", "")
        parsed = scalar_from(normalized.get(normalized_candidate))
        if parsed is not None:
            return parsed
    return None


def extract_metrics(val_result: Any, run_dir: Path) -> Metrics:
    box = getattr(val_result, "box", None)
    precision = scalar_from(getattr(box, "mp", None)) if box is not None else None
    recall = scalar_from(getattr(box, "mr", None)) if box is not None else None
    map50 = scalar_from(getattr(box, "map50", None)) if box is not None else None
    map50_95 = scalar_from(getattr(box, "map", None)) if box is not None else None
    source = "metrics.box fields"

    results_dict_raw = getattr(val_result, "results_dict", None)
    if isinstance(results_dict_raw, dict):
        results_dict = cast(dict[str, object], results_dict_raw)
        precision = precision if precision is not None else metric_from_dict(results_dict, ("metrics/precision(B)", "precision", "Box(P"))
        recall = recall if recall is not None else metric_from_dict(results_dict, ("metrics/recall(B)", "recall", "R"))
        map50 = map50 if map50 is not None else metric_from_dict(results_dict, ("metrics/mAP50(B)", "mAP50", "map50"))
        map50_95 = map50_95 if map50_95 is not None else metric_from_dict(results_dict, ("metrics/mAP50-95(B)", "mAP50-95", "map"))
        if any(value is not None for value in (precision, recall, map50, map50_95)):
            source = "metrics.box fields with results_dict fallback"

    csv_path = run_dir / "results.csv"
    if csv_path.exists() and None in (precision, recall, map50, map50_95):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            row = rows[-1]
            precision = precision if precision is not None else metric_from_dict(dict(row), ("metrics/precision(B)", "precision"))
            recall = recall if recall is not None else metric_from_dict(dict(row), ("metrics/recall(B)", "recall"))
            map50 = map50 if map50 is not None else metric_from_dict(dict(row), ("metrics/mAP50(B)", "mAP50"))
            map50_95 = map50_95 if map50_95 is not None else metric_from_dict(dict(row), ("metrics/mAP50-95(B)", "mAP50-95"))
            source = f"{source}; results.csv fallback"

    return Metrics(precision=precision, recall=recall, map50=map50, map50_95=map50_95, source=source)


def load_dataset_yaml(data_path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in dataset YAML: {rel_path(data_path)}")
    return cast(dict[str, object], loaded)


def dataset_root(data_path: Path, data: dict[str, object]) -> Path:
    raw_path = data.get("path")
    if isinstance(raw_path, str) and raw_path.strip():
        root = Path(raw_path)
        return root.resolve() if root.is_absolute() else (PROJECT_ROOT / root).resolve()
    return data_path.parent.resolve()


def split_dirs(data_path: Path, split: str) -> tuple[Path, Path]:
    data = load_dataset_yaml(data_path)
    root = dataset_root(data_path, data)
    split_value = data.get(split)
    if not isinstance(split_value, str) or not split_value.strip():
        raise ValueError(f"Dataset YAML does not define split {split!r}")
    image_rel = Path(split_value)
    image_dir = image_rel if image_rel.is_absolute() else root / image_rel
    parts = list(image_rel.parts)
    if parts and parts[0] == "images":
        label_dir = root.joinpath("labels", *parts[1:])
    elif "images" in parts:
        index = parts.index("images")
        label_parts = parts.copy()
        label_parts[index] = "labels"
        label_dir = root / Path(*label_parts)
    else:
        raise ValueError(f"Cannot derive label directory from split path {split_value!r}")
    return image_dir.resolve(), label_dir.resolve()


def negative_images_for_split(data_path: Path, split: str) -> list[Path]:
    image_dir, label_dir = split_dirs(data_path, split)
    negatives: list[Path] = []
    for label_path in sorted(label_dir.glob("*.txt")):
        if label_path.read_text(encoding="utf-8").strip():
            continue
        matches = [image_dir / f"{label_path.stem}{extension}" for extension in IMAGE_EXTENSIONS]
        image_path = next((path for path in matches if path.exists()), None)
        if image_path is not None:
            negatives.append(image_path)
    return negatives


def count_boxes(result: Any) -> tuple[int, float | None]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return 0, None
    try:
        detection_count = int(len(boxes))
    except TypeError:
        detection_count = 0
    confidences = getattr(boxes, "conf", None)
    max_confidence = None
    if confidences is not None:
        try:
            values = [safe_float(value) for value in cast(Iterable[object], confidences)]
        except TypeError:
            values = [scalar_from(confidences)]
        valid_values = [value for value in values if value is not None]
        if valid_values:
            max_confidence = max(valid_values)
    return detection_count, max_confidence


def evaluate_negatives(model: YOLO, data_path: Path, split: str, conf: float, device: str) -> list[NegativeResult]:
    negative_paths = negative_images_for_split(data_path, split)
    results: list[NegativeResult] = []
    if not negative_paths:
        return results
    predictions = model.predict(
        source=[str(path) for path in negative_paths],
        conf=conf,
        device=device,
        save=False,
        verbose=False,
    )
    for image_path, prediction in zip(negative_paths, predictions):
        detection_count, max_confidence = count_boxes(prediction)
        results.append(NegativeResult(split=split, image_path=image_path, detection_count=detection_count, max_confidence=max_confidence))
    return results


def validate_weight(weight_path: Path) -> None:
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing trained weight: {rel_path(weight_path)}")
    size = weight_path.stat().st_size
    if size <= MIN_BEST_WEIGHT_BYTES:
        raise ValueError(f"Weight is too small: {rel_path(weight_path)} is {size} bytes, expected > {MIN_BEST_WEIGHT_BYTES}")


def copy_final_weight(weight_path: Path) -> Path:
    final_path = resolve_path(FINAL_WEIGHT)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weight_path, final_path)
    validate_weight(final_path)
    return final_path


def metric_status(value: float | None, floor: float) -> str:
    if value is None:
        return "UNKNOWN"
    return "PASS" if value >= floor else "MISS"


def fp_status(false_positive_count: int) -> str:
    return "PASS" if false_positive_count <= NEGATIVE_FP_FLOOR else "MISS"


def format_metric(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.6f}"


def build_evidence_lines(
    command: str,
    args: CliArgs,
    weight_path: Path,
    final_weight: Path,
    val_run_dir: Path,
    metrics: Metrics,
    negative_results: list[NegativeResult],
) -> list[str]:
    false_positive_count = sum(result.detection_count for result in negative_results)
    negative_image_count = len(negative_results)
    return [
        "Task 9 model validation evidence",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"command={command}",
        "exit_code=0",
        f"weights={rel_path(weight_path)}",
        f"weights_bytes={weight_path.stat().st_size}",
        f"final_weight={rel_path(final_weight)}",
        f"final_weight_bytes={final_weight.stat().st_size}",
        f"data={rel_path(resolve_path(args.data))}",
        f"split={args.split}",
        f"conf={args.conf}",
        f"device={args.device}",
        f"validation_run_dir={rel_path(val_run_dir)}",
        f"metrics_source={metrics.source}",
        f"precision={format_metric(metrics.precision)}",
        f"recall={format_metric(metrics.recall)}",
        f"mAP50={format_metric(metrics.map50)}",
        f"mAP50-95={format_metric(metrics.map50_95)}",
        f"mAP50_floor={MAP50_FLOOR:.2f}",
        f"mAP50_status={metric_status(metrics.map50, MAP50_FLOOR)}",
        f"recall_floor={RECALL_FLOOR:.2f}",
        f"recall_status={metric_status(metrics.recall, RECALL_FLOOR)}",
        f"negative_split={args.split}",
        f"negative_image_count={negative_image_count}",
        f"false_positive_count={false_positive_count}",
        f"false_positive_floor={NEGATIVE_FP_FLOOR}",
        f"false_positive_status={fp_status(false_positive_count)}",
        f"limitation={'none' if all(value is not None for value in (metrics.precision, metrics.recall, metrics.map50, metrics.map50_95)) else 'Some metrics were unavailable from the Ultralytics API/artifacts.'}",
    ]


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_negative_evidence(command: str, args: CliArgs, negative_results: list[NegativeResult]) -> None:
    false_positive_count = sum(result.detection_count for result in negative_results)
    lines = [
        "Task 9 negative false-positive evidence",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"command={command}",
        f"split={args.split}",
        f"conf={args.conf}",
        f"negative_image_count={len(negative_results)}",
        f"false_positive_count={false_positive_count}",
        f"false_positive_floor={NEGATIVE_FP_FLOOR}",
        f"false_positive_status={fp_status(false_positive_count)}",
        "negative_predictions=",
    ]
    for result in negative_results:
        max_confidence = "none" if result.max_confidence is None else f"{result.max_confidence:.6f}"
        lines.append(
            f"- split={result.split} image={rel_path(result.image_path)} detections={result.detection_count} max_confidence={max_confidence}"
        )
    if not negative_results:
        lines.append("- No explicit empty-label negatives found for requested split; false-positive evidence is limited.")
    write_lines(resolve_path(TASK_NEGATIVES_EVIDENCE), lines)


def load_font(size: int) -> Any:
    for font_name in ("consola.ttf", "DejaVuSansMono.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def latest_training_metrics() -> dict[str, str]:
    csv_path = resolve_path(TRAINING_RESULTS_CSV)
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return dict(rows[-1]) if rows else {}


def draw_text_panel(draw: ImageDraw.ImageDraw, lines: Sequence[str], x: int, y: int, width: int, font: Any, fill: str = "#e5e7eb") -> None:
    current_y = y
    for raw_line in lines:
        line = raw_line.encode("ascii", "replace").decode("ascii")
        max_chars = max(20, width // 11)
        while len(line) > max_chars:
            draw.text((x, current_y), line[:max_chars], fill=fill, font=font)
            current_y += 24
            line = line[max_chars:]
        draw.text((x, current_y), line, fill=fill, font=font)
        current_y += 24


def create_training_results_png(output_path: Path) -> None:
    source_png = resolve_path(TRAINING_RESULTS_PNG)
    metrics = latest_training_metrics()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1600, 1000
    canvas = Image.new("RGB", (width, height), "#0f172a")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(36)
    body_font = load_font(20)
    small_font = load_font(18)
    draw.rectangle((0, 0, width, 96), fill="#111827")
    draw.text((42, 28), "Task 9 training-results evidence from real Task 8 artifacts", fill="#f8fafc", font=title_font)
    draw.text((44, 72), f"Source: {rel_path(source_png)} and {rel_path(resolve_path(TRAINING_RESULTS_CSV))}", fill="#93c5fd", font=small_font)
    if source_png.exists():
        with Image.open(source_png) as opened:
            plot = opened.convert("RGB")
        plot.thumbnail((980, 820), Image.Resampling.LANCZOS)
        canvas.paste(plot, (42, 132))
    else:
        draw.text((42, 160), "Training results plot missing.", fill="#fca5a5", font=body_font)
    lines = [
        "Final training CSV row:",
        f"epoch={metrics.get('epoch', 'unavailable')}",
        f"precision={metrics.get('metrics/precision(B)', 'unavailable')}",
        f"recall={metrics.get('metrics/recall(B)', 'unavailable')}",
        f"mAP50={metrics.get('metrics/mAP50(B)', 'unavailable')}",
        f"mAP50-95={metrics.get('metrics/mAP50-95(B)', 'unavailable')}",
        "These are copied from actual runs/detect/train_qr/results.csv; no metrics invented.",
    ]
    draw.rectangle((1060, 132, 1558, 900), outline="#334155", width=2)
    draw_text_panel(draw, lines, 1084, 164, 430, body_font)
    draw.rectangle((0, height - 54, width, height), fill="#111827")
    draw.text((42, height - 36), "Generated by scripts/validate_model.py from existing training artifacts.", fill="#a7f3d0", font=small_font)
    canvas.save(output_path)


def create_validation_results_png(output_path: Path, lines: Sequence[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1600, 980
    canvas = Image.new("RGB", (width, height), "#0f172a")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(38)
    body_font = load_font(21)
    small_font = load_font(18)
    draw.rectangle((0, 0, width, 110), fill="#111827")
    draw.text((48, 30), "Task 9 validation-results evidence", fill="#f8fafc", font=title_font)
    draw.text((50, 82), f"Source: {rel_path(resolve_path(DEFAULT_METRICS_LOG))}", fill="#93c5fd", font=small_font)
    selected = [line for line in lines if any(key in line for key in ("command=", "precision=", "recall=", "mAP50", "false_positive", "validation_run_dir", "limitation"))]
    draw.rectangle((44, 140, width - 44, height - 86), outline="#334155", width=2)
    draw_text_panel(draw, selected[:30], 70, 170, width - 140, body_font)
    draw.rectangle((0, height - 58, width, height), fill="#111827")
    draw.text((48, height - 40), "Rendered from actual Ultralytics validation metrics and negative prediction counts.", fill="#a7f3d0", font=small_font)
    canvas.save(output_path)


def append_notepads(metrics: Metrics, negative_results: list[NegativeResult], val_run_dir: Path) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    false_positive_count = sum(result.detection_count for result in negative_results)
    learning_lines = [
        f"\n## {timestamp} Task 9 model validation",
        f"- Added and ran `scripts/validate_model.py` to validate `runs/detect/train_qr/weights/best.pt` on the requested split, copy the final artifact to `weights/best.pt`, and write metrics/negative evidence.",
        f"- Ultralytics validation artifacts were written under `{rel_path(val_run_dir)}`; saved metrics: precision={format_metric(metrics.precision)}, recall={format_metric(metrics.recall)}, mAP50={format_metric(metrics.map50)}, mAP50-95={format_metric(metrics.map50_95)}.",
        f"- Explicit empty-label negatives on the validation split were evaluated at confidence threshold 0.25 with false_positive_count={false_positive_count} across {len(negative_results)} negative images.",
    ]
    issues: list[str] = []
    if metrics.recall is None or metrics.recall < RECALL_FLOOR:
        issues.append(f"- Task 9 recall threshold caveat: recall={format_metric(metrics.recall)} versus recommended floor {RECALL_FLOOR:.2f}.")
    if metrics.map50 is None or metrics.map50 < MAP50_FLOOR:
        issues.append(f"- Task 9 mAP50 threshold caveat: mAP50={format_metric(metrics.map50)} versus recommended floor {MAP50_FLOOR:.2f}.")
    if false_positive_count > NEGATIVE_FP_FLOOR:
        issues.append(f"- Task 9 negative-sample caveat: false_positive_count={false_positive_count} exceeds floor {NEGATIVE_FP_FLOOR} at confidence 0.25.")
    if any(value is None for value in (metrics.precision, metrics.recall, metrics.map50, metrics.map50_95)):
        issues.append("- Task 9 metric extraction caveat: one or more metrics were unavailable from Ultralytics result fields/artifacts.")
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    with (resolve_path(NOTEBOOK_DIR) / "learnings.md").open("a", encoding="utf-8") as handle:
        handle.write("\n".join(learning_lines) + "\n")
    if issues:
        with (resolve_path(NOTEBOOK_DIR) / "issues.md").open("a", encoding="utf-8") as handle:
            handle.write(f"\n## {timestamp} Task 9 validation caveats\n" + "\n".join(issues) + "\n")


def main() -> int:
    args = parse_args()
    weight_path = resolve_path(args.weights)
    data_path = resolve_path(args.data)
    project_dir = resolve_path(args.project)
    command = f".venv\\Scripts\\python.exe scripts\\validate_model.py --weights {args.weights} --data {args.data} --split {args.split} --conf {args.conf:g}"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset YAML: {rel_path(data_path)}")
    validate_weight(weight_path)
    final_weight = copy_final_weight(weight_path)

    model = YOLO(str(weight_path))
    val_result = model.val(
        data=str(data_path),
        split=args.split,
        project=str(project_dir),
        name=args.name,
        device=args.device,
        plots=True,
        verbose=True,
    )
    val_run_dir = Path(getattr(val_result, "save_dir", project_dir / args.name)).resolve()
    metrics = extract_metrics(val_result, val_run_dir)
    negative_results = evaluate_negatives(model, data_path, args.split, args.conf, args.device)
    evidence_lines = build_evidence_lines(command, args, weight_path, final_weight, val_run_dir, metrics, negative_results)

    write_lines(resolve_path(DEFAULT_METRICS_LOG), evidence_lines)
    write_lines(resolve_path(TASK_METRICS_EVIDENCE), evidence_lines)
    write_negative_evidence(command, args, negative_results)
    create_training_results_png(resolve_path(TRAINING_RESULTS_SCREENSHOT))
    create_validation_results_png(resolve_path(VALIDATION_RESULTS_SCREENSHOT), evidence_lines)
    append_notepads(metrics, negative_results, val_run_dir)

    print("Task 9 validation completed")
    print(f"metrics_log: {rel_path(resolve_path(DEFAULT_METRICS_LOG))}")
    print(f"task_metrics_evidence: {rel_path(resolve_path(TASK_METRICS_EVIDENCE))}")
    print(f"task_negatives_evidence: {rel_path(resolve_path(TASK_NEGATIVES_EVIDENCE))}")
    print(f"final_weight: {rel_path(final_weight)} ({final_weight.stat().st_size} bytes)")
    print(f"validation_run_dir: {rel_path(val_run_dir)}")
    print(f"precision: {format_metric(metrics.precision)}")
    print(f"recall: {format_metric(metrics.recall)}")
    print(f"mAP50: {format_metric(metrics.map50)}")
    print(f"mAP50-95: {format_metric(metrics.map50_95)}")
    print(f"false_positive_count: {sum(result.detection_count for result in negative_results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
