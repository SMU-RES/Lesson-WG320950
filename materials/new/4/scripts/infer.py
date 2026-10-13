from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any, cast

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO  # pyright: ignore[reportMissingImports]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIN_WEIGHT_BYTES = 1_048_576
TASK_INFERENCE_EVIDENCE = Path(".sisyphus/evidence/task-10-inference.txt")
TASK_NOQR_EVIDENCE = Path(".sisyphus/evidence/task-10-noqr.txt")
FINAL_INFERENCE_SCREENSHOT = Path("evidence/screenshots/08-final-inference.png")
NOTEBOOK_DIR = Path(".sisyphus/notepads/yolo-invoice-qr-detection")
DEFAULT_MANIFEST = Path("samples/inference/manifest.csv")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class CliArgs:
    weights: Path
    source: Path
    conf: float
    save: bool
    project: Path
    name: str
    device: str


@dataclass(frozen=True)
class ImagePrediction:
    image_path: Path
    detection_count: int
    max_confidence: float | None
    annotated_path: Path | None


@dataclass(frozen=True)
class ManifestRow:
    filename: str
    source_image: str
    source_label: str
    qr_present: bool


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(description="Run YOLO QR object detection inference and write Task 10 evidence.")
    parser.add_argument("--weights", type=Path, required=True, help="Trained YOLO weight path for inference.")
    parser.add_argument("--source", type=Path, required=True, help="Image file or directory to run inference on.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--save", action="store_true", help="Save annotated prediction images.")
    parser.add_argument("--project", type=Path, default=Path("runs/detect"), help="Ultralytics prediction project directory.")
    parser.add_argument("--name", default="predict_qr", help="Ultralytics prediction run name.")
    parser.add_argument("--device", default="cpu", help="Inference device.")
    namespace = parser.parse_args()
    return CliArgs(
        weights=cast(Path, namespace.weights),
        source=cast(Path, namespace.source),
        conf=cast(float, namespace.conf),
        save=cast(bool, namespace.save),
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


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def validate_weight(weight_path: Path) -> None:
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing trained weight: {rel_path(weight_path)}")
    size = weight_path.stat().st_size
    if size <= MIN_WEIGHT_BYTES:
        raise ValueError(f"Weight is too small: {rel_path(weight_path)} is {size} bytes, expected > {MIN_WEIGHT_BYTES}")


def image_inputs(source_path: Path) -> list[Path]:
    if not source_path.exists():
        raise FileNotFoundError(f"Missing inference source: {rel_path(source_path)}")
    if source_path.is_file():
        if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Inference source is not a supported image: {rel_path(source_path)}")
        return [source_path]
    images = sorted(path for path in source_path.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"No supported images found in inference source: {rel_path(source_path)}")
    return images


def safe_float(value: object) -> float | None:
    try:
        parsed = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed


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
            values = [safe_float(value) for value in cast(Any, confidences)]
        except TypeError:
            values = [safe_float(confidences)]
        valid_values = [value for value in values if value is not None]
        if valid_values:
            max_confidence = max(valid_values)
    return detection_count, max_confidence


def annotated_path_for(run_dir: Path, image_path: Path) -> Path | None:
    direct = run_dir / image_path.name
    if direct.exists():
        return direct
    matches = sorted(run_dir.glob(f"{image_path.stem}.*"))
    image_matches = [path for path in matches if path.suffix.lower() in IMAGE_EXTENSIONS]
    return image_matches[0] if image_matches else None


def actual_save_dir(results: list[Any], project_dir: Path, name: str) -> Path:
    for result in results:
        save_dir = getattr(result, "save_dir", None)
        if save_dir is not None:
            return Path(save_dir).resolve()
    return (project_dir / name).resolve()


def load_manifest(source_path: Path) -> dict[str, ManifestRow]:
    candidate_paths = []
    if source_path.is_dir():
        candidate_paths.append(source_path / "manifest.csv")
    candidate_paths.append(resolve_path(DEFAULT_MANIFEST))
    manifest_path = next((path for path in candidate_paths if path.exists()), None)
    if manifest_path is None:
        return {}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    manifest: dict[str, ManifestRow] = {}
    for row in rows:
        filename = (row.get("filename") or "").strip()
        if not filename:
            continue
        qr_present = (row.get("qr_present") or "").strip().lower() == "true"
        manifest[filename] = ManifestRow(
            filename=filename,
            source_image=(row.get("source_image") or "").strip(),
            source_label=(row.get("source_label") or "").strip(),
            qr_present=qr_present,
        )
    return manifest


def load_font(size: int) -> Any:
    for font_name in ("consola.ttf", "DejaVuSansMono.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], font: Any) -> None:
    x, y = xy
    try:
        bbox = draw.textbbox((x, y), text, font=font)
    except AttributeError:
        width = len(text) * 9
        bbox = (x, y, x + width, y + 18)
    draw.rectangle((bbox[0] - 6, bbox[1] - 4, bbox[2] + 6, bbox[3] + 4), fill="#111827")
    draw.text((x, y), text, fill="#f8fafc", font=font)


def create_contact_sheet(predictions: list[ImagePrediction], output_path: Path, run_dir: Path) -> None:
    annotated = [prediction for prediction in predictions if prediction.annotated_path is not None]
    if not annotated:
        raise ValueError("No annotated prediction images were available for the final inference contact sheet.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tile_w, tile_h = 420, 520
    columns = min(3, max(1, len(annotated)))
    rows = ceil(len(annotated) / columns)
    header_h = 96
    footer_h = 52
    canvas = Image.new("RGB", (columns * tile_w, header_h + rows * tile_h + footer_h), "#0f172a")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(28)
    body_font = load_font(16)
    small_font = load_font(14)
    draw.rectangle((0, 0, canvas.width, header_h), fill="#111827")
    draw.text((24, 22), "Task 10 final inference predictions", fill="#f8fafc", font=title_font)
    draw.text((26, 62), f"Actual Ultralytics output: {rel_path(run_dir)}", fill="#93c5fd", font=body_font)
    for index, prediction in enumerate(annotated):
        assert prediction.annotated_path is not None
        col = index % columns
        row = index // columns
        left = col * tile_w
        top = header_h + row * tile_h
        draw.rectangle((left + 12, top + 12, left + tile_w - 12, top + tile_h - 12), fill="#1e293b", outline="#334155", width=2)
        with Image.open(prediction.annotated_path) as opened:
            image = opened.convert("RGB")
        image.thumbnail((tile_w - 44, tile_h - 108), Image.Resampling.LANCZOS)
        image_left = left + (tile_w - image.width) // 2
        image_top = top + 54
        canvas.paste(image, (image_left, image_top))
        draw_label(draw, prediction.image_path.name, (left + 24, top + 24), small_font)
        max_conf = "none" if prediction.max_confidence is None else f"{prediction.max_confidence:.3f}"
        draw.text((left + 24, top + tile_h - 44), f"detections={prediction.detection_count} max_conf={max_conf}", fill="#a7f3d0", font=body_font)
    draw.rectangle((0, canvas.height - footer_h, canvas.width, canvas.height), fill="#111827")
    draw.text((24, canvas.height - 34), "Rendered from saved annotated PNG predictions; no fabricated detections.", fill="#cbd5e1", font=small_font)
    canvas.save(output_path)


def format_confidence(value: float | None) -> str:
    return "none" if value is None else f"{value:.6f}"


def build_command(args: CliArgs) -> str:
    parts = [
        r".venv\Scripts\python.exe",
        r"scripts\infer.py",
        "--weights",
        str(args.weights),
        "--source",
        str(args.source),
        "--conf",
        f"{args.conf:g}",
    ]
    if args.save:
        parts.append("--save")
    if args.project != Path("runs/detect"):
        parts.extend(["--project", str(args.project)])
    if args.name != "predict_qr":
        parts.extend(["--name", args.name])
    if args.device != "cpu":
        parts.extend(["--device", args.device])
    return " ".join(parts)


def write_inference_evidence(
    command: str,
    args: CliArgs,
    weight_path: Path,
    source_path: Path,
    run_dir: Path,
    predictions: list[ImagePrediction],
    manifest: dict[str, ManifestRow],
) -> None:
    annotated_count = sum(1 for prediction in predictions if prediction.annotated_path is not None and prediction.annotated_path.exists())
    prediction_by_name = {prediction.image_path.name: prediction for prediction in predictions}
    qr_rows = [row for row in manifest.values() if row.qr_present]
    noqr_rows = [row for row in manifest.values() if not row.qr_present]
    qr_positive_detected_count = sum(1 for row in qr_rows if prediction_by_name.get(row.filename) is not None and prediction_by_name[row.filename].detection_count > 0)
    lines = [
        "Task 10 final inference evidence",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"command={command}",
        "exit_code=0",
        f"weights={rel_path(weight_path)}",
        f"weights_bytes={weight_path.stat().st_size}",
        f"source={rel_path(source_path)}",
        f"conf={args.conf:g}",
        f"device={args.device}",
        f"save={args.save}",
        f"actual_output_dir={rel_path(run_dir)}",
        f"input_count={len(predictions)}",
        f"annotated_output_count={annotated_count}",
        f"manifest_qr_positive_count={len(qr_rows)}",
        f"manifest_noqr_count={len(noqr_rows)}",
        f"qr_positive_detected_count={qr_positive_detected_count}",
        "per_image_detection_counts=",
    ]
    for prediction in predictions:
        annotated = "none" if prediction.annotated_path is None else rel_path(prediction.annotated_path)
        lines.append(
            f"- image={rel_path(prediction.image_path)} detections={prediction.detection_count} max_confidence={format_confidence(prediction.max_confidence)} annotated={annotated}"
        )
    write_lines(resolve_path(TASK_INFERENCE_EVIDENCE), lines)


def write_noqr_evidence(
    command: str,
    args: CliArgs,
    manifest: dict[str, ManifestRow],
    predictions: list[ImagePrediction],
) -> int:
    prediction_by_name = {prediction.image_path.name: prediction for prediction in predictions}
    noqr_rows = [row for row in manifest.values() if not row.qr_present]
    false_positive_count = 0
    lines = [
        "Task 10 no-QR inference evidence",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"command={command}",
        f"conf={args.conf:g}",
        f"noqr_sample_count={len(noqr_rows)}",
        "noqr_predictions=",
    ]
    for row in noqr_rows:
        prediction = prediction_by_name.get(row.filename)
        if prediction is None:
            lines.append(f"- sample={row.filename} source_image={row.source_image} source_label={row.source_label} detections=missing max_confidence=none")
            continue
        false_positive_count += prediction.detection_count
        lines.append(
            f"- sample={rel_path(prediction.image_path)} source_image={row.source_image} source_label={row.source_label} detections={prediction.detection_count} max_confidence={format_confidence(prediction.max_confidence)}"
        )
    if not noqr_rows:
        lines.append("- No no-QR rows were present in the inference manifest; false-positive evidence is limited.")
    lines.append(f"false_positive_count={false_positive_count}")
    lines.append(f"false_positive_occurred={str(false_positive_count > 0).lower()}")
    write_lines(resolve_path(TASK_NOQR_EVIDENCE), lines)
    return false_positive_count


def append_notepads(predictions: list[ImagePrediction], run_dir: Path, false_positive_count: int, noqr_count: int) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    annotated_count = sum(1 for prediction in predictions if prediction.annotated_path is not None and prediction.annotated_path.exists())
    learning_lines = [
        f"\n## {timestamp} Task 10 final inference",
        f"- Added and ran `scripts/infer.py` for YOLO-only QR bounding-box inference using `weights/best.pt` on `samples/inference/` test-split samples.",
        f"- Saved Ultralytics annotated predictions under `{rel_path(run_dir)}` and generated `{FINAL_INFERENCE_SCREENSHOT.as_posix()}` as a contact sheet from those actual prediction images.",
        f"- Inference evidence records {len(predictions)} inputs, {annotated_count} annotated outputs, and no-QR false positives={false_positive_count} across {noqr_count} manifest no-QR samples at confidence 0.25.",
    ]
    append_lines(resolve_path(NOTEBOOK_DIR) / "learnings.md", learning_lines)
    issue_lines = [f"\n## {timestamp} Task 10 inference caveats"]
    if false_positive_count > 0:
        issue_lines.append(f"- Task 10 no-QR caveat: false_positive_count={false_positive_count} across {noqr_count} no-QR samples at confidence 0.25.")
    else:
        issue_lines.append(f"- Task 10 no-QR check found no false positives across {noqr_count} no-QR samples at confidence 0.25.")
    append_lines(resolve_path(NOTEBOOK_DIR) / "issues.md", issue_lines)


def run_inference(args: CliArgs) -> tuple[list[ImagePrediction], Path]:
    weight_path = resolve_path(args.weights)
    source_path = resolve_path(args.source)
    project_dir = resolve_path(args.project)
    validate_weight(weight_path)
    inputs = image_inputs(source_path)
    model = YOLO(str(weight_path))
    raw_results = model.predict(
        source=str(source_path),
        conf=args.conf,
        save=args.save,
        project=str(project_dir),
        name=args.name,
        device=args.device,
    )
    results = list(raw_results)
    run_dir = actual_save_dir(results, project_dir, args.name)
    predictions: list[ImagePrediction] = []
    for image_path, result in zip(inputs, results):
        detection_count, max_confidence = count_boxes(result)
        annotated_path = annotated_path_for(run_dir, image_path) if args.save else None
        predictions.append(
            ImagePrediction(
                image_path=image_path,
                detection_count=detection_count,
                max_confidence=max_confidence,
                annotated_path=annotated_path,
            )
        )
    return predictions, run_dir


def main() -> int:
    args = parse_args()
    weight_path = resolve_path(args.weights)
    source_path = resolve_path(args.source)
    command = build_command(args)
    predictions, run_dir = run_inference(args)
    manifest = load_manifest(source_path)
    write_inference_evidence(command, args, weight_path, source_path, run_dir, predictions, manifest)
    false_positive_count = write_noqr_evidence(command, args, manifest, predictions)
    noqr_count = sum(1 for row in manifest.values() if not row.qr_present)
    if args.save:
        create_contact_sheet(predictions, resolve_path(FINAL_INFERENCE_SCREENSHOT), run_dir)
    append_notepads(predictions, run_dir, false_positive_count, noqr_count)

    annotated_count = sum(1 for prediction in predictions if prediction.annotated_path is not None and prediction.annotated_path.exists())
    print("Task 10 inference completed")
    print(f"inference_evidence: {rel_path(resolve_path(TASK_INFERENCE_EVIDENCE))}")
    print(f"noqr_evidence: {rel_path(resolve_path(TASK_NOQR_EVIDENCE))}")
    print(f"screenshot: {rel_path(resolve_path(FINAL_INFERENCE_SCREENSHOT))}")
    print(f"actual_output_dir: {rel_path(run_dir)}")
    print(f"input_count: {len(predictions)}")
    print(f"annotated_output_count: {annotated_count}")
    print(f"noqr_false_positive_count: {false_positive_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
