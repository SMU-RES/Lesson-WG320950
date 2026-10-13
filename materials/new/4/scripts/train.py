# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import contextlib
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Iterator, TextIO, cast

import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "yolo11n.pt"
DEFAULT_DATA = Path("dataset/data.yaml")
DEFAULT_PROJECT = Path("runs/detect")
DEFAULT_NAME = "train_qr"
DEFAULT_LOG = Path("evidence/logs/train.log")
DEFAULT_SCREENSHOT = Path("evidence/screenshots/05-training-process.png")
BEST_WEIGHT_EVIDENCE = Path(".sisyphus/evidence/task-8-best-weight.txt")
ARTIFACT_EVIDENCE = Path(".sisyphus/evidence/task-8-artifacts.txt")
MIN_BEST_WEIGHT_BYTES = 1_048_576


@dataclass(frozen=True)
class CliArgs:
    model: str
    data: Path
    epochs: int
    imgsz: int
    batch: int
    patience: int
    workers: int
    project: Path
    name: str
    device: str
    log: Path
    screenshot: Path


class Tee(TextIO):
    def __init__(self, *streams: IO[str]) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    @property
    def encoding(self) -> str:
        return getattr(self._streams[0], "encoding", "utf-8") or "utf-8"


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(description="Transfer-train a YOLO QR detector with Ultralytics.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Pretrained Ultralytics weight file/name.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--patience", type=int, default=5, help="Early-stopping patience.")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="Ultralytics project directory.")
    parser.add_argument("--name", default=DEFAULT_NAME, help="Ultralytics run name prefix.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device. Use 'auto' to select CUDA 0 when available, otherwise CPU.",
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="Training log output path.")
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=DEFAULT_SCREENSHOT,
        help="PNG evidence rendered from the training log.",
    )
    namespace = parser.parse_args()
    return CliArgs(
        model=cast(str, namespace.model),
        data=cast(Path, namespace.data),
        epochs=cast(int, namespace.epochs),
        imgsz=cast(int, namespace.imgsz),
        batch=cast(int, namespace.batch),
        patience=cast(int, namespace.patience),
        workers=cast(int, namespace.workers),
        project=cast(Path, namespace.project),
        name=cast(str, namespace.name),
        device=cast(str, namespace.device),
        log=cast(Path, namespace.log),
        screenshot=cast(Path, namespace.screenshot),
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


def resolve_device(requested: str) -> tuple[str, bool]:
    cuda_available = torch.cuda.is_available()
    if requested.lower() == "auto":
        return ("0" if cuda_available else "cpu", cuda_available)
    return requested, cuda_available


@contextlib.contextmanager
def tee_output(log_path: Path) -> Iterator[None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def latest_run_dir(project_dir: Path, name: str, started_at: float) -> Path | None:
    if not project_dir.exists():
        return None
    candidates = [
        path
        for path in project_dir.iterdir()
        if path.is_dir() and path.name.startswith(name) and path.stat().st_mtime >= started_at - 5
    ]
    if not candidates:
        candidates = [path for path in project_dir.iterdir() if path.is_dir() and path.name.startswith(name)]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_run_dir(train_result: Any, project_dir: Path, name: str, started_at: float) -> Path:
    save_dir = getattr(train_result, "save_dir", None)
    if save_dir:
        return Path(save_dir).resolve()
    discovered = latest_run_dir(project_dir, name, started_at)
    if discovered is None:
        raise FileNotFoundError(f"Could not locate Ultralytics run directory under {rel_path(project_dir)}")
    return discovered.resolve()


def required_artifacts(run_dir: Path) -> dict[str, Path]:
    return {
        "best_weight": run_dir / "weights" / "best.pt",
        "results_png": run_dir / "results.png",
        "results_csv": run_dir / "results.csv",
        "args_yaml": run_dir / "args.yaml",
    }


def verify_artifacts(run_dir: Path) -> dict[str, Path]:
    artifacts = required_artifacts(run_dir)
    missing = [name for name, path in artifacts.items() if not path.exists()]
    if missing:
        missing_text = ", ".join(f"{name}={rel_path(artifacts[name])}" for name in missing)
        raise FileNotFoundError(f"Missing required training artifacts: {missing_text}")

    best_weight = artifacts["best_weight"]
    best_size = best_weight.stat().st_size
    if best_size <= MIN_BEST_WEIGHT_BYTES:
        raise ValueError(
            f"best.pt is too small: {rel_path(best_weight)} is {best_size} bytes, "
            f"expected > {MIN_BEST_WEIGHT_BYTES} bytes"
        )
    return artifacts


def write_evidence(
    args: CliArgs,
    run_dir: Path,
    artifacts: dict[str, Path],
    resolved_device: str,
    cuda_available: bool,
) -> None:
    best_weight = artifacts["best_weight"]
    best_size = best_weight.stat().st_size
    BEST_WEIGHT_EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    BEST_WEIGHT_EVIDENCE.write_text(
        "Task 8 best-weight evidence\n"
        f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}\n"
        f"model: {args.model}\n"
        f"data: {rel_path(resolve_path(args.data))}\n"
        f"run_dir: {rel_path(run_dir)}\n"
        f"best_weight: {rel_path(best_weight)}\n"
        f"best_weight_bytes: {best_size}\n"
        f"threshold_bytes: {MIN_BEST_WEIGHT_BYTES}\n"
        f"passes_size_check: {best_size > MIN_BEST_WEIGHT_BYTES}\n"
        f"resolved_device: {resolved_device}\n"
        f"torch_cuda_available: {cuda_available}\n",
        encoding="utf-8",
    )

    lines = [
        "Task 8 training-artifact evidence",
        f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}",
        f"model: {args.model}",
        f"run_dir: {rel_path(run_dir)}",
        f"train_log: {rel_path(resolve_path(args.log))}",
        f"training_process_png: {rel_path(resolve_path(args.screenshot))}",
    ]
    for name, path in artifacts.items():
        lines.append(f"{name}: {rel_path(path)} ({path.stat().st_size} bytes)")
    ARTIFACT_EVIDENCE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_font(size: int) -> Any:
    for font_name in ("consola.ttf", "DejaVuSansMono.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def interesting_log_lines(log_text: str, max_lines: int) -> list[str]:
    keywords = (
        "Task 8",
        "Ultralytics",
        "Transferred",
        "Starting training",
        "Epoch",
        "Model summary",
        "optimizer",
        "results saved",
        "Validating",
        "best.pt",
        "resolved_device",
        "CPU-only",
        "CUDA",
    )
    selected = [line.rstrip() for line in log_text.splitlines() if any(keyword in line for keyword in keywords)]
    if len(selected) < 12:
        selected = [line.rstrip() for line in log_text.splitlines() if line.strip()]
    return selected[-max_lines:]


def draw_training_process_png(log_path: Path, screenshot_path: Path, run_dir: Path) -> None:
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    lines = interesting_log_lines(log_text, max_lines=30)
    width = 1600
    height = 980
    image = Image.new("RGB", (width, height), "#0f172a")
    draw = ImageDraw.Draw(image)
    title_font = load_font(40)
    body_font = load_font(22)
    small_font = load_font(18)

    draw.rectangle((0, 0, width, 120), fill="#111827")
    draw.text((48, 32), "Task 8 Ultralytics transfer training evidence", fill="#e5e7eb", font=title_font)
    draw.text((52, 86), f"Source: {rel_path(log_path)} | Run: {rel_path(run_dir)}", fill="#93c5fd", font=small_font)

    y = 150
    draw.text((52, y), "Real training log excerpt", fill="#fbbf24", font=body_font)
    y += 38
    for line in lines:
        safe_line = line.encode("ascii", "replace").decode("ascii")
        if len(safe_line) > 132:
            safe_line = safe_line[:129] + "..."
        draw.text((72, y), safe_line, fill="#d1d5db", font=small_font)
        y += 25
        if y > height - 90:
            break

    footer = "Generated from evidence/logs/train.log after the final YOLO training run; no synthetic metrics were invented."
    draw.rectangle((0, height - 58, width, height), fill="#111827")
    draw.text((52, height - 40), footer, fill="#a7f3d0", font=small_font)
    image.save(screenshot_path)


def main() -> int:
    args = parse_args()
    data_path = resolve_path(args.data)
    project_dir = resolve_path(args.project)
    log_path = resolve_path(args.log)
    screenshot_path = resolve_path(args.screenshot)
    resolved_device, cuda_available = resolve_device(args.device)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML does not exist: {rel_path(data_path)}")

    with tee_output(log_path):
        print("Task 8 Ultralytics transfer training")
        print(f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}")
        print(f"model: {args.model}")
        print(f"data: {rel_path(data_path)}")
        print(f"epochs: {args.epochs}")
        print(f"imgsz: {args.imgsz}")
        print(f"batch: {args.batch}")
        print(f"patience: {args.patience}")
        print(f"workers: {args.workers}")
        print(f"project: {rel_path(project_dir)}")
        print(f"name: {args.name}")
        print(f"torch_version: {torch.__version__}")
        print(f"torch_cuda_available: {cuda_available}")
        print(f"resolved_device: {resolved_device}")
        if resolved_device == "cpu":
            print(
                "CPU-only runtime constraints detected or requested; using reduced epochs/imgsz "
                f"for a short real transfer-training run (epochs={args.epochs}, imgsz={args.imgsz})."
            )
        print("Starting YOLO(model).train(...) with pretrained transfer weights.")

        started_at = time.time()
        model = YOLO(args.model)
        train_result = model.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            workers=args.workers,
            project=str(project_dir),
            name=args.name,
            device=resolved_device,
            plots=True,
            save=True,
            verbose=True,
        )

        run_dir = resolve_run_dir(train_result, project_dir, args.name, started_at)
        artifacts = verify_artifacts(run_dir)
        write_evidence(args, run_dir, artifacts, resolved_device, cuda_available)

        print("Training completed and required artifacts verified.")
        print(f"run_dir: {rel_path(run_dir)}")
        print(f"best_weight: {rel_path(artifacts['best_weight'])}")
        print(f"best_weight_bytes: {artifacts['best_weight'].stat().st_size}")
        print(f"results_png: {rel_path(artifacts['results_png'])}")
        print(f"results_csv: {rel_path(artifacts['results_csv'])}")
        print(f"args_yaml: {rel_path(artifacts['args_yaml'])}")
        print(f"best_weight_evidence: {rel_path(resolve_path(BEST_WEIGHT_EVIDENCE))}")
        print(f"artifact_evidence: {rel_path(resolve_path(ARTIFACT_EVIDENCE))}")

    draw_training_process_png(log_path, screenshot_path, run_dir)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"training_process_png: {rel_path(screenshot_path)}\n")
    print(f"Generated training-process evidence PNG: {rel_path(screenshot_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
