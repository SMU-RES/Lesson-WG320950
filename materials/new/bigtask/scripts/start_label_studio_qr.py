from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "runtime" / "labeling" / "label-studio"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "runtime" / "labeling" / "images"
DEFAULT_EXPORT_DIR = PROJECT_ROOT / "runtime" / "labeling" / "exports"
LABEL_CONFIG = PROJECT_ROOT / "labeling" / "label_studio_qr.xml"


def resolve_label_studio_command() -> list[str]:
    executable = shutil.which("label-studio")
    if executable:
        return [executable]
    local_executable = Path.home() / ".local" / "bin" / "label-studio.exe"
    if local_executable.exists():
        return [str(local_executable)]
    return ["uv", "tool", "run", "--offline", "label-studio"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a local Label Studio server for QR YOLO annotation.")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the local Label Studio server.")
    parser.add_argument("--port", default="8080", help="Port for the local Label Studio server.")
    parser.add_argument("--project-name", default="invoice-qr-labeling", help="Label Studio project namespace.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Local Label Studio data directory.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="Suggested directory for images to annotate.")
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR, help="Suggested directory for exports.")
    return parser.parse_args(argv)


def project_state_dir(args: argparse.Namespace) -> Path:
    project_name = Path(str(args.project_name))
    if project_name.is_absolute():
        return project_name
    return Path(args.data_dir) / project_name


def build_label_studio_command(args: argparse.Namespace, base_command: list[str] | None = None) -> list[str]:
    return [
        *(base_command or resolve_label_studio_command()),
        "start",
        str(project_state_dir(args)),
        "--no-browser",
        "--internal-host",
        str(args.host),
        "--port",
        str(args.port),
        "--data-dir",
        str(args.data_dir),
    ]


def build_label_studio_init_command(args: argparse.Namespace, base_command: list[str] | None = None) -> list[str]:
    return [
        *(base_command or resolve_label_studio_command()),
        "init",
        str(project_state_dir(args)),
        "--quiet",
        "--label-config",
        str(LABEL_CONFIG),
        "--data-dir",
        str(args.data_dir),
    ]


def build_label_studio_env(project_root: Path = PROJECT_ROOT) -> dict[str, str]:
    env = os.environ.copy()
    env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
    env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = str(project_root)
    return env


def ensure_labeling_dirs(args: argparse.Namespace) -> None:
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.image_dir.mkdir(parents=True, exist_ok=True)
    args.export_dir.mkdir(parents=True, exist_ok=True)
    project_state_dir(args).parent.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_labeling_dirs(args)
    print("Label Studio local QR annotation environment")
    print(f"URL: http://{args.host}:{args.port}")
    print(f"Label config: {LABEL_CONFIG}")
    print(f"Project state: {project_state_dir(args)}")
    print(f"Put images to annotate under: {args.image_dir}")
    print(f"Export YOLO labels under: {args.export_dir}")
    print("Use one rectangle label only: qr")
    if not project_state_dir(args).exists():
        init_result = subprocess.run(build_label_studio_init_command(args), cwd=PROJECT_ROOT, env=build_label_studio_env())
        if init_result.returncode != 0:
            return init_result.returncode
    return subprocess.run(build_label_studio_command(args), cwd=PROJECT_ROOT, env=build_label_studio_env()).returncode


if __name__ == "__main__":
    raise SystemExit(main())
