from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunArgs:
    dataset: str
    epochs: int


def run_step(command: list[str], workdir: Path) -> None:
    print(f">>> Running: {' '.join(command)}")
    completed = subprocess.run(command, cwd=workdir, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def parse_args() -> RunArgs:
    parser = argparse.ArgumentParser(description="Prepare dataset and train the plate digit CNN.")
    _ = parser.add_argument(
        "--dataset",
        choices=("public", "demo"),
        default="public",
        help="Dataset source to prepare before training.",
    )
    _ = parser.add_argument("--epochs", type=int, default=12)
    namespace = parser.parse_args()
    namespace_dict = vars(namespace)
    dataset_value: object = namespace_dict["dataset"]
    epochs_value: object = namespace_dict["epochs"]
    if not isinstance(dataset_value, str):
        raise TypeError("Expected dataset argument to be a string.")
    if not isinstance(epochs_value, int):
        raise TypeError("Expected epochs argument to be an integer.")
    return RunArgs(dataset=dataset_value, epochs=epochs_value)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    if args.dataset == "public":
        run_step([sys.executable, "prepare_public_dataset.py"], project_root)
        data_root = "data/public_digits"
    else:
        run_step([sys.executable, "make_demo_dataset.py"], project_root)
        data_root = "data/demo_digits"

    run_step([sys.executable, "train.py", "--data-root", data_root, "--epochs", str(args.epochs)], project_root)

    loss_curve = project_root / "outputs" / "results" / "loss_curve.png"
    print(f"Loss curve saved to: {loss_curve}")


if __name__ == "__main__":
    main()
