from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, TypedDict

import torch
from torch import nn
from torch.utils.data import DataLoader

from plate_cnn.data import IMAGE_SIZE, expected_dataset_layout, load_image_folder_dataset
from plate_cnn.model import SimplePlateCNN
from plate_cnn.utils import ensure_dir, plot_training_curves, save_checkpoint, save_json


class TrainArgs(TypedDict):
    data_root: Path
    output_dir: Path
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="Train a simple CNN for plate digit classification.")
    parser.add_argument("--data-root", type=Path, default=Path("data/demo_digits"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    namespace = parser.parse_args()
    return {
        "data_root": namespace.data_root,
        "output_dir": namespace.output_dir,
        "epochs": namespace.epochs,
        "batch_size": namespace.batch_size,
        "learning_rate": namespace.learning_rate,
        "seed": namespace.seed,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_dataset_layout(data_root: Path) -> None:
    required_dirs = [data_root / split for split in ("train", "val", "test")]
    missing = [str(path) for path in required_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset directories: "
            + ", ".join(missing)
            + "\nExpected layout:\n"
            + expected_dataset_layout()
        )


def run_epoch(
    model: SimplePlateCNN,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(mode=is_training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if optimizer is not None:
            optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def evaluate_test_set(
    model: SimplePlateCNN,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    loss, acc = run_epoch(model, loader, criterion, device)
    return {"test_loss": loss, "test_accuracy": acc}


def main() -> None:
    args = parse_args()
    set_seed(args["seed"])
    check_dataset_layout(args["data_root"])

    output_dir = ensure_dir(args["output_dir"])
    results_dir = ensure_dir(output_dir / "results")
    checkpoint_path = output_dir / "best_model.pth"

    train_dataset = load_image_folder_dataset(args["data_root"] / "train", train=True)
    val_dataset = load_image_folder_dataset(args["data_root"] / "val", train=False)
    test_dataset = load_image_folder_dataset(args["data_root"] / "test", train=False)

    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimplePlateCNN(num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"])

    history: dict[str, list[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0

    print(f"Using device: {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    for epoch in range(1, args["epochs"] + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{args['epochs']} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, checkpoint_path, train_dataset.classes, IMAGE_SIZE)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_test_set(model, test_loader, criterion, device)

    plot_training_curves(history, results_dir)
    summary = {
        "data_root": str(args["data_root"]),
        "epochs": args["epochs"],
        "batch_size": args["batch_size"],
        "learning_rate": args["learning_rate"],
        "image_size": IMAGE_SIZE,
        "classes": train_dataset.classes,
        "best_val_acc": best_val_acc,
        **test_metrics,
    }
    save_json(results_dir / "training_summary.json", summary)

    print("Training finished.")
    print(f"Model saved to: {checkpoint_path.resolve()}")
    print(f"Results saved to: {results_dir.resolve()}")
    print(f"Test accuracy: {test_metrics['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
