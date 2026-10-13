from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    output_dir: Path


class DigitCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    _ = torch.manual_seed(seed)


def render_digit(digit: int, rng: np.random.Generator) -> np.ndarray:
    image = Image.new("L", (32, 32), color=0)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=26)
    text_box = draw.textbbox((0, 0), str(digit), font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    offset_x = int(rng.integers(-3, 4))
    offset_y = int(rng.integers(-3, 4))
    x = (32 - text_width) // 2 + offset_x
    y = (32 - text_height) // 2 + offset_y
    brightness = int(rng.integers(190, 256))
    draw.text((x, y), str(digit), fill=brightness, font=font)

    array = np.asarray(image, dtype=np.float32) / 255.0
    noise = rng.normal(0.0, 0.035, size=array.shape).astype(np.float32)
    return np.clip(array + noise, 0.0, 1.0).astype(np.float32)


def make_digit_dataset(samples_per_digit: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    labels: list[int] = []

    for digit in range(10):
        for _ in range(samples_per_digit):
            images.append(render_digit(digit, rng))
            labels.append(digit)

    image_array = np.stack(images).astype(np.float32)
    label_array = np.asarray(labels, dtype=np.int64)
    return image_array, label_array


def stratified_indices(labels: np.ndarray, seed: int, test_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for digit in range(10):
        digit_indices = np.flatnonzero(labels == digit)
        rng.shuffle(digit_indices)
        test_count = max(1, int(round(len(digit_indices) * test_ratio)))
        test_parts.append(digit_indices[:test_count])
        train_parts.append(digit_indices[test_count:])

    train_indices = np.concatenate(train_parts)
    test_indices = np.concatenate(test_parts)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices


def load_digit_data(
    seed: int,
    batch_size: int,
) -> tuple[
    DataLoader[tuple[torch.Tensor, ...]],
    DataLoader[tuple[torch.Tensor, ...]],
    torch.Tensor,
    torch.Tensor,
]:
    images, labels = make_digit_dataset(samples_per_digit=160, seed=seed)
    train_indices, test_indices = stratified_indices(labels, seed, test_ratio=0.2)

    x_train = images[train_indices]
    x_test = images[test_indices]
    y_train = labels[train_indices]
    y_test = labels[test_indices]

    train_x = torch.from_numpy(x_train).unsqueeze(1)
    test_x = torch.from_numpy(x_test).unsqueeze(1)
    train_y = torch.from_numpy(y_train)
    test_y = torch.from_numpy(y_test)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=256)
    return train_loader, test_loader, test_x, test_y


def evaluate(model: nn.Module, loader: DataLoader[tuple[torch.Tensor, ...]], loss_fn: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []

    with torch.inference_mode():
        for features, labels in loader:
            logits = model(features)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)
            predictions.extend(logits.argmax(dim=1).tolist())
            targets.extend(labels.tolist())

    average_loss = total_loss / len(targets)
    accuracy = calculate_accuracy(targets, predictions)
    return average_loss, accuracy


def calculate_accuracy(targets: list[int] | np.ndarray, predictions: list[int] | np.ndarray) -> float:
    correct = sum(1 for target, prediction in zip(targets, predictions) if target == prediction)
    return correct / len(targets)


def build_confusion_matrix(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    matrix = np.zeros((10, 10), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    return matrix


def train(config: TrainConfig) -> dict[str, list[float]]:
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader, test_x, test_y = load_digit_data(config.seed, config.batch_size)
    model = DigitCNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, config.epochs + 1):
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(model, train_loader, loss_fn)
        val_loss, val_acc = evaluate(model, test_loader, loss_fn)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    save_outputs(model, history, test_x, test_y, config.output_dir)
    return history


def save_outputs(
    model: DigitCNN,
    history: dict[str, list[float]],
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    output_dir: Path,
) -> None:
    model.eval()
    model_path = output_dir / "digit_cnn.pt"
    torch.save(model.state_dict(), model_path)

    with torch.inference_mode():
        logits = model(test_x)
        predictions = logits.argmax(dim=1)

    plot_training_curves(history, output_dir / "training_curves.png")
    plot_prediction_examples(
        test_x,
        test_y,
        predictions,
        output_dir / "prediction_examples.png",
    )
    save_sample_image(test_x[0], output_dir / "sample_digit.png")

    test_labels = test_y.numpy()
    predicted_labels = predictions.numpy()
    matrix = build_confusion_matrix(test_labels, predicted_labels)
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")

    final_acc = calculate_accuracy(test_labels, predicted_labels)
    print(f"\nModel saved to: {model_path}")
    print(f"Training curves saved to: {output_dir / 'training_curves.png'}")
    print(f"Prediction examples saved to: {output_dir / 'prediction_examples.png'}")
    print(f"Sample inference image saved to: {output_dir / 'sample_digit.png'}")
    print(f"Final validation accuracy: {final_acc:.4f}")


def plot_training_curves(history: dict[str, list[float]], output_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], marker="o", label="Train loss")
    plt.plot(epochs, history["val_loss"], marker="s", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], marker="o", label="Train accuracy")
    plt.plot(epochs, history["val_acc"], marker="s", label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Accuracy Curve")
    plt.ylim(0.0, 1.02)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_prediction_examples(
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    predictions: torch.Tensor,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 5))
    for index in range(12):
        image = test_x[index].squeeze(0).numpy()
        label = int(test_y[index])
        predicted = int(predictions[index])
        color = "green" if label == predicted else "red"

        plt.subplot(3, 4, index + 1)
        plt.imshow(image, cmap="gray")
        plt.title(f"True: {label}  Pred: {predicted}", color=color)
        plt.axis("off")

    plt.suptitle("License-Plate-Style Digit Classification Examples")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_sample_image(image_tensor: torch.Tensor, output_path: Path) -> None:
    image = image_tensor.squeeze(0).numpy()
    plt.imsave(output_path, image, cmap="gray")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a simple CNN for digit classification.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory.")
    args = parser.parse_args()
    return TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    train(parse_args())
