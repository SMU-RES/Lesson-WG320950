from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn


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


def preprocess_image(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("L").resize((32, 32))
    array = np.asarray(image, dtype=np.float32)
    if array.max() > 0:
        array = array / array.max()
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    return tensor


def predict(model_path: Path, image_path: Path) -> tuple[int, float, list[float]]:
    model = DigitCNN()
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    image_tensor = preprocess_image(image_path)
    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        predicted_class = int(probabilities.argmax().item())
        confidence = float(probabilities[predicted_class].item())

    return predicted_class, confidence, probabilities.tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer a digit class from a new license-plate-style image.")
    parser.add_argument("--image", type=Path, required=True, help="Path to an input digit image.")
    parser.add_argument("--model", type=Path, default=Path("outputs/digit_cnn.pt"), help="Path to model weights.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predicted_digit, confidence_score, class_probabilities = predict(args.model, args.image)
    probability_text = ", ".join(
        f"{digit}:{probability:.3f}" for digit, probability in enumerate(class_probabilities)
    )
    print(f"Input image: {args.image}")
    print(f"Predicted digit: {predicted_digit}")
    print(f"Confidence: {confidence_score:.4f}")
    print(f"Class probabilities: {probability_text}")
