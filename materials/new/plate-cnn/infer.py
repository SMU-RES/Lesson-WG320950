from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from plate_cnn.data import eval_transform
from plate_cnn.model import SimplePlateCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a single plate digit image.")
    parser.add_argument("image", type=Path, help="Path to a single cropped digit image.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/best_model.pth"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    class_names = checkpoint["class_names"]
    model = SimplePlateCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = Image.open(args.image).convert("RGB")
    transformed = eval_transform()(image)
    if not isinstance(transformed, torch.Tensor):
        raise TypeError("Expected eval_transform() to return a torch.Tensor.")
    tensor = transformed.unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_index = int(probabilities.argmax().item())

    print(f"Image: {args.image}")
    print(f"Predicted class: {class_names[predicted_index]}")
    print(f"Confidence: {probabilities[predicted_index].item():.4f}")


if __name__ == "__main__":
    main()
