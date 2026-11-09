from __future__ import annotations

from pathlib import Path

from torchvision import datasets, transforms

IMAGE_SIZE = 32
CLASS_NAMES = [str(index) for index in range(10)]


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomAffine(degrees=8, translate=(0.08, 0.08)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
        ]
    )


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )


def load_image_folder_dataset(root: Path, train: bool) -> datasets.ImageFolder:
    transform = train_transform() if train else eval_transform()
    return datasets.ImageFolder(root=str(root), transform=transform)


def expected_dataset_layout() -> str:
    return """data/
  train/
    0/ 1/ 2/ ... 9/
  val/
    0/ 1/ 2/ ... 9/
  test/
    0/ 1/ 2/ ... 9/"""
