from __future__ import annotations

import argparse
from pathlib import Path

from scripts import start_label_studio_qr, validate_yolo_dataset


def write_dataset(root: Path, bad_label: str | None = None) -> Path:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        (root / "images" / split / f"{split}_invoice.png").write_bytes(b"image")
        label = bad_label if bad_label is not None and split == "train" else "0 0.5 0.5 0.25 0.25"
        (root / "labels" / split / f"{split}_invoice.txt").write_text(label + "\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: qr",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_validate_yolo_dataset_accepts_qr_dataset(tmp_path: Path) -> None:
    data_yaml = write_dataset(tmp_path)

    assert validate_yolo_dataset.validate_dataset(data_yaml) == []


def test_validate_yolo_dataset_rejects_wrong_class_id(tmp_path: Path) -> None:
    data_yaml = write_dataset(tmp_path, bad_label="1 0.5 0.5 0.25 0.25")

    failures = validate_yolo_dataset.validate_dataset(data_yaml)

    assert any("expected class id 0" in failure for failure in failures)


def test_validate_yolo_dataset_rejects_out_of_range_boxes(tmp_path: Path) -> None:
    data_yaml = write_dataset(tmp_path, bad_label="0 1.2 0.5 0.25 0.25")

    failures = validate_yolo_dataset.validate_dataset(data_yaml)

    assert any("must be normalized to [0, 1]" in failure for failure in failures)


def test_start_label_studio_qr_builds_local_command(tmp_path: Path) -> None:
    args = argparse.Namespace(project_name="qr-project", host="127.0.0.1", port="9090", data_dir=tmp_path / "label-studio")

    command = start_label_studio_qr.build_label_studio_command(args, base_command=["label-studio"])

    assert command[:2] == ["label-studio", "start"]
    assert str(tmp_path / "label-studio" / "qr-project") in command
    assert "--no-browser" in command
    assert "--internal-host" in command and "127.0.0.1" in command
    assert "--port" in command and "9090" in command
    assert "--data-dir" in command and str(tmp_path / "label-studio") in command


def test_start_label_studio_qr_builds_init_command(tmp_path: Path) -> None:
    args = argparse.Namespace(project_name="qr-project", host="127.0.0.1", port="9090", data_dir=tmp_path / "label-studio")

    command = start_label_studio_qr.build_label_studio_init_command(args, base_command=["label-studio"])

    assert command[:2] == ["label-studio", "init"]
    assert str(tmp_path / "label-studio" / "qr-project") in command
    assert "--quiet" in command
    assert "--label-config" in command
    assert str(start_label_studio_qr.LABEL_CONFIG) in command
    assert "--data-dir" in command and str(tmp_path / "label-studio") in command


def test_start_label_studio_qr_enables_local_file_serving(tmp_path: Path) -> None:
    env = start_label_studio_qr.build_label_studio_env(tmp_path)

    assert env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] == "true"
    assert env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] == str(tmp_path)
