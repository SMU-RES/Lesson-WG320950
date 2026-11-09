from __future__ import annotations

import math
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

class MonkeyPatchLike(Protocol):
    def setattr(self, target: object, name: str, value: object) -> None: ...


class InferModule(Protocol):
    def run_segmentation(
        self,
        image_path: str | Path,
        output_dir: str | Path = ...,
        model_name: str = ...,
        conf: float = ...,
        device: str = ...,
    ) -> tuple[str, dict[str, object]]: ...


infer = cast(InferModule, cast(object, import_module("segmentation_demo.infer")))


class DummyBoxes:
    def __init__(self) -> None:
        self.cls: NDArray[np.int64] = np.array([0])
        self.conf: NDArray[np.float64] = np.array([0.88])
        self.xyxy: NDArray[np.float64] = np.array([[10, 20, 30, 40]], dtype=float)


class DummyMasks:
    def __init__(self, masks: list[NDArray[np.uint8]]):
        self.data: list[NDArray[np.uint8]] = masks


class DummyResult:
    def __init__(self):
        self.names: dict[int, str] = {0: "person"}
        self.boxes: DummyBoxes = DummyBoxes()
        self.masks: DummyMasks = DummyMasks([np.array([[0, 1], [1, 1]], dtype=np.uint8)])

    def plot(self) -> NDArray[np.uint8]:
        return np.zeros((8, 8, 3), dtype=np.uint8)


class DummyModel:
    def predict(self, source: str, conf: float, device: str, verbose: bool) -> list[DummyResult]:
        assert source
        assert math.isclose(conf, 0.25)
        assert device == "cpu"
        assert verbose is False
        return [DummyResult()]


def test_run_segmentation_saves_output_and_metadata(monkeypatch: MonkeyPatchLike, tmp_path: Path) -> None:
    image_path = tmp_path / "input.jpg"
    Image.new("RGB", (16, 16), color="white").save(image_path)

    def make_model(_model_name: str) -> DummyModel:
        return DummyModel()

    monkeypatch.setattr(infer, "get_model", make_model)

    output_path, metadata = infer.run_segmentation(
        image_path=image_path,
        output_dir=tmp_path / "outputs",
        model_name="yolo11n-seg.pt",
        conf=0.25,
        device="cpu",
    )

    assert Path(output_path).exists()
    meta = cast(dict[str, object], metadata)
    instances = cast(list[dict[str, object]], meta["instances"])
    assert meta["model_name"] == "yolo11n-seg.pt"
    assert meta["num_instances"] == 1
    assert instances[0]["class_name"] == "person"
    assert math.isclose(float(cast(float, instances[0]["confidence"])), 0.88, rel_tol=1e-6)


def test_run_segmentation_rejects_missing_image() -> None:
    try:
        _ = infer.run_segmentation("missing.jpg")
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass


def test_run_segmentation_rejects_confidence_out_of_range(tmp_path: Path) -> None:
    image_path = tmp_path / "input.jpg"
    Image.new("RGB", (16, 16), color="white").save(image_path)

    try:
        _ = infer.run_segmentation(image_path, conf=1.5)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
