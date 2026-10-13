# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import ClassVar

import pytest
from PIL import Image

import invoice_app.adapters.yolo_detector as yolo_module
from invoice_app.adapters.yolo_detector import MissingDependencyError, YoloQrDetector


class FakeBoxes:
    xyxy = [[10, 10, 30, 30], [60, 40, 100, 80]]
    conf = [0.25, 0.92]

    def __len__(self) -> int:
        return len(self.conf)


class FakeResult:
    boxes = FakeBoxes()


class RecordingYolo:
    calls: ClassVar[list[tuple[str, object]]] = []

    def __init__(self, weights_path: str) -> None:
        self.calls.append(("init", weights_path))

    def predict(self, source: str, conf: float, imgsz: int, device: str, verbose: bool) -> list[FakeResult]:
        self.calls.append(("predict", (source, conf, imgsz, device, verbose)))
        return [FakeResult()]


def _install_fake_ultralytics(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("ultralytics")
    setattr(module, "YOLO", RecordingYolo)
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    RecordingYolo.calls.clear()


def _invoice_image(path: Path) -> None:
    Image.new("RGB", (160, 120), "white").save(path)


def test_yolo_detector_requires_ultralytics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "invoice.png"
    _invoice_image(source)

    def fake_import_module(name: str) -> ModuleType:
        if name == "ultralytics":
            raise ModuleNotFoundError(name)
        raise AssertionError(name)

    monkeypatch.setattr(yolo_module.importlib, "import_module", fake_import_module)

    with pytest.raises(MissingDependencyError, match="uv sync --extra real"):
        YoloQrDetector(tmp_path / "best.pt").detect_and_crop(source, tmp_path / "artifacts")


def test_yolo_detector_reports_missing_weights(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_ultralytics(monkeypatch)
    source = tmp_path / "invoice.png"
    _invoice_image(source)

    with pytest.raises(FileNotFoundError, match="INVOICE_APP_YOLO_WEIGHTS"):
        YoloQrDetector(tmp_path / "missing.pt").detect_and_crop(source, tmp_path / "artifacts")


def test_yolo_detector_uses_highest_confidence_box(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_ultralytics(monkeypatch)
    source = tmp_path / "invoice.png"
    weights = tmp_path / "best.pt"
    _invoice_image(source)
    weights.write_text("fake test weight", encoding="utf-8")

    result = YoloQrDetector(weights, conf=0.42, imgsz=1280, device="cuda:0").detect_and_crop(source, tmp_path / "artifacts")

    assert result.bbox_xyxy == (52, 32, 108, 88)
    assert result.confidence == 0.92
    assert result.crop_path.exists()
    assert result.annotated_path is not None and result.annotated_path.exists()
    assert RecordingYolo.calls == [("init", str(weights)), ("predict", (str(source), 0.42, 1280, "cuda:0", False))]


def test_yolo_detector_does_not_import_cv2_as_detector(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    imported: list[str] = []
    source = tmp_path / "invoice.png"
    weights = tmp_path / "best.pt"
    _invoice_image(source)
    weights.write_text("fake test weight", encoding="utf-8")

    def fake_import_module(name: str) -> ModuleType:
        imported.append(name)
        if name == "ultralytics":
            module = ModuleType("ultralytics")
            setattr(module, "YOLO", RecordingYolo)
            return module
        raise AssertionError(name)

    monkeypatch.setattr(yolo_module.importlib, "import_module", fake_import_module)
    YoloQrDetector(weights).detect_and_crop(source, tmp_path / "artifacts")

    assert imported == ["ultralytics"]
