# pyright: reportMissingTypeStubs=false
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import invoice_app.app as app_module


class FakeYoloDetector:
    instances: ClassVar[list[tuple[Path, float, int, str]]] = []

    def __init__(self, weights_path: Path, conf: float, imgsz: int, device: str) -> None:
        self.weights_path = weights_path
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.instances.append((weights_path, conf, imgsz, device))


class FakeDecoder:
    pass


class FakeOcr:
    instances: ClassVar[list[str]] = []

    def __init__(self, device: str) -> None:
        self.device = device
        self.instances.append(device)


class FakeLlm:
    pass


def test_real_mode_wires_strict_required_stack(monkeypatch, tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    FakeYoloDetector.instances.clear()
    FakeOcr.instances.clear()
    monkeypatch.setattr(app_module, "YoloQrDetector", FakeYoloDetector)
    monkeypatch.setattr(app_module, "OpenCvQrDecoder", FakeDecoder)
    monkeypatch.setattr(app_module, "PaddleOcrEngine", FakeOcr)
    monkeypatch.setattr(app_module, "OpenAICompatibleLlmExtractor", FakeLlm)
    monkeypatch.setattr(app_module, "SampleQrDetector", lambda: (_ for _ in ()).throw(AssertionError("sample detector used in real mode")))

    app = app_module.create_app(
        {
            "TESTING": True,
            "MODE": "real",
            "DATABASE_PATH": tmp_path / "db.sqlite3",
            "UPLOAD_DIR": tmp_path / "uploads",
            "ARTIFACT_DIR": tmp_path / "artifacts",
            "YOLO_WEIGHTS": weights,
            "YOLO_CONF": 0.37,
            "YOLO_IMGSZ": 960,
            "YOLO_DEVICE": "cuda:0",
            "PADDLE_DEVICE": "gpu:0",
        }
    )

    pipeline = app_module._build_pipeline(app)

    assert isinstance(pipeline.detector, FakeYoloDetector)
    assert isinstance(pipeline.decoder, FakeDecoder)
    assert isinstance(pipeline.ocr, FakeOcr)
    assert isinstance(pipeline.llm, FakeLlm)
    assert FakeYoloDetector.instances == [(weights, 0.37, 960, "cuda:0")]
    assert FakeOcr.instances == ["gpu:0"]
