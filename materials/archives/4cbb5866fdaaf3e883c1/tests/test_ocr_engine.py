from __future__ import annotations

import numpy as np
from PIL import Image

from container_ocr.ocr_engine import PaddleOCREngine, normalize_ocr_result


class FakePredictOCR:
    def predict(self, image: object) -> list[dict[str, object]]:
        return [
            {
                "rec_texts": ["MSCU1234566", "22G1"],
                "rec_scores": [0.98, 0.91],
                "dt_polys": [
                    [[0, 0], [100, 0], [100, 20], [0, 20]],
                    [[0, 30], [40, 30], [40, 50], [0, 50]],
                ],
            }
        ]


def test_normalizes_paddleocr_v3_result() -> None:
    texts = normalize_ocr_result(FakePredictOCR().predict("image.jpg"))
    assert [item.text for item in texts] == ["MSCU1234566", "22G1"]
    assert texts[0].confidence == 0.98
    assert texts[0].box is not None


def test_engine_infers_with_fake_backend() -> None:
    engine = PaddleOCREngine(ocr_factory=FakePredictOCR, require_gpu_check=False)
    result = engine.infer("image.jpg")
    assert [item.value for item in result.container_numbers] == ["MSCU1234566"]
    assert result.container_numbers[0].is_valid is True
    assert [item.code for item in result.size_type_codes] == ["22G1"]


def test_engine_converts_pil_image_to_numpy_payload() -> None:
    class CapturePredictOCR(FakePredictOCR):
        def __init__(self) -> None:
            self.last_payload: object | None = None

        def predict(self, image: object) -> list[dict[str, object]]:
            self.last_payload = image
            return super().predict(image)

    engine = PaddleOCREngine(ocr_factory=CapturePredictOCR, require_gpu_check=False)
    engine.infer(Image.new("RGB", (16, 8), "white"))

    assert isinstance(engine.ocr.last_payload, np.ndarray)
    assert engine.ocr.last_payload.shape == (8, 16, 3)


def test_normalizes_paddleocr_v2_result() -> None:
    raw = [[[[[0, 0], [10, 0], [10, 10], [0, 10]], ("MAEU1234567", 0.88)]]]
    texts = normalize_ocr_result(raw)
    assert texts[0].text == "MAEU1234567"
    assert texts[0].confidence == 0.88
