from PIL import Image

from container_ocr.app import run_web_inference
from container_ocr.parsing import validate_container_number
from container_ocr.types import InferenceResult, OCRText, SizeTypeCode


class FakeEngine:
    def infer(self, image: object) -> InferenceResult:
        return InferenceResult(
            image_path=None,
            ocr_texts=[OCRText("MSCU1234566", 0.99, [(5, 5), (120, 5), (120, 25), (5, 25)])],
            container_numbers=[validate_container_number("MSCU1234566", text_index=0)],
            size_type_codes=[SizeTypeCode("22G1", "22G1", 0.9, 0)],
        )


def test_web_callback_with_fake_engine() -> None:
    image = Image.new("RGB", (160, 80), "white")
    annotated, containers, sizes, status, raw = run_web_inference(image, engine_factory=FakeEngine)
    assert annotated is not None
    assert containers[0][0] == "MSCU1234566"
    assert sizes[0][0] == "22G1"
    assert status.startswith("Done")
    assert raw["container_numbers"][0]["is_valid"] is True


def test_web_callback_accepts_file_path(tmp_path) -> None:
    image_path = tmp_path / "container.jpg"
    Image.new("RGB", (160, 80), "white").save(image_path)

    seen: dict[str, object] = {}

    class RecordingEngine(FakeEngine):
        def infer(self, image: object) -> InferenceResult:
            seen["image"] = image
            return super().infer(image)

    annotated, containers, sizes, status, raw = run_web_inference(str(image_path), engine_factory=RecordingEngine)

    assert seen["image"] == str(image_path)
    assert annotated is not None
    assert containers[0][0] == "MSCU1234566"
    assert sizes[0][0] == "22G1"
    assert status.startswith("Done")
    assert raw["container_numbers"][0]["is_valid"] is True
