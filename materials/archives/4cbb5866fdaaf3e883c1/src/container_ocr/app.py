from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any

from PIL import Image

from .annotation import annotate_image
from .gpu import GPUUnavailableError, check_gpu
from .ocr_engine import PaddleOCREngine
from .types import InferenceResult


@lru_cache(maxsize=1)
def get_engine() -> PaddleOCREngine:
    return PaddleOCREngine()


def _container_rows(result: InferenceResult) -> list[list[Any]]:
    return [
        [item.value, item.owner_code, item.equipment_category, item.serial_number, item.check_digit, item.is_valid, item.reason]
        for item in result.container_numbers
    ]


def _size_rows(result: InferenceResult) -> list[list[Any]]:
    return [[item.code, item.confidence, item.raw_text] for item in result.size_type_codes]


def _load_display_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def run_web_inference(
    image: str | Path | Image.Image | None,
    engine_factory: Any = get_engine,
) -> tuple[Image.Image | None, list[list[Any]], list[list[Any]], str, dict[str, Any]]:
    if image is None:
        return None, [], [], "Upload a container image first.", {}

    try:
        engine = engine_factory()
        display_image = _load_display_image(image)
        result = engine.infer(image)
        annotated = annotate_image(display_image, result.ocr_texts, result.container_numbers, result.size_type_codes)
    except GPUUnavailableError as exc:
        status = check_gpu()
        return None, [], [], f"GPU unavailable: {exc}", asdict(status)

    valid_count = sum(1 for item in result.container_numbers if item.is_valid)
    status_text = f"Done. OCR texts: {len(result.ocr_texts)}; valid container numbers: {valid_count}."
    payload = asdict(result)
    payload.pop("raw_result", None)
    return annotated, _container_rows(result), _size_rows(result), status_text, payload


def build_demo() -> Any:
    gr = import_module("gradio")

    gpu_status = check_gpu()
    with gr.Blocks(title="Container OCR") as demo:
        gr.Markdown("# Container OCR")
        gr.Markdown("GPU-only PaddleOCR inference for container number and size/type extraction.")
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Container image", type="filepath")
                run_button = gr.Button("Run OCR", variant="primary")
                status_output = gr.Textbox(label="Status", value=gpu_status.reason, interactive=False)
            with gr.Column(scale=1):
                annotated_output = gr.Image(label="Annotated result", type="pil")
        container_output = gr.Dataframe(
            headers=["value", "owner", "category", "serial", "digit", "valid", "reason"],
            label="Container numbers",
        )
        size_output = gr.Dataframe(headers=["code", "confidence", "source"], label="Size/type codes")
        raw_output = gr.JSON(label="Structured result")
        run_button.click(
            run_web_inference,
            inputs=image_input,
            outputs=[annotated_output, container_output, size_output, status_output, raw_output],
        )
    return demo


def main() -> None:
    build_demo().launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
