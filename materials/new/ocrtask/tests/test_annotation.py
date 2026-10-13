from PIL import Image

from container_ocr.annotation import annotate_image
from container_ocr.parsing import validate_container_number
from container_ocr.types import OCRText, SizeTypeCode


def test_annotation_keeps_image_size() -> None:
    image = Image.new("RGB", (160, 80), "white")
    texts = [OCRText("MSCU1234566", 0.99, [(10, 10), (120, 10), (120, 30), (10, 30)])]
    containers = [validate_container_number("MSCU1234566", text_index=0)]
    sizes = [SizeTypeCode("22G1", "22G1", text_index=0)]
    annotated = annotate_image(image, texts, containers, sizes)
    assert annotated.size == image.size
    assert annotated.mode == "RGB"
