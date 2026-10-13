from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image


@dataclass
class DetectionSummary:
    class_id: int | None
    class_name: str | None
    confidence: float | None
    box_xyxy: list[float] | None
    mask_points: int | None


class _MasksLike(Protocol):
    data: object


class _YOLOModel(Protocol):
    def predict(self, source: str, conf: float = ..., device: str = ..., verbose: bool = ...) -> Sequence[_ResultLike]:
        ...


class _ResultLike(Protocol):
    names: Mapping[int, str]
    boxes: object
    masks: _MasksLike | None

    def plot(self) -> object:
        ...


_MODEL_CACHE: dict[str, _YOLOModel] = {}


def get_model(model_name: str) -> _YOLOModel:
    if model_name not in _MODEL_CACHE:
        try:
            yolo_cls = cast(Callable[[str], _YOLOModel], getattr(import_module("ultralytics"), "YOLO"))
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed; run `uv sync --extra dev`") from exc
        _MODEL_CACHE[model_name] = yolo_cls(model_name)
    return _MODEL_CACHE[model_name]


def _validate_conf(conf: float) -> None:
    if not 0 <= conf <= 1:
        raise ValueError("conf must be between 0 and 1")


def _validate_image_path(image_path: str | Path) -> Path:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"image path is not a file: {path}")
    return path


def _safe_model_tag(model_name: str) -> str:
    return Path(model_name).stem.replace(" ", "_")


def _to_pil_image(annotated: object) -> Image.Image:
    if isinstance(annotated, Image.Image):
        return annotated
    array = np.asarray(annotated)
    if array.ndim == 3 and array.shape[2] >= 3:
        array = array[:, :, :3][:, :, ::-1]
    return Image.fromarray(array.astype(np.uint8))


def _to_numpy(value: object) -> NDArray[np.float64]:
    current = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()
    numpy = getattr(current, "numpy", None)
    if callable(numpy):
        current = numpy()
    return np.asarray(current)


def _class_name(names: Mapping[int, str] | Sequence[str], class_id: int | None) -> str | None:
    if class_id is None:
        return None
    if isinstance(names, Mapping):
        return names.get(class_id, str(class_id))
    if 0 <= class_id < len(names):
        return names[class_id]
    return str(class_id)


def _extract_summaries(result: _ResultLike) -> list[DetectionSummary]:
    summaries: list[DetectionSummary] = []
    boxes = result.boxes
    cls_values = _to_numpy(getattr(boxes, "cls", [])).reshape(-1)
    conf_values = _to_numpy(getattr(boxes, "conf", [])).reshape(-1)
    xyxy_raw = _to_numpy(getattr(boxes, "xyxy", []))
    xyxy_values = xyxy_raw.reshape(-1, 4) if xyxy_raw.size else np.empty((0, 4), dtype=np.float64)

    mask_counts: list[int | None] = []
    if result.masks is not None:
        mask_data = _to_numpy(result.masks.data)
        if mask_data.ndim >= 3:
            mask_counts = [int(np.count_nonzero(mask_data[idx])) for idx in range(mask_data.shape[0])]

    instance_count = max(len(cls_values), len(conf_values), len(xyxy_values), len(mask_counts))
    for idx in range(instance_count):
        cls_id = int(cls_values[idx]) if idx < len(cls_values) else None
        conf = float(conf_values[idx]) if idx < len(conf_values) else None
        xyxy = [float(v) for v in xyxy_values[idx].tolist()] if idx < len(xyxy_values) else None
        mask_points = mask_counts[idx] if idx < len(mask_counts) else None

        summaries.append(
            DetectionSummary(
                class_id=cls_id,
                class_name=_class_name(result.names, cls_id),
                confidence=conf,
                box_xyxy=xyxy,
                mask_points=mask_points,
            )
        )

    return summaries


def run_segmentation(
    image_path: str | Path,
    output_dir: str | Path = "outputs",
    model_name: str = "yolo11n-seg.pt",
    conf: float = 0.25,
    device: str = "cpu",
) -> tuple[str, Mapping[str, object]]:
    _validate_conf(conf)
    input_path = _validate_image_path(image_path)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    model = get_model(model_name)
    results = model.predict(source=str(input_path), conf=conf, device=device, verbose=False)
    if not results:
        raise RuntimeError("no inference results returned")

    result = results[0]
    annotated = result.plot()
    if annotated is None:
        raise RuntimeError("failed to render annotated result")

    image = _to_pil_image(annotated)

    output_name = f"{input_path.stem}_{_safe_model_tag(model_name)}_segmented.png"
    output_path = output_root / output_name
    image.save(output_path)

    summaries = _extract_summaries(result)
    metadata: dict[str, object] = {
        "input_image": str(input_path),
        "output_image": str(output_path),
        "model_name": model_name,
        "confidence_threshold": conf,
        "device": device,
        "num_instances": len(summaries),
        "instances": [summary.__dict__ for summary in summaries],
    }
    return str(output_path), metadata
