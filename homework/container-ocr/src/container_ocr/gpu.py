from __future__ import annotations

import json
from dataclasses import asdict
from importlib import import_module
from types import ModuleType
from typing import Any

from .types import GPUStatus


class GPUUnavailableError(RuntimeError):
    pass


def import_paddle() -> ModuleType:
    try:
        module = import_module("paddle")
    except ImportError as exc:
        raise GPUUnavailableError(
            "PaddlePaddle is not installed. Install the official paddlepaddle-gpu wheel "
            "that matches your CUDA version before running inference."
        ) from exc
    if not isinstance(module, ModuleType):
        raise GPUUnavailableError("Imported paddle module has an unexpected type.")
    return module


def _call_bool(obj: Any, name: str) -> bool | None:
    value = getattr(obj, name, None)
    if not callable(value):
        return None
    result = value()
    return bool(result)


def _device_count(paddle_module: ModuleType | Any) -> int | None:
    cuda = getattr(getattr(paddle_module, "device", None), "cuda", None)
    count_fn = getattr(cuda, "device_count", None)
    if callable(count_fn):
        raw_count: Any = count_fn()
        return int(raw_count)
    return None


def check_gpu(paddle_module: ModuleType | Any | None = None, device: str = "gpu:0") -> GPUStatus:
    try:
        paddle = paddle_module if paddle_module is not None else import_paddle()
    except GPUUnavailableError as exc:
        return GPUStatus(available=False, reason=str(exc), device=device)

    paddle_version = str(getattr(paddle, "__version__", "unknown"))
    paddle_device = getattr(paddle, "device", None)
    compiled = _call_bool(paddle_device, "is_compiled_with_cuda") if paddle_device is not None else None
    if compiled is not True:
        return GPUStatus(
            available=False,
            reason="Installed PaddlePaddle is not compiled with CUDA. Replace CPU Paddle with paddlepaddle-gpu.",
            device=device,
            paddle_version=paddle_version,
        )

    count = _device_count(paddle)
    if count is not None and count < 1:
        return GPUStatus(
            available=False,
            reason="PaddlePaddle CUDA build is installed, but no CUDA device is visible.",
            device=device,
            device_count=count,
            paddle_version=paddle_version,
        )

    return GPUStatus(
        available=True,
        reason="CUDA GPU is available to PaddlePaddle.",
        device=device,
        device_count=count,
        paddle_version=paddle_version,
    )


def require_gpu(paddle_module: ModuleType | Any | None = None, device: str = "gpu:0") -> GPUStatus:
    paddle = paddle_module if paddle_module is not None else import_paddle()
    status = check_gpu(paddle, device=device)
    if not status.available:
        raise GPUUnavailableError(status.reason)

    set_device = getattr(getattr(paddle, "device", None), "set_device", None)
    if not callable(set_device):
        set_device = getattr(paddle, "set_device", None)
    if callable(set_device):
        try:
            set_device(device)
        except Exception as exc:
            raise GPUUnavailableError(f"Failed to select Paddle device {device}: {exc}") from exc

    return status


def main() -> int:
    status = check_gpu()
    print(json.dumps(asdict(status), ensure_ascii=False, indent=2))
    if not status.available:
        raise SystemExit(1)
    require_gpu(device=status.device or "gpu:0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
