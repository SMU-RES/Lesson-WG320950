from __future__ import annotations

import pytest

from container_ocr.gpu import GPUUnavailableError, check_gpu, require_gpu


class FakeCuda:
    def __init__(self, count: int) -> None:
        self.count = count

    def device_count(self) -> int:
        return self.count


class FakeDevice:
    def __init__(self, compiled: bool, count: int) -> None:
        self.cuda = FakeCuda(count)
        self.selected: str | None = None
        self.compiled = compiled

    def is_compiled_with_cuda(self) -> bool:
        return self.compiled

    def set_device(self, device: str) -> None:
        self.selected = device


class FakePaddle:
    __version__ = "test"

    def __init__(self, compiled: bool = True, count: int = 1) -> None:
        self.device = FakeDevice(compiled, count)


def test_check_gpu_success() -> None:
    status = check_gpu(FakePaddle())
    assert status.available is True
    assert status.device_count == 1


def test_check_gpu_rejects_cpu_build() -> None:
    status = check_gpu(FakePaddle(compiled=False))
    assert status.available is False
    assert "not compiled with CUDA" in status.reason


def test_require_gpu_selects_device() -> None:
    paddle = FakePaddle()
    require_gpu(paddle, device="gpu:0")
    assert paddle.device.selected == "gpu:0"


def test_require_gpu_raises_for_no_device() -> None:
    with pytest.raises(GPUUnavailableError):
        require_gpu(FakePaddle(count=0))
