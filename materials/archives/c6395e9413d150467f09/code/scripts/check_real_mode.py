from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER_TOKENS = ("fake", "placeholder", "replace", "change-me", "your-key", "填")


def has_usable_llm_key(value: str | None) -> bool:
    if value is None:
        return False
    stripped = value.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    return not any(token in lowered for token in PLACEHOLDER_TOKENS)


def dependency_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def requires_cuda(device: str | None) -> bool:
    if device is None:
        return False
    normalized = device.strip().lower()
    return normalized not in {"", "cpu"}


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def paddle_cuda_available(device: str) -> bool:
    probe = (
        "import sys, paddle; "
        "device=sys.argv[1]; "
        "raise_system_exit = getattr(sys, 'exit'); "
        "raise_system_exit(2) if not paddle.device.is_compiled_with_cuda() else None; "
        "paddle.device.set_device(device); "
        "print(paddle.device.get_device())"
    )
    completed = subprocess.run([sys.executable, "-c", probe, device], capture_output=True, text=True, check=False, cwd=PROJECT_ROOT)
    return completed.returncode == 0


def check_real_mode() -> list[str]:
    load_dotenv(PROJECT_ROOT / ".env")
    failures: list[str] = []
    mode = os.getenv("INVOICE_APP_MODE", "sample")
    if mode != "real":
        failures.append("INVOICE_APP_MODE 不是 real")

    weights_path = resolve_project_path(os.getenv("INVOICE_APP_YOLO_WEIGHTS", "runtime/models/yolo/best.pt"))
    if not weights_path.exists():
        failures.append(f"YOLO 权重不存在：{weights_path}")

    for module_name, package_name in [
        ("ultralytics", "ultralytics"),
        ("cv2", "opencv-python"),
        ("paddleocr", "paddleocr"),
        ("paddle", "paddlepaddle-gpu"),
    ]:
        if not dependency_available(module_name):
            failures.append(f"缺少 {module_name}，请安装 {package_name}")

    if requires_cuda(os.getenv("INVOICE_APP_YOLO_DEVICE", "0")):
        try:
            import torch
        except ModuleNotFoundError:
            failures.append("INVOICE_APP_YOLO_DEVICE 指向 GPU，但缺少 torch")
        else:
            if not torch.cuda.is_available():
                failures.append("INVOICE_APP_YOLO_DEVICE 指向 GPU，但当前 PyTorch 不是 CUDA 可用版本")

    if requires_cuda(os.getenv("INVOICE_APP_PADDLE_DEVICE", "gpu:0")):
        if not paddle_cuda_available(os.getenv("INVOICE_APP_PADDLE_DEVICE", "gpu:0")):
            failures.append("INVOICE_APP_PADDLE_DEVICE 指向 GPU，但 PaddlePaddle GPU 子进程预检未通过")

    if not has_usable_llm_key(os.getenv("LLM_API_KEY")):
        failures.append("LLM_API_KEY 未填写或仍是占位假密钥")
    return failures


def main() -> int:
    failures = check_real_mode()
    if failures:
        print("真实模式预检失败：")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("真实模式预检通过：依赖、YOLO 权重和 NVIDIA NIM 密钥均已就绪。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
