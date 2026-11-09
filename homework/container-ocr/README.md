# Container OCR Web Demo

Local GPU-only OCR inference for shipping-container photos. The app uses PaddleOCR 3.x for inference, Gradio for the web UI, and TRUDI as the preferred test image source. It does not train or fine-tune any model.

## Scope

- Recognize ISO 6346 container numbers such as `MSCU1234566`.
- Extract common size/type codes such as `20GP`, `40HQ`, `22G1`, and `45G1`.
- Run inference on NVIDIA GPU only. CPU fallback is treated as an error.
- Show annotated result images in a local web UI.

## Development Setup

Install `uv`, then create the lightweight development environment:

```powershell
uv python install 3.10
uv sync --extra dev
```

Run the default tests:

```powershell
uv run pytest
```

Default tests do not install PaddleOCR, download models, require TRUDI, or require an NVIDIA GPU.

## OCR GPU Setup

Install PaddleOCR only when you are ready to run real inference:

```powershell
uv sync --extra dev --extra ocr
```

Then install the official GPU build of PaddlePaddle that matches your machine. Current PaddlePaddle Windows GPU docs list `paddlepaddle-gpu==3.2.2` wheels for CUDA indexes such as `cu118`, `cu126`, and `cu129`; choose the index from the official install page for your driver/CUDA compatibility.

Example command, to be verified against the official PaddlePaddle page before use:

```powershell
uv pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

Do not install CPU `paddlepaddle`. The project calls Paddle GPU checks before constructing PaddleOCR and fails fast if CUDA is unavailable.

Verify the machine and Paddle runtime:

```powershell
nvidia-smi
uv run python scripts/verify_gpu.py
```

Official references:

- PaddleOCR 3.x install: https://www.paddleocr.ai/main/en/version3.x/installation.html
- PaddleOCR quick start: https://www.paddleocr.ai/main/en/version3.x/paddlex/quick_start.html
- PaddlePaddle install: https://www.paddlepaddle.org.cn/install/quick

## TRUDI Dataset

Preferred dataset: https://github.com/egulsoylu/trudi

TRUDI is a real transportation-unit dataset with ground and aerial perspectives. The visible repository README reports 733 images and 35034 annotated instances, with varying cameras, lighting, weather, and zoom levels. The repository did not show a clear LICENSE file or explicit license statement during inspection, so use it for coursework/demo purposes and cite the authors.

Download TRUDI from its official repository link, extract it locally, and select a small deterministic sample set:

```powershell
uv run python scripts/prepare_trudi_samples.py --source C:\path\to\TRUDI --limit 20 --seed 42 --out data\trudi_samples
```

The script copies image files only. It does not download the full dataset and does not commit images to the project.

## Run

Command-line inference:

```powershell
uv run container-ocr data\trudi_samples\sample.jpg --out outputs
```

Equivalent module form:

```powershell
uv run python -m container_ocr.cli data\trudi_samples\sample.jpg --out outputs
```

Web UI:

```powershell
uv run container-ocr-ui
```

Open the printed Gradio URL, usually `http://127.0.0.1:7860`.

The UI returns an annotated image, a parsed container-number table, a size/type table, status text, and structured JSON.

## Troubleshooting

If `uv run` fails on Windows with `UnicodeDecodeError` inside `site.py` and the project path contains Chinese characters, rewrite the editable install path file to an ASCII relative path:

```powershell
Set-Content -Path .venv\Lib\site-packages\_editable_impl_container_ocr.pth -Value "..\..\..\src" -Encoding ascii
```

If `scripts/verify_gpu.py` says PaddlePaddle is not installed, install the official `paddlepaddle-gpu` wheel. If it says Paddle is not compiled with CUDA, remove CPU Paddle and install the GPU wheel.

## Submission Screenshots

Recommended screenshots for the coursework:

- Personal avatar/account screenshot.
- Terminal showing `uv` environment and GPU verification.
- Web UI with uploaded TRUDI image.
- Annotated OCR result image.
- Parsed container number and size/type output.
