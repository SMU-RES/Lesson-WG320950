from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from .infer import run_segmentation


gr = import_module("gradio")
MODEL_CHOICES = ["yolo11n-seg.pt", "yolov8n-seg.pt"]


class _Launchable(Protocol):
    def launch(self, server_name: str, server_port: int, inbrowser: bool = ..., css: str = ...) -> object:
        ...


CSS = """
:root {
  --ink: #17201a;
  --leaf: #2f6f4e;
  --mint: #e9f5ec;
  --paper: #fbfaf2;
  --line: rgba(47, 111, 78, 0.18);
}
.gradio-container {
  background:
    radial-gradient(circle at 12% 15%, rgba(91, 140, 90, 0.20), transparent 28%),
    linear-gradient(135deg, #fbfaf2 0%, #eff7ef 52%, #dfeee4 100%);
  color: var(--ink);
}
#hero {
  border: 1px solid var(--line);
  border-radius: 26px;
  padding: 26px 30px;
  background: rgba(255, 255, 255, 0.72);
  box-shadow: 0 24px 70px rgba(29, 70, 49, 0.12);
}
#hero h1 {
  letter-spacing: -0.04em;
  margin-bottom: 8px;
}
.panel {
  border: 1px solid var(--line) !important;
  border-radius: 22px !important;
  background: rgba(255, 255, 255, 0.70) !important;
  box-shadow: 0 18px 48px rgba(29, 70, 49, 0.10) !important;
}
button.primary {
  background: linear-gradient(135deg, #2f6f4e, #75a35d) !important;
  border: none !important;
}
"""


def _format_summary(metadata: Mapping[str, object]) -> str:
    lines = [
        "### 推理结果",
        f"- 模型：`{metadata.get('model_name', 'unknown')}`",
        f"- 设备：`{metadata.get('device', 'cpu')}`",
        f"- 置信度阈值：`{metadata.get('confidence_threshold', 'n/a')}`",
        f"- 分割实例数量：**{metadata.get('num_instances', 0)}**",
        f"- 结果图片：`{metadata.get('output_image', '')}`",
    ]

    instances = metadata.get("instances", [])
    if isinstance(instances, list) and instances:
        lines.append("\n### 实例明细")
        for index, item in enumerate(instances, start=1):
            if not isinstance(item, Mapping):
                continue
            label = item.get("class_name") or "unknown"
            score = item.get("confidence")
            score_text = f"{score:.3f}" if isinstance(score, float) else "n/a"
            mask_points = item.get("mask_points") or "n/a"
            lines.append(f"{index}. 类别：`{label}`，置信度：`{score_text}`，掩膜像素：`{mask_points}`")
    else:
        lines.append("\n没有检测到可分割实例，可以降低置信度后再试。")

    return "\n".join(lines)


def segment_image(image_path: str | None, conf: float, model_name: str) -> tuple[str | None, str]:
    if not image_path:
        return None, "请先上传、拍摄或粘贴一张图片。"

    try:
        output_path, metadata = run_segmentation(
            image_path=Path(image_path),
            output_dir=Path("outputs"),
            model_name=model_name,
            conf=float(conf),
            device="cpu",
        )
    except Exception as exc:
        return None, f"推理失败：{exc}"

    return output_path, _format_summary(metadata)


def build_demo() -> _Launchable:
    with gr.Blocks(title="本地图像分割演示") as demo:
        gr.Markdown(
            """
<div id="hero">

# 本地 YOLO 图像分割演示

上传、拍摄或粘贴一张图片，程序会在本机使用预训练分割模型生成结果图。默认使用 CPU，不需要训练，不需要 Docker。

</div>
"""
        )

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["panel"]):
                image_input = gr.Image(
                    label="输入图片",
                    type="filepath",
                    sources=["upload", "webcam", "clipboard"],
                    height=360,
                )
                model = gr.Dropdown(
                    label="分割模型",
                    choices=MODEL_CHOICES,
                    value=MODEL_CHOICES[0],
                    info="首次运行会自动下载对应的 .pt 权重文件。",
                )
                conf = gr.Slider(
                    label="置信度阈值",
                    minimum=0.05,
                    maximum=0.95,
                    value=0.25,
                    step=0.05,
                    info="数值越低，保留的目标越多；数值越高，结果越严格。",
                )
                run_button = gr.Button("开始分割", variant="primary")

            with gr.Column(scale=1, elem_classes=["panel"]):
                output_image = gr.Image(label="分割结果图", type="filepath", height=360)
                summary = gr.Markdown("等待上传图片后开始推理。")

        run_button.click(
            fn=segment_image,
            inputs=[image_input, conf, model],
            outputs=[output_image, summary],
            api_name="segment",
        )

    return cast(_Launchable, demo)


def main() -> None:
    demo = build_demo()
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, css=CSS)

__all__ = ["build_demo", "main", "segment_image"]

if __name__ == "__main__":
    main()
