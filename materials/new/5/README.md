# 本地图像分割模型部署

这个项目使用 Ultralytics YOLO 预训练分割模型在本机做图片实例分割。项目不包含训练流程，默认使用 CPU，可以通过命令行或 Gradio 网页界面推理。

## 1. 使用 uv 创建环境

在当前项目目录打开 PowerShell：

```powershell
uv venv
uv sync --extra dev
```

之后推荐直接用 `uv run` 运行项目命令，不需要手动激活环境。也可以手动激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 阻止激活虚拟环境，先运行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. 启动交互界面

```powershell
uv run python -m segmentation_demo.app
```

浏览器打开：

```text
http://127.0.0.1:7860
```

界面支持三种输入方式：

- 上传本地图片
- 调用摄像头拍照
- 从剪贴板粘贴图片

第一次推理会自动下载模型权重，例如 `yolo11n-seg.pt`，所以首次运行可能较慢。

## 3. 命令行推理

把测试图片放到 `samples` 目录，例如 `samples\test.jpg`，然后运行：

```powershell
uv run python -m segmentation_demo.cli --image .\samples\test.jpg
```

可以调整置信度和模型：

```powershell
uv run python -m segmentation_demo.cli --image .\samples\test.jpg --conf 0.35 --model yolov8n-seg.pt
```

结果图片会保存到 `outputs` 目录。

## 4. 项目结构

```text
.
├─ segmentation_demo\infer.py       # 分割推理核心
├─ segmentation_demo\cli.py         # 命令行入口
├─ segmentation_demo\app.py         # Gradio 交互界面
├─ samples\                         # 放输入图片
├─ outputs\                         # 保存推理结果
├─ tests\                           # 单元测试
├─ pyproject.toml                    # uv 使用的项目依赖配置
├─ uv.lock                           # uv 锁定的依赖版本
└─ requirements.txt                  # 备用依赖列表
```

## 5. 验证

```powershell
uv run pytest
uv run python -m segmentation_demo.cli --help
```

如需真实推理验证，准备一张图片后运行命令行推理或打开网页界面上传图片。

## 6. 常见问题

- `ultralytics is not installed`：确认已经运行 `uv sync --extra dev`。
- `No module named segmentation_demo`：在项目根目录运行 `uv sync --extra dev` 后再使用 `uv run`。
- 模型下载失败：检查网络，或把已下载的 `.pt` 权重文件放在项目目录。
- `yolo11n-seg.pt` 无法使用：在界面或命令行切换为 `yolov8n-seg.pt`。
- 推理很慢：CPU 推理正常会比 GPU 慢，使用小模型 `yolo11n-seg.pt` 即可。
