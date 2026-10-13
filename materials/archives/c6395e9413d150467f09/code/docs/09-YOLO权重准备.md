# YOLO 权重准备说明

## 1. 严格真实模式要求

真实模式必须使用 Ultralytics YOLO 定位发票二维码区域。OpenCV 只能解析 YOLO 生成的二维码裁剪图，不能直接解析整张发票图片来替代 YOLO。PaddleOCR 通过隔离子进程使用本地 GPU 做 OCR，NVIDIA NIM 负责远程字段抽取。

默认权重路径为：

```text
runtime/models/yolo/best.pt
```

也可以在 `.env` 中设置：

```dotenv
INVOICE_APP_YOLO_WEIGHTS=你的/best.pt/路径
```

当前交付在 `code/runtime/models/yolo/best.pt` 中包含默认权重。其它 `.pt`、`.onnx`、`.env`、真实票据、数据库、训练 runs 和运行产物都不会进入提交包。

## 2. 预检命令

安装真实模式依赖后运行：

```bash
uv sync --extra real
uv pip install --index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/ paddlepaddle-gpu==3.3.1
python scripts/check_real_mode.py
```

预检会检查 `INVOICE_APP_MODE`、YOLO 权重文件、`ultralytics`、`cv2`、`paddleocr`、`paddle`、按 `INVOICE_APP_YOLO_DEVICE` 请求的 PyTorch CUDA、按 `INVOICE_APP_PADDLE_DEVICE` 请求的 Paddle CUDA 和 `LLM_API_KEY`。脚本不会打印真实密钥。

## 3. 训练入口

准备好 YOLO 格式数据集后，可以运行：

```bash
python scripts/train_yolo_qr.py --data path/to/data.yaml --model yolov8n.pt --epochs 30 --imgsz 960 --device 0
```

训练结束后脚本会输出 `best.pt` 路径。将该路径写入 `INVOICE_APP_YOLO_WEIGHTS`，或复制到 `runtime/models/yolo/best.pt`。

## 4. 数据来源建议

本机增强训练数据集位于忽略目录 `runtime/datasets/processed/qr_yolo_v2/`，将以下来源统一为单类 `qr`：

- `kolabit/qr-codes`：YOLO 格式 QR 数据，仓库许可证 Apache-2.0。
- `mapo80/barcodes`：Hugging Face 数据集，许可证 CC BY-SA 4.0；仅取 `QRCode` 类并从 COCO 标注转换为 YOLO。

还可用于 QR 检测训练或评估的数据来源包括：

- `simplew2011/barcode_qrcode_kuihuama`
- `BenSouchet/barcode-datasets`
- QR-DN1.0 或 SBD 合成 QR 数据集
- 许可证明确且可导出 YOLO 格式的 Roboflow QR 数据集

使用任何公开数据集或公开权重前，都应记录来源、许可证、类别定义和导出格式。不要把未经脱敏的个人真实票据放入仓库。

## 5. 当前交付边界

提交包分为 `code/`、`演示/`、`报告pdf版本/` 三部分。`code/` 包含严格真实模式代码、预检脚本、训练入口和已训练的默认 YOLO QR 权重 `runtime/models/yolo/best.pt`；`演示/` 包含截图；`报告pdf版本/` 包含课程报告 PDF。公开数据集文件和训练 runs 不进入交付包；新机器也可以自行训练或用 `INVOICE_APP_YOLO_WEIGHTS` 指向其它权重。在没有真实依赖、YOLO 权重或密钥的机器上，真实模式应明确失败，不能静默切回 sample，也不能用固定样例字段冒充真实识别。
