# 车牌数字分类 CNN 随堂作业

本项目使用 **PyTorch** 构造了一个简单 CNN，对数字 `0-9` 进行分类，可作为“车牌数字分类”的基础版本。为了保证课堂环境可直接运行，训练数据由程序自动生成：用 Pillow 绘制车牌风格数字，并加入随机平移、亮度和噪声增强，不需要联网下载数据集。

## 文件说明

- `train_cnn.py`：训练 CNN，导出模型，并生成训练曲线和预测结果图。
- `infer_digit.py`：加载训练好的模型，对新的数字图片进行推理。
- `pyproject.toml`：`uv` 项目依赖配置。
- `outputs/digit_cnn.pt`：训练后导出的 PyTorch 模型参数。
- `outputs/training_curves.png`：训练/验证准确率和损失曲线。
- `outputs/prediction_examples.png`：测试集样例预测结果图。
- `outputs/sample_digit.png`：用于推理演示的新图片。

## 运行方法（必须使用 uv）

```bash
uv sync
uv run python train_cnn.py --epochs 8
uv run python infer_digit.py --image outputs/sample_digit.png
```

运行完成后，可以截图终端输出、代码窗口，以及 `outputs/training_curves.png` / `outputs/prediction_examples.png`，满足“代码和结果图”的提交要求。

## 模型结构

输入为 `1 × 32 × 32` 灰度数字图像，网络结构如下：

1. `Conv2d(1, 16, kernel_size=3, padding=1)` + ReLU
2. `MaxPool2d(2)`
3. `Conv2d(16, 32, kernel_size=3, padding=1)` + ReLU
4. `MaxPool2d(2)`
5. `Linear(32 * 8 * 8, 64)` + ReLU
6. `Linear(64, 10)` 输出 10 类数字概率

## 说明

如果之后换成真实车牌数字数据，只需要把每个数字裁剪成单字符图片，并在推理前转成灰度、缩放到 `32×32` 或调整网络输入尺寸即可。
