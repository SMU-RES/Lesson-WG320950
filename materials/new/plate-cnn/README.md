# plate-cnn

一个使用 **uv + PyTorch** 搭建的简单 CNN 车牌数字分类作业项目。

## 项目目标

- 对车牌中的 **单个数字字符** 做 `0~9` 十分类。
- 使用简单 CNN 完成训练与测试。
- 导出训练好的模型。
- 编写单张图片推理程序。
- 输出训练曲线图，方便截图和课程提交。

> 当前项目先把完整流程搭好；真实车牌数字数据后续可直接替换到 `data/` 目录下。

## 环境准备

```bash
uv init plate-cnn
uv venv
uv add torch torchvision matplotlib pillow
```

在本项目中运行脚本时，推荐使用 `uv run`。

## 数据目录格式

真实数据应整理为如下格式：

```text
data/
  your_dataset/
    train/
      0/ 1/ 2/ ... 9/
    val/
      0/ 1/ 2/ ... 9/
    test/
      0/ 1/ 2/ ... 9/
```

每个类别文件夹下放对应数字图片。

## 演示数据

项目提供了一个演示数据生成脚本，可先快速验证训练/导出/推理流程：

```bash
uv run python make_demo_dataset.py
```

这会生成：

```text
data/demo_digits/
  train/ val/ test/
```

## 训练

默认使用演示数据集训练：

```bash
uv run python train.py
```

如果你想直接一条命令完成“下载公开数据 + 整理数据集 + 训练 + 输出 loss 曲线图片”，运行：

```bash
uv run python run_direct_train.py
```

默认会：

1. 自动下载公开字符数据集 `data.zip`
2. 提取其中的数字 `0~9`
3. 整理成 `data/public_digits/train|val|test/0..9`
4. 训练模型并输出曲线图片

如果你只想继续使用本地演示数据，可以运行：

```bash
uv run python run_direct_train.py --dataset demo --epochs 6
```

运行完成后，loss 曲线会保存到：

```text
outputs/results/loss_curve.png
```

公开数据整理脚本也可以单独运行：

```bash
uv run python prepare_public_dataset.py
```

如果你后续换成自己的车牌数字数据，可指定路径：

```bash
uv run python train.py --data-root data/your_dataset --epochs 20 --batch-size 32
```

训练输出：

- `outputs/best_model.pth`：训练好的模型
- `outputs/results/loss_curve.png`：损失曲线
- `outputs/results/accuracy_curve.png`：准确率曲线
- `outputs/results/training_summary.json`：训练摘要

## 推理

对一张新的单字符数字图片做预测：

```bash
uv run python infer.py path/to/image.png
```

输出内容：

- 图片路径
- 预测类别
- 置信度

## 文件说明

```text
plate_cnn/
  __init__.py
  data.py        # 数据预处理与数据集加载
  model.py       # 简单 CNN 模型
  utils.py       # 结果保存、作图、checkpoint
make_demo_dataset.py  # 生成演示数据
train.py              # 训练脚本
infer.py              # 推理脚本
outputs/              # 训练结果输出目录
```

## 截图建议

提交时建议至少准备以下截图，并保证截图中包含数字平台个人图片：

1. **代码截图**：`train.py`、`model.py` 或 `infer.py`
2. **训练结果截图**：终端训练日志 + `accuracy_curve.png` / `loss_curve.png`
3. **推理结果截图**：运行 `infer.py` 后的预测输出

## 后续接入真实车牌数据的建议

1. 从公开车牌数据中裁剪出单个数字字符。
2. 统一缩放到接近 `32x32` 尺寸。
3. 按 `train / val / test` 划分。
4. 放入 `0~9` 十个类别目录后直接训练。
