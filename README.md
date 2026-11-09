# 人工智能与工程实践

课程号：`WG320950`。

## 资料目录

```text
.
├── homework/                              # 各次作业的源码、要求及已有结果
│   ├── handwritten-digit-recognition/     # 用 SVM、全连接网络和 CNN 识别 MNIST 手写数字，提供 Tkinter 手写输入界面
│   ├── financial-time-series-rnn/         # 使用循环神经网络处理金融时间序列，附截图生成脚本和作业要求
│   ├── invoice-reimbursement-cli/         # 解析发票文本并通过命令行生成报销结果，附示例发票文本
│   ├── invoice-qr-detection/              # 发票二维码的合成数据、YOLO 标注、训练与验证脚本及历史结果
│   ├── container-ocr/                     # 使用 PaddleOCR 识别集装箱箱号及尺寸类型代码，提供命令行和 Gradio 界面
│   ├── instance-segmentation/             # 使用预训练 YOLO 模型进行实例分割推理，提供命令行和 Gradio 界面
│   ├── plate-digit-cnn/                   # 训练车牌数字分类 CNN，附演示数据生成与公开数据整理脚本
│   └── synthetic-digit-cnn/               # 生成车牌风格数字图像并训练 CNN，附单图推理脚本和混淆矩阵
├── content/
│   └── invoice-reimbursement/             # 大作业：发票二维码/OCR 识别、字段校验与重复报销检测系统的已有源码、文档和测试
└── reports/
    └── 手写数字识别系统设计与实践大报告.pdf   # 对比 MNIST 的 SVM、全连接网络和 CNN 实现，含流程图与运行截图
```

## 使用与归档说明

- 各作业和大作业分别保留自己的依赖清单及已有说明，请在相应目录查看 `README.md`、`pyproject.toml` 或 `requirements.txt`。
- 手写数字识别作业保留 `uv` 工程及锁文件；报告对应其中的三个识别程序。
- 发票二维码作业的正式标签位于 `homework/invoice-qr-detection/data/labels/`，合成数据工作区标签位于 `data/working/*/labels/`；这些 `.txt` 是数据，不是临时说明文件。
- 示例发票文本、类别文件、依赖清单和 Python 包元数据中的 `.txt` 保留原格式；作业要求及历史验证日志已整理为 Markdown。历史日志保留当时记录的路径和数值，生成脚本仍按原设计输出日志。
- 本库保留的是已有归档文件，部分原项目说明引用的模块、图片、数据或运行产物并未收录；目录中的历史结果不代表重新运行验证。仓库不含模型权重或视频。
- 大报告直接恢复自历史中已有的 PDF，未保留 Word 原件。
