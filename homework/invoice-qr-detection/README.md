# 发票二维码检测

保留发票二维码检测作业的合成数据脚本、YOLO 标签、标注规范、训练与推理脚本，以及历史验证记录。

```text
.
├── scripts/                 # 合成发票、训练、推理和数据/模型验证脚本
├── data/
│   ├── data.yaml            # YOLO 数据配置
│   ├── classes.txt          # 类别名称 qr_code
│   ├── labels/              # train、val、test 三个划分的 200 个标签文件
│   ├── working/             # 三组合成数据工作区，共 220 个标签文件及样本清单
│   └── *.md、*.csv          # 数据来源、使用限制和清单
├── evidence/logs/           # 转为 Markdown 的历史验证日志
├── runs/                    # 已有训练结果 CSV
├── samples/                 # 推理样本清单
├── annotation_guidelines.md # 二维码标注规范
└── requirements.txt         # Python 依赖清单
```

标签内容保持原样；原来的 `dataset/` 和 `data_working/` 分别整理为 `data/` 和 `data/working/`，脚本、清单和说明中的数据路径同步调整。历史日志中的路径保留当时的记录。

当前归档不含原始图片或模型权重；历史日志中的验证结果记录的是原项目运行情况。
