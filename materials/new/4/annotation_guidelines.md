# 发票二维码 YOLO 标注规范 / Invoice QR YOLO Annotation Guidelines

本规范用于本项目的发票/票据二维码目标检测数据标注。项目目标只是在图像中定位二维码位置，不做二维码内容解码、OCR、发票字段抽取或票据结构理解。

## 1. 类别配置 / Class policy

- 唯一类别：`qr_code`
- 唯一 class id：`0`
- `dataset/classes.txt` 必须只有一行非空内容：`qr_code`
- 禁止新增其他类别；不要把条形码、Logo、印章、表格、发票字段、二维码解码文本或 OCR 区域当作二维码类别。

## 2. YOLO 标签格式 / YOLO label format

每个有二维码目标的图片对应一个同名 `.txt` 标签文件。每个二维码目标占一行，格式为：

```text
class x_center y_center width height
```

字段含义：

- `class`：类别 id，本项目只能是 `0`。
- `x_center`、`y_center`：边界框中心点相对整张图像宽高的归一化坐标。
- `width`、`height`：边界框宽高相对整张图像宽高的归一化尺寸。
- 所有坐标和尺寸必须归一化到 `[0,1]`，且 `width > 0`、`height > 0`。

示例：

```text
0 0.732500 0.246000 0.115000 0.115000
```

## 3. 边界框策略 / Tight bounding-box policy

- 标注框必须紧贴二维码可见外接正方形/矩形区域，只包住二维码图案的外边界。
- 不要包含发票字段、标题、金额、表格线、二维码下方/旁边的解码文本、URL、说明文字或空白背景。
- 对于有白边 quiet zone 的二维码：如果白边是二维码视觉结构的一部分且紧邻二维码，可适度包含；不要额外扩大到票据背景。
- 目标框应覆盖可见二维码外轮廓；不要为了“好看”画松框，也不要截断可见二维码。

## 4. 边界情况 / Edge cases

| 情况 | 标注规则 |
| --- | --- |
| multiple QR codes / 多个二维码 | 每个二维码各标一行 `0 x_center y_center width height`，不要合并成一个大框。 |
| rotated QR / 旋转二维码 | 使用水平 YOLO bbox 包住旋转后二维码的可见外接矩形；类别仍为 `0`。 |
| partial QR / 局部二维码 | 如果二维码外形和定位点仍能明确识别，标注可见部分的紧框；如果只剩极小碎片且不可判断为二维码，则不标。 |
| occluded QR / 遮挡二维码 | 可见面积足够、能判断为二维码时标注可见二维码外轮廓；严重遮挡导致不可辨认时不标。 |
| blurry QR / 模糊二维码 | 仍能肉眼判断为二维码时标注；完全糊成纹理或无法确认时不标。 |
| tiny QR / 很小二维码 | 能清楚看出二维码结构时标注；低于人工可稳定识别尺寸、只像噪声点时不标。 |
| low-resolution scans / 低分辨率扫描 | 按可见二维码外轮廓紧框标注；如果压缩/扫描导致不可辨认，则按 unreadable QR 处理。 |
| unreadable QR codes / 无法读取或不可辨认二维码 | 只要视觉上能确认是二维码，即使内容不可解码也标注；视觉上不能确认时不标。 |
| no-QR images / 无二维码图片 | 作为 negative samples；可以没有同名 label 文件，或存在空 label 文件，后续验证脚本必须一致处理。 |
| barcodes/logos/stamps/decorative squares / 条形码、Logo、印章、装饰方块 | 不标注。它们是混淆负样本，不属于 `qr_code`。 |

## 5. 负样本与混淆样本 / Negative samples

- no-QR 发票、收据、账单、扫描件应保留为负样本，以降低模型对票据纹理的误检。
- 条形码、Logo、印章、表格密集线、装饰性黑白方块、发票章、收款码说明文字等都不得标为二维码。
- 负样本图片可以没有 `.txt` 标签文件，也可以有空 `.txt` 标签文件；本项目后续校验脚本将把两者都视为合法 no-QR 负样本。
- 不要为了让每张图片都有标签而给非二维码目标画框。

## 6. LabelImg 工作流 / LabelImg workflow

推荐优先使用 LabelImg 进行本地标注：

1. 使用项目虚拟环境启动 LabelImg：`.\.venv\Scripts\python.exe -m labelImg` 或 `.\.venv\Scripts\labelImg.exe`（实际命令以本机安装结果为准）。
2. 点击 **Open Dir**，选择待标注图片目录，例如 `dataset/images/train` 或待整理的工作目录。
3. 点击 **Change Save Dir**，选择对应标签目录，例如 `dataset/labels/train`。
4. 在左侧/菜单中选择 **YOLO** 保存格式。
5. 确认类别文件只包含 `qr_code`；如果工具提示类别列表，选择或输入 `qr_code`，其 class id 必须对应 `0`。
6. 对每个可见二维码绘制紧贴外轮廓的 bbox，保存后生成同名 `.txt` 标签文件。
7. 对 no-QR negative samples，不画框；允许不保存标签文件，或保存空 `.txt` 文件，后续由验证脚本统一处理。

注意：LabelImg 只负责标注，不负责训练；本任务不对真实数据集图片做实际标注。

## 7. CVAT 备用工作流 / CVAT fallback

如果 LabelImg 无法安装或桌面 GUI 无法启动，可使用 CVAT：

1. 新建 CVAT task，上传发票/票据图片。
2. Label/class 只创建一个：`qr_code`。
3. 按本规范绘制紧贴二维码可见外轮廓的矩形框。
4. 导出时选择 **YOLO**、**Ultralytics YOLO** 或等价 YOLO 检测格式。
5. 检查导出的标签：每行必须为 `0 x_center y_center width height`，类别文件或 `data.yaml` 中只能出现 `qr_code`。
6. 如果 CVAT 导出包含 `data.yaml`，Task 6 再统一整理；本任务不创建 `dataset/data.yaml`。

## 8. 质量检查清单 / QA checklist

- `dataset/classes.txt` 恰好一行：`qr_code`。
- 所有正样本标签行都以 class id `0` 开头。
- bbox 紧贴二维码可见外边界，不包含非二维码区域。
- multiple、occluded、rotated、partial、blurry、tiny、low-resolution、unreadable QR、no-QR negative samples 均按上方规则处理。
- 无二维码图片不应出现任何 `qr_code` 标注。
