# 证据目录

该目录保存课程验收用的真实公开发票样例、来源说明、处理图片和 Web 页面截图。

## 真实样例

本次交付使用 `guanshuicheng/invoice` 仓库中的 `test-invoice/电子发票.png`，许可证为 MIT License。文件已保存为：

- `evidence/real_samples/real_e_invoice.png`
- `evidence/real_samples/real_e_invoice_qr_crop.png`
- `evidence/real_samples/real_e_invoice_annotated.png`
- `evidence/real_samples/SOURCE.md`

原图中的二维码肉眼可见。证据核验时用临时 zxing-cpp 解码得到：

```text
01,10,012001800311,33207675,46.62,20190508,76939056883466677916,E1BD,
```

临时 zxing-cpp 只用于确认样例二维码内容，不是运行时应用的一部分。

## 截图

`evidence/screenshots/01_upload.png` 到 `12_duplicate_popup.png` 已用这张公开真实发票样例重新生成，覆盖上传、二维码区域、裁剪、解析、OCR 文本、字段抽取、校验、自动填单、校验通过弹窗和重复报销拦截弹窗。

## 技术边界

代码中已经有严格真实模式适配器：YOLO 检测 QR 区域，OpenCV 只解析 YOLO 裁剪图，PaddleOCR 在隔离子进程中做本地 GPU OCR，NVIDIA NIM 的 OpenAI 兼容在线 LLM API 做字段抽取。默认接口地址为 `https://integrate.api.nvidia.com/v1`，默认模型为 `meta/llama-3.3-70b-instruct`。真实模式证据需要以 `scripts/check_real_mode.py`、本地 YOLO 权重、真实 PaddleOCR 运行和在线 LLM 调用共同确认；本目录中的证据说明公开真实发票样例、本地页面流程和真实二维码核验结果。
