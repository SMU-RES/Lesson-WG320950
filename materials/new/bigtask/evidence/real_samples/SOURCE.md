# 真实发票样例来源

本目录使用公开项目 `guanshuicheng/invoice` 的测试发票图片：

- 仓库：https://github.com/guanshuicheng/invoice
- 文件：`test-invoice/电子发票.png`
- 许可证：MIT License，Copyright 2018 chineseocr

本项目将该图片保存为 `real_e_invoice.png`，并从左上角真实二维码裁剪得到 `real_e_invoice_qr_crop.png`。二维码经 `zxing-cpp` 临时环境解码，内容为：

```text
01,10,012001800311,33207675,46.62,20190508,76939056883466677916,E1BD,
```

图片中的关键字段：发票代码 `012001800311`，发票号码 `33207675`，开票日期 `2019-05-08`，金额 `46.62`，税额 `6.08`，价税合计 `52.70`，销售方 `天津瑞佳讯贸易有限公司`。
