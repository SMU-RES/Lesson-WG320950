# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnusedFunction=false
from __future__ import annotations

from html import escape
import os
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from flask import Flask, Response, abort, request, send_from_directory
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from .adapters import SampleLlmExtractor, SampleOcrEngine, SampleQrDecoder, SampleQrDetector
from .adapters.llm_extractor import OpenAICompatibleLlmExtractor
from .adapters.paddle_ocr import PaddleOcrEngine
from .adapters.qr_decoder import OpenCvQrDecoder
from .adapters.yolo_detector import YoloQrDetector
from .config import AppConfig
from .database import ReimbursementRepository
from .models import ProcessingResult
from .pipeline import InvoicePipeline


ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_app(test_config: dict[str, object] | None = None) -> Flask:
    load_dotenv(PROJECT_ROOT / ".env")
    app = Flask(__name__)
    config = AppConfig.from_env()
    app.config.update(
        MODE=config.mode,
        UPLOAD_DIR=config.upload_dir,
        ARTIFACT_DIR=config.artifact_dir,
        DATABASE_PATH=config.database_path,
        YOLO_WEIGHTS=config.yolo_weights,
        YOLO_CONF=config.yolo_conf,
        YOLO_IMGSZ=config.yolo_imgsz,
        YOLO_DEVICE=config.yolo_device,
        PADDLE_DEVICE=config.paddle_device,
        SECRET_KEY=config.secret_key,
        RESULT_STORE={},
    )
    app.config.update(test_config or {})
    for key in ("UPLOAD_DIR", "ARTIFACT_DIR"):
        app.config[key] = Path(app.config[key]).resolve()
        Path(app.config[key]).mkdir(parents=True, exist_ok=True)
    app.config["DATABASE_PATH"] = Path(app.config["DATABASE_PATH"]).resolve()
    repository = ReimbursementRepository(Path(app.config["DATABASE_PATH"]))
    repository.initialize()
    app.config["REPOSITORY"] = repository

    @app.get("/")
    def index() -> str:
        return _page(
            "上传发票",
            """
            <section class="card">
              <h1>智能发票识别与自动报销填单系统</h1>
              <p>上传真实发票图片后，系统会先定位二维码并裁剪，再完成二维码解析、OCR 识别、字段抽取、信息校验和重复报销检查。</p>
              <form action="/process" method="post" enctype="multipart/form-data">
                <label>上传发票图片 <input type="file" name="invoice" accept="image/*" required></label>
                <button type="submit">开始识别</button>
              </form>
            </section>
            """,
        )

    @app.post("/process")
    def process_upload() -> Response:
        uploaded = request.files.get("invoice")
        if uploaded is None or uploaded.filename == "":
            abort(Response("请选择发票图片", status=400))
        if not _allowed(uploaded):
            abort(Response("仅支持图片文件", status=400))
        source_path = _save_upload(uploaded, Path(app.config["UPLOAD_DIR"]))
        pipeline = _build_pipeline(app)
        result = pipeline.process(source_path)
        store = app.config["RESULT_STORE"]
        store[result.job_id] = result
        response = Response(_render_result(result), status=200, mimetype="text/html")
        response.headers["X-Job-Id"] = result.job_id
        return response

    @app.post("/submit/<job_id>")
    def submit(job_id: str) -> Response:
        if not _is_job_id(job_id):
            abort(Response("处理任务编号无效", status=404))
        result = app.config["RESULT_STORE"].get(job_id)
        if result is None:
            abort(Response("处理任务不存在", status=404))
        if result.validation.status == "fail":
            return Response(
                _page("提交结果", _modal("校验失败，禁止提交", "fail") + '<section class="card"><h1>提交结果</h1><p>校验失败，禁止提交</p></section>'),
                status=409,
                mimetype="text/html",
            )
        repo = app.config["REPOSITORY"]
        submit_result = repo.submit(result.form, result.validation)
        status = 200 if submit_result.accepted else 409
        modal_tone = "pass" if submit_result.accepted else "fail"
        body = _modal(submit_result.message, modal_tone) + f'<section class="card"><h1>提交结果</h1><p>{_h(submit_result.message)}</p></section>'
        return Response(_page("提交结果", body), status=status, mimetype="text/html")

    @app.get("/artifacts/<job_id>/<filename>")
    def artifact(job_id: str, filename: str) -> Response:
        if not _is_job_id(job_id):
            abort(404)
        artifact_dir = Path(app.config["ARTIFACT_DIR"]) / job_id
        return send_from_directory(artifact_dir, filename)

    @app.get("/favicon.ico")
    def favicon() -> Response:
        return Response(status=204)

    return app


def main() -> None:
    create_app().run(debug=_flask_debug_enabled())


def _flask_debug_enabled() -> bool:
    return os.getenv("INVOICE_APP_FLASK_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _allowed(uploaded: FileStorage) -> bool:
    suffix = Path(uploaded.filename or "").suffix.lower()
    return suffix in ALLOWED_EXTENSIONS


def _save_upload(uploaded: FileStorage, upload_dir: Path) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)
    name = secure_filename(uploaded.filename or "invoice.png") or "invoice.png"
    target = upload_dir / f"{uuid4().hex}_{name}"
    uploaded.save(target)
    return target


def _is_job_id(value: str) -> bool:
    return len(value) == 32 and all(char in "0123456789abcdef" for char in value.lower())


def _build_pipeline(app: Flask) -> InvoicePipeline:
    mode = str(app.config.get("MODE", "sample"))
    artifact_dir = Path(app.config["ARTIFACT_DIR"])
    if mode == "real":
        detector = YoloQrDetector(
            Path(app.config["YOLO_WEIGHTS"]),
            conf=float(app.config["YOLO_CONF"]),
            imgsz=int(app.config["YOLO_IMGSZ"]),
            device=str(app.config["YOLO_DEVICE"]),
        )
        return InvoicePipeline(detector, OpenCvQrDecoder(), PaddleOcrEngine(str(app.config["PADDLE_DEVICE"])), OpenAICompatibleLlmExtractor(), artifact_dir)
    return InvoicePipeline(SampleQrDetector(), SampleQrDecoder(), SampleOcrEngine(), SampleLlmExtractor(), artifact_dir)


def _render_result(result: ProcessingResult) -> str:
    findings = "".join(f"<li>{_h(finding.field)}：{_h(finding.status)}，{_h(finding.message)}</li>" for finding in result.validation.findings)
    annotated_image = _artifact_img(result, result.qr_detection.annotated_path, "YOLO 二维码检测图")
    crop_image = _artifact_img(result, result.qr_detection.crop_path, "二维码裁剪图")
    modal_message = _validation_modal_message(result.validation.status)
    html = f"""
    {_modal(modal_message, result.validation.status)}
    <section class="card">
      <h1>识别结果</h1>
      <p class="status">校验状态：{_h(result.validation.status)}</p>
      <div class="grid">
        <section class="panel" id="section-yolo">
          <h2>YOLO 二维码检测框</h2>
          {annotated_image}
          <p>bbox={_h(str(result.qr_detection.bbox_xyxy))} confidence={result.qr_detection.confidence:.2f}</p>
        </section>
        <section class="panel" id="section-qr-crop">
          <h2>二维码裁剪图</h2>
          {crop_image}
          <p>{_h(str(result.qr_detection.crop_path))}</p>
        </section>
      </div>
      <section class="panel" id="section-qr-parse">
        <h2>二维码解析结果</h2>
        <p>{_h(result.qr_result.raw_text)}</p>
      </section>
      <section class="panel" id="section-ocr">
        <h2>PaddleOCR 离线识别结果</h2>
        <pre>{_h(result.ocr_result.text)}</pre>
      </section>
      <section class="panel" id="section-llm">
        <h2>LLM 字段抽取结果</h2>
        <p>发票号码：{_h(result.llm_fields.invoice_number)}；价税合计：{result.llm_fields.total_amount:.2f}</p>
      </section>
      <section class="panel" id="section-validation">
        <h2>QR 与 OCR/LLM 校验结果</h2>
        <ul>{findings}</ul>
      </section>
      <section class="panel" id="section-form">
        <h2>自动报销表单</h2>
        <p>申请人：{_h(result.form.applicant.name)}；费用类型：{_h(result.form.expense_type)}；金额：{result.form.amount:.2f}</p>
      </section>
      <form action="/submit/{result.job_id}" method="post"><button type="submit">提交报销</button></form>
    </section>
    """
    return _page("识别结果", html)


def _artifact_img(result: ProcessingResult, path: Path | None, alt: str) -> str:
    if path is None:
        return ""
    return f'<img src="/artifacts/{_h(result.job_id)}/{_h(path.name)}" alt="{_h(alt)}">'


def _h(value: str) -> str:
    return escape(value, quote=True)


def _validation_modal_message(status: str) -> str:
    if status == "pass":
        return "提示弹窗：校验通过，可提交报销"
    if status == "warning":
        return "提示弹窗：校验存在异常，请人工复核"
    return "提示弹窗：校验失败，禁止提交"


def _modal(message: str, tone: str) -> str:
    safe_tone = tone if tone in {"pass", "warning", "fail"} else "warning"
    return f'<aside class="modal {safe_tone}" role="alertdialog" aria-live="assertive"><strong>提示弹窗</strong><p>{_h(message)}</p></aside>'


def _page(title: str, body: str) -> str:
    return f"""
    <!doctype html>
    <html lang="zh-CN">
    <head>
      <meta charset="utf-8">
      <title>{title}</title>
      <style>
        body {{ margin: 0; font-family: "Microsoft YaHei", Arial, sans-serif; background: #f6f1e8; color: #2f261f; }}
        .card {{ max-width: 980px; margin: 40px auto; padding: 28px; background: #fffdf8; border-radius: 18px; box-shadow: 0 12px 36px rgba(47, 38, 31, .14); }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
        .panel {{ margin-top: 18px; padding: 18px; border: 1px solid #e7d6bd; border-radius: 14px; background: #fbf6ed; }}
        .status {{ display: inline-block; padding: 8px 14px; border-radius: 999px; background: #e7f4df; color: #14532d; font-weight: 700; }}
        .modal {{ max-width: 980px; margin: 24px auto -18px; padding: 16px 20px; border-radius: 16px; background: #fffdf8; box-shadow: 0 10px 28px rgba(47, 38, 31, .14); }}
        .modal.pass {{ border-left: 6px solid #2f7d32; }}
        .modal.warning {{ border-left: 6px solid #b45309; }}
        .modal.fail {{ border-left: 6px solid #991b1b; }}
        .modal p {{ margin: 8px 0 0; }}
        img {{ max-width: 100%; border-radius: 12px; border: 1px solid #d8c5a7; background: white; }}
        h1 {{ color: #4a2f1a; }}
        h2 {{ margin-top: 24px; color: #6b3f16; }}
        button {{ padding: 10px 18px; border: 0; border-radius: 10px; background: #7c3f12; color: white; cursor: pointer; }}
        input {{ margin: 12px 0; }}
        pre {{ white-space: pre-wrap; background: #241c17; color: #f8f1e7; padding: 16px; border-radius: 12px; }}
      </style>
    </head>
    <body>{body}</body>
    </html>
    """
