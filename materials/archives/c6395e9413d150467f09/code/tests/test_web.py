# pyright: reportMissingParameterType=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnusedCallResult=false
from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image

import invoice_app.app as app_module
from invoice_app.app import create_app


def _sample_config(tmp_path: Path) -> dict[str, object]:
    return {"TESTING": True, "MODE": "sample", "DATABASE_PATH": tmp_path / "db.sqlite3", "UPLOAD_DIR": tmp_path / "uploads", "ARTIFACT_DIR": tmp_path / "artifacts"}


def _png_bytes() -> BytesIO:
    stream = BytesIO()
    Image.new("RGB", (300, 200), "white").save(stream, format="PNG")
    stream.seek(0)
    return stream


def test_index_renders_upload_form(tmp_path: Path) -> None:
    app = create_app(_sample_config(tmp_path))
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert "上传发票" in response.get_data(as_text=True)


def test_real_sample_upload_renders_full_result(tmp_path: Path) -> None:
    app = create_app(_sample_config(tmp_path))
    response = app.test_client().post("/process", data={"invoice": (_png_bytes(), "invoice.png")}, content_type="multipart/form-data")
    body = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "校验状态：pass" in body
    assert "提示弹窗：校验通过" in body
    assert "33207675" in body
    assert "01,10,012001800311,33207675,46.62,20190508" in body
    assert "二维码裁剪图" in body
    assert "自动报销表单" in body


def test_invalid_upload_is_rejected(tmp_path: Path) -> None:
    app = create_app(_sample_config(tmp_path))
    response = app.test_client().post("/process", data={"invoice": (BytesIO(b"not image"), "bad.txt")}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "仅支持图片文件" in response.get_data(as_text=True)


def test_duplicate_submit_is_blocked(tmp_path: Path) -> None:
    app = create_app(_sample_config(tmp_path))
    client = app.test_client()
    processed = client.post("/process", data={"invoice": (_png_bytes(), "invoice.png")}, content_type="multipart/form-data")
    job_id = processed.headers["X-Job-Id"]
    first = client.post(f"/submit/{job_id}")
    second = client.post(f"/submit/{job_id}")
    assert first.status_code == 200
    assert "提交成功" in first.get_data(as_text=True)
    assert "提示弹窗" in first.get_data(as_text=True)
    assert second.status_code == 409
    assert "重复报销" in second.get_data(as_text=True)
    assert "提示弹窗" in second.get_data(as_text=True)


def test_artifact_route_rejects_invalid_job_id(tmp_path: Path) -> None:
    app = create_app(_sample_config(tmp_path))
    response = app.test_client().get("/artifacts/..%2F..%2Fetc/passwd")
    assert response.status_code == 404


def test_flask_debug_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("INVOICE_APP_FLASK_DEBUG", raising=False)
    assert app_module._flask_debug_enabled() is False


def test_flask_debug_requires_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("INVOICE_APP_FLASK_DEBUG", "true")
    assert app_module._flask_debug_enabled() is True


def test_project_root_dotenv_is_loaded(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("INVOICE_APP_MODE=sample\nINVOICE_APP_YOLO_CONF=0.33\nINVOICE_APP_YOLO_IMGSZ=1280\nINVOICE_APP_YOLO_DEVICE=cpu\nINVOICE_APP_PADDLE_DEVICE=cpu\n", encoding="utf-8")
    monkeypatch.delenv("INVOICE_APP_MODE", raising=False)
    monkeypatch.delenv("INVOICE_APP_YOLO_CONF", raising=False)
    monkeypatch.delenv("INVOICE_APP_YOLO_IMGSZ", raising=False)
    monkeypatch.delenv("INVOICE_APP_YOLO_DEVICE", raising=False)
    monkeypatch.setattr(app_module, "PROJECT_ROOT", tmp_path)
    app = create_app({"TESTING": True, "DATABASE_PATH": tmp_path / "db.sqlite3", "UPLOAD_DIR": tmp_path / "uploads", "ARTIFACT_DIR": tmp_path / "artifacts"})
    assert app.config["MODE"] == "sample"
    assert app.config["YOLO_CONF"] == 0.33
    assert app.config["YOLO_IMGSZ"] == 1280
    assert app.config["YOLO_DEVICE"] == "cpu"
    assert app.config["PADDLE_DEVICE"] == "cpu"


def test_real_mode_yolo_env_config_is_loaded(monkeypatch, tmp_path: Path) -> None:
    weights = tmp_path / "weights.pt"
    monkeypatch.setenv("INVOICE_APP_MODE", "real")
    monkeypatch.setenv("INVOICE_APP_YOLO_WEIGHTS", str(weights))
    monkeypatch.setenv("INVOICE_APP_YOLO_CONF", "0.42")
    monkeypatch.setenv("INVOICE_APP_YOLO_IMGSZ", "1600")
    monkeypatch.setenv("INVOICE_APP_YOLO_DEVICE", "0")
    monkeypatch.setenv("INVOICE_APP_PADDLE_DEVICE", "gpu:0")
    app = create_app({"TESTING": True, "DATABASE_PATH": tmp_path / "db.sqlite3", "UPLOAD_DIR": tmp_path / "uploads", "ARTIFACT_DIR": tmp_path / "artifacts"})
    assert app.config["YOLO_WEIGHTS"] == weights
    assert app.config["YOLO_CONF"] == 0.42
    assert app.config["YOLO_IMGSZ"] == 1600
    assert app.config["YOLO_DEVICE"] == "0"
    assert app.config["PADDLE_DEVICE"] == "gpu:0"
