# pyright: reportMissingTypeStubs=false
from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

import invoice_app.adapters.paddle_ocr as paddle_module
from invoice_app.adapters.paddle_ocr import PaddleOcrEngine
from invoice_app.adapters.paddle_ocr_worker import _extract_lines


def test_paddle_ocr_engine_invokes_isolated_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[list[str], int]] = []

    def fake_run(command: list[str], capture_output: bool, text: bool, check: bool, timeout: int) -> subprocess.CompletedProcess[str]:
        assert capture_output
        assert text
        assert not check
        calls.append((command, timeout))
        return subprocess.CompletedProcess(command, 0, '{"lines": ["发票号码 33207675", "价税合计 52.70"]}', "")

    monkeypatch.setattr(paddle_module.subprocess, "run", fake_run)

    result = PaddleOcrEngine(device="gpu:0", timeout_seconds=12).recognize(tmp_path / "invoice.png")

    command, timeout = calls[0]
    assert timeout == 12
    assert command[1:4] == ["-m", "invoice_app.adapters.paddle_ocr_worker", "--image"]
    assert command[-2:] == ["--device", "gpu:0"]
    assert result.lines == ["发票号码 33207675", "价税合计 52.70"]
    assert result.text == "发票号码 33207675\n价税合计 52.70"


def test_paddle_ocr_engine_reports_worker_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(command: list[str], capture_output: bool, text: bool, check: bool, timeout: int) -> subprocess.CompletedProcess[str]:
        _ = capture_output
        _ = text
        _ = check
        _ = timeout
        return subprocess.CompletedProcess(command, 1, "", "cuda dll load failed")

    monkeypatch.setattr(paddle_module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="exit code 1"):
        PaddleOcrEngine().recognize(tmp_path / "invoice.png")


def test_paddle_ocr_engine_rejects_malformed_worker_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(command: list[str], capture_output: bool, text: bool, check: bool, timeout: int) -> subprocess.CompletedProcess[str]:
        _ = capture_output
        _ = text
        _ = check
        _ = timeout
        return subprocess.CompletedProcess(command, 0, "not-json", "")

    monkeypatch.setattr(paddle_module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="invalid JSON"):
        PaddleOcrEngine().recognize(tmp_path / "invoice.png")


def test_paddle_ocr_engine_reports_worker_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(command: list[str], capture_output: bool, text: bool, check: bool, timeout: int) -> subprocess.CompletedProcess[str]:
        _ = capture_output
        _ = text
        _ = check
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(paddle_module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="timed out after 3 seconds"):
        PaddleOcrEngine(timeout_seconds=3).recognize(tmp_path / "invoice.png")


def test_worker_extracts_rec_texts_from_pages() -> None:
    raw_pages = [{"rec_texts": ["第一行", "第二行"]}, {"rec_texts": [33207675]}, object()]

    assert _extract_lines(raw_pages) == ["第一行", "第二行", "33207675"]
