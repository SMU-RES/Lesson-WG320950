from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from PIL import Image

class CliModule(Protocol):
    def main(self, argv: list[str] | None = None) -> int: ...


cli = cast(CliModule, cast(object, import_module("segmentation_demo.cli")))


class MonkeyPatchLike(Protocol):
    def setattr(self, target: object, name: str, value: object) -> None: ...


class CaptureResultLike(Protocol):
    out: str
    err: str


class CapsysLike(Protocol):
    def readouterr(self) -> CaptureResultLike: ...


def test_cli_main_success(monkeypatch: MonkeyPatchLike, tmp_path: Path, capsys: CapsysLike) -> None:
    image_path = tmp_path / "input.jpg"
    Image.new("RGB", (16, 16), color="white").save(image_path)

    def fake_run_segmentation(**_kwargs: object) -> tuple[str, dict[str, object]]:
        out = tmp_path / "outputs" / "result.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        _ = out.write_bytes(b"fake")
        return str(out), {
            "num_instances": 1,
            "instances": [{"class_name": "person", "confidence": 0.9}],
        }

    monkeypatch.setattr(cli, "run_segmentation", fake_run_segmentation)

    code: int = cli.main(["--image", str(image_path), "--output-dir", str(tmp_path / "outputs")])
    captured = capsys.readouterr()

    assert code == 0
    assert "分割完成" in captured.out


def test_cli_main_failure(monkeypatch: MonkeyPatchLike, capsys: CapsysLike) -> None:
    def fake_run_segmentation(**_kwargs: object) -> tuple[str, dict[str, object]]:
        raise ValueError("bad conf")

    monkeypatch.setattr(cli, "run_segmentation", fake_run_segmentation)

    code: int = cli.main(["--image", "missing.jpg"])
    captured = capsys.readouterr()

    assert code == 1
    assert "推理失败" in captured.err
