# pyright: reportUnusedCallResult=false
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


EXCLUDE_PARTS = {".venv", ".env", "__pycache__", ".pytest_cache", ".omo", ".sisyphus", ".playwright-mcp", "runs"}
EXCLUDE_NAMES = {".DS_Store"}
EXCLUDE_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".zip", ".pt", ".onnx"}
INCLUDE_RUNTIME_FILES = {Path("runtime/models/yolo/best.pt")}
CODE_ROOT = "code"
DEMO_ROOT = "演示"
REPORT_PDF_ROOT = "报告pdf版本"
SCREENSHOT_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def delivery_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDE_PARTS for part in relative.parts):
            continue
        if path.name in EXCLUDE_NAMES or any(part.endswith(".egg-info") for part in relative.parts):
            continue
        if relative in INCLUDE_RUNTIME_FILES:
            files.append(relative)
            continue
        if path.suffix.lower() in EXCLUDE_SUFFIXES:
            continue
        if relative.as_posix() == "report/智能发票识别与自动报销填单系统_课程报告.docx":
            continue
        if relative.parts[:1] == ("runtime",) and path.name != ".gitkeep":
            continue
        files.append(relative)
    return sorted(files)


def archive_name(relative: Path) -> str:
    if is_report_pdf(relative):
        return (Path(REPORT_PDF_ROOT) / relative.name).as_posix()
    if is_screenshot(relative):
        return (Path(DEMO_ROOT) / relative.name).as_posix()
    return (Path(CODE_ROOT) / relative).as_posix()


def is_report_pdf(relative: Path) -> bool:
    return relative.parts[:1] == ("report",) and relative.suffix.lower() == ".pdf"


def is_screenshot(relative: Path) -> bool:
    if relative.suffix.lower() not in SCREENSHOT_SUFFIXES:
        return False
    if relative.parts[:2] == ("evidence", "screenshots"):
        return True
    return "截图" in relative.stem or "screenshot" in relative.stem.lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or build the three-part project delivery package.")
    parser.add_argument("--dry-run", action="store_true", help="Print files that would be packaged.")
    parser.add_argument("--zip", dest="zip_path", type=Path, help="Create a ZIP package at the given path.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = delivery_files(root)
    if args.dry_run:
        for path in files:
            print(archive_name(path))
        print(f"DRY_RUN_FILE_COUNT={len(files)}")
        return
    if args.zip_path is not None:
        target = args.zip_path
        if not target.is_absolute():
            target = root / target
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(root / path, archive_name(path))
        print(f"ZIP_CREATED={target}")
        print(f"ZIP_FILE_COUNT={len(files)}")
        return
    print("Use --dry-run to inspect package contents, or --zip <path> to create a ZIP.")


if __name__ == "__main__":
    main()
