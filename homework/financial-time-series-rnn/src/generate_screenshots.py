from __future__ import annotations

from pathlib import Path
from textwrap import wrap

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

from train_rnn import CSV_PATH, PLOT_PATH, SimpleRNNRegressor


ROOT_DIR = Path(__file__).resolve().parent.parent
SCREENSHOT_DIR = ROOT_DIR / "outputs" / "screenshots"
CODE_PATH = ROOT_DIR / "src" / "train_rnn.py"


def ensure_output_dir() -> None:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


def save_text_block_image(
    lines: list[str], output_path: Path, title: str, *, font_size: int = 10, line_height: float = 0.032
) -> None:
    figure_height = max(6, len(lines) * line_height + 1.5)
    fig, ax = plt.subplots(figsize=(14, figure_height))
    ax.axis("off")
    ax.set_title(title, fontsize=16, pad=20)

    y_position = 0.98
    for line in lines:
        ax.text(
            0.02,
            y_position,
            line,
            family="monospace",
            fontsize=font_size,
            va="top",
            transform=ax.transAxes,
        )
        y_position -= line_height

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_dataset_screenshot() -> Path:
    data_frame = pd.read_csv(CSV_PATH)
    dataset_preview = data_frame.head(18).to_string(index=False)
    summary_lines = [
        f"Rows: {len(data_frame)}",
        f"Columns: {', '.join(data_frame.columns)}",
        "",
        "Preview:",
        dataset_preview,
    ]
    output_path = SCREENSHOT_DIR / "dataset_screenshot.png"
    expanded_lines: list[str] = []
    for line in summary_lines:
        expanded_lines.extend(wrap(line, width=110) or [""])
    save_text_block_image(expanded_lines, output_path, "Dataset Screenshot")
    return output_path


def generate_code_screenshot() -> Path:
    code_lines = CODE_PATH.read_text(encoding="utf-8").splitlines()
    selected_lines = code_lines[127:183]
    numbered_lines = [f"{line_number:>3}: {content}" for line_number, content in enumerate(selected_lines, start=128)]
    output_path = SCREENSHOT_DIR / "code_screenshot.png"
    save_text_block_image(numbered_lines, output_path, "Code Screenshot - SimpleRNNRegressor and Training Loop")
    return output_path


def generate_network_structure_screenshot() -> Path:
    model = SimpleRNNRegressor()
    lines = [
        "Model architecture printed by the script:",
        "",
        *str(model).splitlines(),
    ]
    output_path = SCREENSHOT_DIR / "network_structure_screenshot.png"
    save_text_block_image(lines, output_path, "Network Structure Screenshot", font_size=12, line_height=0.05)
    return output_path


def generate_submission_collage() -> Path:
    collage_path = SCREENSHOT_DIR / "submission_collage.png"
    dataset_image = mpimg.imread(SCREENSHOT_DIR / "dataset_screenshot.png")
    code_image = mpimg.imread(SCREENSHOT_DIR / "code_screenshot.png")
    network_image = mpimg.imread(SCREENSHOT_DIR / "network_structure_screenshot.png")
    result_image = mpimg.imread(PLOT_PATH)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    panels = [
        (axes[0, 0], dataset_image, "1. Dataset"),
        (axes[0, 1], code_image, "2. Code"),
        (axes[1, 0], network_image, "3. Network Structure"),
        (axes[1, 1], result_image, "4. Result Plot"),
    ]

    for axis, image, title in panels:
        axis.imshow(image)
        axis.set_title(title, fontsize=14)
        axis.axis("off")

    fig.suptitle("Assignment Submission Screenshot Materials", fontsize=18)
    fig.text(
        0.5,
        0.02,
        "Note: The required personal picture from the digital platform must be added manually by the student.",
        ha="center",
        fontsize=12,
        color="darkred",
    )
    plt.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(collage_path, dpi=200)
    plt.close(fig)
    return collage_path


def main() -> None:
    ensure_output_dir()

    dataset_path = generate_dataset_screenshot()
    code_path = generate_code_screenshot()
    network_path = generate_network_structure_screenshot()
    collage_path = generate_submission_collage()

    print(f"Generated dataset screenshot: {dataset_path}")
    print(f"Generated code screenshot: {code_path}")
    print(f"Generated network screenshot: {network_path}")
    print(f"Generated submission collage: {collage_path}")
    print("Reminder: add your personal picture from the digital platform manually before final submission.")


if __name__ == "__main__":
    main()
