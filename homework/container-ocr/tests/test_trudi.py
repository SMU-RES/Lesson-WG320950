from pathlib import Path

from container_ocr.trudi import find_images, prepare_samples, select_images


def test_find_images_recurses_supported_suffixes(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "a.jpg").write_bytes(b"x")
    (nested / "b.txt").write_text("skip")
    assert [path.name for path in find_images(tmp_path)] == ["a.jpg"]


def test_select_images_is_deterministic(tmp_path: Path) -> None:
    images = [tmp_path / f"{index}.jpg" for index in range(5)]
    first = select_images(images, limit=3, seed=42)
    second = select_images(images, limit=3, seed=42)
    assert first == second
    assert len(first) == 3


def test_prepare_samples_copies_subset(tmp_path: Path) -> None:
    source = tmp_path / "source"
    out = tmp_path / "out"
    source.mkdir()
    for name in ["a.jpg", "b.png", "c.txt"]:
        (source / name).write_bytes(b"x")
    copied = prepare_samples(source, limit=2, seed=1, out_dir=out)
    assert len(copied) == 2
    assert all(path.exists() for path in copied)
