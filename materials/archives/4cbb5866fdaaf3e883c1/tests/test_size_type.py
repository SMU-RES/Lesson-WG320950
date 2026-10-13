from container_ocr.parsing import extract_size_type_codes


def test_extracts_iso_size_type_code() -> None:
    results = extract_size_type_codes("MSCU1234566 22G1")
    assert [item.code for item in results] == ["22G1"]


def test_extracts_marketing_size_code() -> None:
    results = extract_size_type_codes("40HQ MAEU1234567")
    assert [item.code for item in results] == ["40HQ"]


def test_ignores_unrelated_four_character_words() -> None:
    assert extract_size_type_codes("PORT YARD TEST") == []
