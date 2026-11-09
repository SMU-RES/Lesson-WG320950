import pytest

from container_ocr.parsing import calculate_check_digit, extract_container_numbers, validate_container_number


def test_calculates_known_check_digit() -> None:
    assert calculate_check_digit("MSCU123456") == 6
    assert calculate_check_digit("MAEU123456") == 7


def test_extracts_valid_number_with_separators() -> None:
    results = extract_container_numbers("Container MSCU 123456 6 / 22G1")
    assert [item.value for item in results] == ["MSCU1234566"]
    assert results[0].is_valid is True


def test_reports_invalid_check_digit() -> None:
    result = validate_container_number("MSCU1234567")
    assert result.is_valid is False
    assert result.calculated_check_digit == 6
    assert "expected 6" in result.reason


def test_extracts_multiple_candidates() -> None:
    results = extract_container_numbers("MAEU1234567 and MSCU-123456-6")
    assert [item.value for item in results] == ["MAEU1234567", "MSCU1234566"]


def test_rejects_non_candidates() -> None:
    with pytest.raises(ValueError):
        validate_container_number("ABC123")
