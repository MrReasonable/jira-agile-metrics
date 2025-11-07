"""Shared assertion helpers to keep tests DRY and satisfy pylint duplicate-code.

These helpers validate the basic structure of forecast output files across
CSV, JSON, and XLSX formats.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _assert_forecast_file_structure(df: pd.DataFrame, format_name: str) -> None:
    """Assert that a forecast DataFrame has the expected structure.

    Args:
        df: The DataFrame to validate
        format_name: The format name (e.g., "CSV", "JSON", "XLSX") for error messages
    """
    assert "Date" in df.columns, f"{format_name} should have Date column"
    assert len(df.columns) > 1, f"{format_name} should have trial columns"


def assert_forecast_csv_file_valid(file_path: Path) -> pd.DataFrame:
    """Assert that a forecast CSV file exists and has the expected columns.

    Returns the loaded DataFrame for further test-specific assertions.
    """
    assert file_path.exists(), "CSV file should be created"
    assert file_path.stat().st_size > 0, "CSV file should not be empty"

    df = pd.read_csv(file_path, parse_dates=["Date"])
    _assert_forecast_file_structure(df, "CSV")
    return df


def assert_forecast_json_file_valid(file_path: Path) -> pd.DataFrame:
    """Assert that a forecast JSON file exists and has the expected columns.

    Returns the loaded DataFrame for further test-specific assertions.
    """
    assert file_path.exists(), "JSON file should be created"
    assert file_path.stat().st_size > 0, "JSON file should not be empty"

    df = pd.read_json(file_path, orient="records")
    _assert_forecast_file_structure(df, "JSON")
    return df


def assert_forecast_xlsx_file_valid(file_path: Path) -> pd.DataFrame:
    """Assert that a forecast XLSX file exists and has the expected columns.

    Returns the loaded DataFrame for further test-specific assertions.
    """
    assert file_path.exists(), "XLSX file should be created"
    assert file_path.stat().st_size > 0, "XLSX file should not be empty"

    df = pd.read_excel(file_path, sheet_name="Forecast Trials")
    _assert_forecast_file_structure(df, "XLSX")
    return df
