"""Shared assertion helpers to keep tests DRY and satisfy pylint duplicate-code.

These helpers validate the basic structure of forecast output files across
CSV, JSON, and XLSX formats.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def assert_forecast_csv_file_valid(file_path: Path) -> pd.DataFrame:
    """Assert that a forecast CSV file exists and has the expected columns.

    Returns the loaded DataFrame for further test-specific assertions.
    """
    assert file_path.exists(), "CSV file should be created"
    assert file_path.stat().st_size > 0, "CSV file should not be empty"

    df = pd.read_csv(file_path, parse_dates=["Date"])
    assert "Date" in df.columns, "CSV should have Date column"
    assert len(df.columns) > 1, "CSV should have trial columns"
    return df


def assert_forecast_json_file_valid(file_path: Path) -> pd.DataFrame:
    """Assert that a forecast JSON file exists and has the expected columns.

    Returns the loaded DataFrame for further test-specific assertions.
    """
    assert file_path.exists(), "JSON file should be created"
    assert file_path.stat().st_size > 0, "JSON file should not be empty"

    df = pd.read_json(file_path, orient="records")
    assert "Date" in df.columns, "JSON should have Date column"
    assert len(df.columns) > 1, "JSON should have trial columns"
    return df


def assert_forecast_xlsx_file_valid(file_path: Path) -> pd.DataFrame:
    """Assert that a forecast XLSX file exists and has the expected columns.

    Returns the loaded DataFrame for further test-specific assertions.
    """
    assert file_path.exists(), "XLSX file should be created"
    assert file_path.stat().st_size > 0, "XLSX file should not be empty"

    df = pd.read_excel(file_path, sheet_name="Forecast Trials")
    assert "Date" in df.columns, "XLSX should have Date column"
    assert len(df.columns) > 1, "XLSX should have trial columns"
    return df
