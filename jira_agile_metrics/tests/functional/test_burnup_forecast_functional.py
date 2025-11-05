"""Functional tests for burnup forecast calculator."""

import random

import numpy as np
import pandas as pd
import pytest

from jira_agile_metrics.calculators.forecast import BurnupForecastCalculator
from jira_agile_metrics.tests.functional.conftest import (
    get_burnup_base_settings,
    get_default_forecast_settings,
    run_forecast_calculators,
    validate_forecast_result_structure,
    validate_forecast_trial_values,
)
from jira_agile_metrics.tests.helpers.assertions import assert_forecast_csv_file_valid


def test_burnup_forecast_generates_expected_data_structure(
    query_manager, simple_cycle_settings
):
    """Test that burnup forecast calculator generates expected data structure."""
    settings, _ = simple_cycle_settings

    # Set fixed random seed for reproducible results
    random.seed(42)
    np.random.seed(42)

    # Configure forecast settings with minimal trials for faster execution
    settings = get_burnup_base_settings(settings)
    settings.update(get_default_forecast_settings())
    settings.update({
        # Dummy path to allow validation to pass
        "burnup_forecast_chart": "/dev/null",
        "burnup_forecast_chart_trials": 20,  # More trials for better validation
    })

    results = run_forecast_calculators(query_manager, settings)
    forecast_result = results[BurnupForecastCalculator]

    # Verify forecast produces a DataFrame (or None if insufficient data)
    # With our minimal fixture, we should get a DataFrame with trial columns
    assert forecast_result is not None, (
        "Forecast should produce a result with valid data"
    )

    # Verify structure
    validate_forecast_result_structure(forecast_result)

    # Verify forecast extends beyond the last burnup date
    # Last date from fixtures is 2021-01-25, forecast should extend forward
    assert forecast_result.index.min() >= pd.Timestamp("2021-01-25"), (
        "Forecast should start from or after last burnup date"
    )

    # Verify all trial columns have numeric data
    for col in forecast_result.columns:
        assert pd.api.types.is_numeric_dtype(forecast_result[col]), (
            f"Trial column {col} should contain numeric data"
        )

    # Verify forecast has reasonable progression
    validate_forecast_trial_values(forecast_result)

    # Additional check: initial completed-items count should be >= 2
    # Fixture note: `jira_client` loads `tests/fixtures/jira/search_issues.json`
    # where the initial completed-items count is 2
    sample_trials = list(forecast_result.columns)[:3]
    for trial_col in sample_trials:
        trial_values = forecast_result[trial_col].dropna()
        if len(trial_values) > 0:
            assert trial_values.iloc[0] >= 2, (
                f"Trial {trial_col} should start at least at 2 (completed items)"
            )


def test_burnup_forecast_generates_csv_output(
    query_manager, simple_cycle_settings, tmp_path
):
    """Test that burnup forecast calculator generates CSV output matching
    expected structure.

    Verifies that the CSV output structure matches the forecast result DataFrame.
    """
    settings, _ = simple_cycle_settings

    # Set fixed random seed for reproducible results
    random.seed(42)
    np.random.seed(42)

    # Configure forecast settings
    settings = get_burnup_base_settings(settings)
    settings.update(get_default_forecast_settings())
    output_csv = tmp_path / "burnup-forecast-data.csv"
    settings.update({
        "burnup_forecast_chart": "/dev/null",  # Dummy path for validation
        "burnup_forecast_chart_data": str(output_csv),
        "burnup_forecast_chart_trials": 20,  # Enough for meaningful forecast
    })

    results = run_forecast_calculators(query_manager, settings)
    forecast_result = results[BurnupForecastCalculator]

    # Verify forecast produces a DataFrame
    assert forecast_result is not None, (
        "Forecast should produce a result with valid data"
    )

    # Verify CSV file and load
    csv_df = assert_forecast_csv_file_valid(output_csv)
    assert all(col.startswith("Trial ") for col in csv_df.columns if col != "Date"), (
        "All non-Date columns should be trial columns"
    )

    # Verify CSV structure matches forecast result
    assert len(csv_df) == len(forecast_result), "CSV should have same number of rows"
    assert len(csv_df.columns) == len(forecast_result.columns) + 1, (
        "CSV should have Date + trial columns"
    )

    # Verify dates match
    csv_dates = pd.to_datetime(csv_df["Date"])
    assert pd.Index(csv_dates).equals(forecast_result.index), (
        "CSV dates should match forecast index"
    )

    # Set CSV index to Date for proper alignment with forecast_result
    csv_df_indexed = csv_df.set_index("Date")

    # Verify trial values match (allow for small floating point differences)
    for trial_col in forecast_result.columns:
        csv_trial = csv_df_indexed[trial_col]
        forecast_trial = forecast_result[trial_col]
        pd.testing.assert_series_equal(
            csv_trial,
            forecast_trial,
            check_names=False,
            check_dtype=False,
            check_index_type=False,
            check_freq=False,
        )


def test_burnup_forecast_csv_reproducible_with_seed(
    query_manager, simple_cycle_settings, tmp_path
):
    """Test that CSV output is reproducible with fixed random seed."""
    settings, _ = simple_cycle_settings

    # Set fixed random seed for reproducible results
    random.seed(42)
    np.random.seed(42)

    # Configure forecast settings
    settings1 = get_burnup_base_settings(settings.copy())
    settings1.update(get_default_forecast_settings())
    output_csv1 = tmp_path / "burnup-forecast-data1.csv"
    settings1.update({
        "burnup_forecast_chart": "/dev/null",
        "burnup_forecast_chart_data": str(output_csv1),
        "burnup_forecast_chart_trials": 10,  # Small number for faster test
    })

    # Run first forecast
    random.seed(42)
    np.random.seed(42)
    results1 = run_forecast_calculators(query_manager, settings1)
    forecast_result1 = results1[BurnupForecastCalculator]

    # Run second forecast with same seed
    settings2 = get_burnup_base_settings(settings.copy())
    settings2.update(get_default_forecast_settings())
    output_csv2 = tmp_path / "burnup-forecast-data2.csv"
    settings2.update({
        "burnup_forecast_chart": "/dev/null",
        "burnup_forecast_chart_data": str(output_csv2),
        "burnup_forecast_chart_trials": 10,
    })

    random.seed(42)
    np.random.seed(42)
    results2 = run_forecast_calculators(query_manager, settings2)
    forecast_result2 = results2[BurnupForecastCalculator]

    # Both should produce results
    if forecast_result1 is not None and forecast_result2 is not None:
        # Compare CSV files - they should be identical with fixed seed
        if output_csv1.exists() and output_csv2.exists():
            df1 = pd.read_csv(output_csv1, parse_dates=["Date"])
            df2 = pd.read_csv(output_csv2, parse_dates=["Date"])

            # Compare DataFrames - they should be identical with fixed seed
            pd.testing.assert_frame_equal(df1, df2, check_dtype=False)


def test_burnup_forecast_json_output(query_manager, simple_cycle_settings, tmp_path):
    """Test that burnup forecast calculator generates JSON output."""
    settings, _ = simple_cycle_settings

    # Set fixed random seed for reproducible results
    random.seed(42)
    np.random.seed(42)

    # Configure forecast settings
    settings = get_burnup_base_settings(settings)
    settings.update(get_default_forecast_settings())
    output_json = tmp_path / "burnup-forecast-data.json"
    settings.update({
        "burnup_forecast_chart": "/dev/null",
        "burnup_forecast_chart_data": str(output_json),
        "burnup_forecast_chart_trials": 10,
    })

    results = run_forecast_calculators(query_manager, settings)
    forecast_result = results[BurnupForecastCalculator]

    if forecast_result is None:
        pytest.skip("Forecast could not be generated (insufficient data)")

    # Verify JSON file was created
    assert output_json.exists(), "JSON file should be created"
    assert output_json.stat().st_size > 0, "JSON file should not be empty"

    # Read and verify JSON structure
    json_df = pd.read_json(output_json, orient="records")
    assert "Date" in json_df.columns, "JSON should have Date column"
    assert len(json_df.columns) > 1, "JSON should have trial columns"
    assert len(json_df) == len(forecast_result), "JSON should have same number of rows"


def test_burnup_forecast_xlsx_output(query_manager, simple_cycle_settings, tmp_path):
    """Test that burnup forecast calculator generates XLSX output."""
    settings, _ = simple_cycle_settings

    # Set fixed random seed for reproducible results
    random.seed(42)
    np.random.seed(42)

    # Configure forecast settings
    settings = get_burnup_base_settings(settings)
    settings.update(get_default_forecast_settings())
    output_xlsx = tmp_path / "burnup-forecast-data.xlsx"
    settings.update({
        "burnup_forecast_chart": "/dev/null",
        "burnup_forecast_chart_data": str(output_xlsx),
        "burnup_forecast_chart_trials": 10,
    })

    results = run_forecast_calculators(query_manager, settings)
    forecast_result = results[BurnupForecastCalculator]

    if forecast_result is None:
        pytest.skip("Forecast could not be generated (insufficient data)")

    # Verify XLSX file was created
    assert output_xlsx.exists(), "XLSX file should be created"
    assert output_xlsx.stat().st_size > 0, "XLSX file should not be empty"

    # Read and verify XLSX structure
    xlsx_df = pd.read_excel(output_xlsx, sheet_name="Forecast Trials")
    assert "Date" in xlsx_df.columns, "XLSX should have Date column"
    assert len(xlsx_df.columns) > 1, "XLSX should have trial columns"
    assert len(xlsx_df) == len(forecast_result), "XLSX should have same number of rows"
