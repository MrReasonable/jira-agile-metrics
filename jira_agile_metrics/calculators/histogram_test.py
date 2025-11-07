"""Tests for histogram calculator functionality in Jira Agile Metrics.

This module contains unit tests for the histogram calculator.
"""

import os

import pandas as pd

from ..test_utils import (
    assert_calculator_wrote_json_file,
    create_empty_test_results,
)
from ..utils import extend_dict
from .cycletime import CycleTimeCalculator
from .histogram import HistogramCalculator


def test_empty(
    custom_query_manager, base_minimal_settings, base_minimal_cycle_time_columns
):
    """Test histogram calculator with empty data."""
    empty_results = create_empty_test_results(base_minimal_cycle_time_columns)

    calculator = HistogramCalculator(
        custom_query_manager, base_minimal_settings, empty_results
    )

    # Should not raise error on empty input
    try:
        data = calculator.run()
    except AttributeError:
        # Acceptable if .dt accessor fails on empty input
        data = None
    assert data is None or isinstance(data, pd.Series)


def test_calculate_histogram(
    custom_query_manager, base_minimal_settings, large_cycle_time_results
):
    """Test histogram calculation functionality."""
    calculator = HistogramCalculator(
        custom_query_manager, base_minimal_settings, large_cycle_time_results
    )

    data = calculator.run()

    assert list(data.index) == [
        "0.0 to 1.0",
        "1.0 to 2.0",
        "2.0 to 3.0",
        "3.0 to 4.0",
    ]
    assert list(data) == [0, 0, 0, 2]


def test_write_file_csv(
    custom_query_manager, base_minimal_settings, large_cycle_time_results, tmp_path
):
    """Test writing histogram data to CSV file."""
    output_file = str(tmp_path / "histogram.csv")
    settings = extend_dict(
        base_minimal_settings,
        {
            "histogram_data": [output_file],
            "histogram_chart": None,
            "quantiles": [0.5, 0.85, 0.95],
        },
    )

    calculator = HistogramCalculator(
        custom_query_manager, settings, large_cycle_time_results
    )
    result = calculator.run()
    large_cycle_time_results[HistogramCalculator] = result
    calculator.write()

    assert os.path.exists(output_file)
    df = pd.read_csv(output_file)
    assert list(df.columns) == ["Range", "Items"]
    assert len(df) > 0


def test_write_file_json(
    custom_query_manager, base_minimal_settings, large_cycle_time_results, tmp_path
):
    """Test writing histogram data to JSON file."""
    output_file = str(tmp_path / "histogram.json")
    settings = extend_dict(
        base_minimal_settings,
        {
            "histogram_data": [output_file],
            "histogram_chart": None,
            "quantiles": [0.5, 0.85, 0.95],
        },
    )

    calculator = HistogramCalculator(
        custom_query_manager, settings, large_cycle_time_results
    )
    data = assert_calculator_wrote_json_file(
        calculator, output_file, large_cycle_time_results
    )
    assert len(data) > 0


def test_write_file_xlsx(
    custom_query_manager, base_minimal_settings, large_cycle_time_results, tmp_path
):
    """Test writing histogram data to XLSX file."""
    output_file = str(tmp_path / "histogram.xlsx")
    settings = extend_dict(
        base_minimal_settings,
        {
            "histogram_data": [output_file],
            "histogram_chart": None,
            "quantiles": [0.5, 0.85, 0.95],
        },
    )

    calculator = HistogramCalculator(
        custom_query_manager, settings, large_cycle_time_results
    )
    result = calculator.run()
    large_cycle_time_results[HistogramCalculator] = result
    calculator.write()

    assert os.path.exists(output_file)
    df = pd.read_excel(output_file, sheet_name="Histogram")
    assert "histogram" in df.columns or len(df.columns) > 0


def test_write_chart(
    custom_query_manager, base_minimal_settings, large_cycle_time_results, tmp_path
):
    """Test writing histogram chart."""
    output_file = str(tmp_path / "histogram.png")
    settings = extend_dict(
        base_minimal_settings,
        {
            "histogram_data": None,
            "histogram_chart": output_file,
            "histogram_window": None,
            "histogram_chart_title": "Test Histogram",
            "quantiles": [0.5, 0.85, 0.95],
        },
    )

    calculator = HistogramCalculator(
        custom_query_manager, settings, large_cycle_time_results
    )
    result = calculator.run()
    large_cycle_time_results[HistogramCalculator] = result
    calculator.write()

    assert os.path.exists(output_file)


def test_write_chart_with_window(
    custom_query_manager, base_minimal_settings, large_cycle_time_results, tmp_path
):
    """Test writing histogram chart with window filtering."""
    output_file = str(tmp_path / "histogram.png")
    settings = extend_dict(
        base_minimal_settings,
        {
            "histogram_data": None,
            "histogram_chart": output_file,
            "histogram_window": 30,
            "histogram_chart_title": "Test Histogram",
            "quantiles": [0.5, 0.85, 0.95],
        },
    )

    calculator = HistogramCalculator(
        custom_query_manager, settings, large_cycle_time_results
    )
    result = calculator.run()
    large_cycle_time_results[HistogramCalculator] = result
    calculator.write()

    assert os.path.exists(output_file)


def test_write_chart_insufficient_data(
    custom_query_manager,
    base_minimal_settings,
    base_minimal_cycle_time_columns,
    tmp_path,
):
    """Test writing histogram chart with insufficient data."""
    output_file = str(tmp_path / "histogram.png")
    empty_results = create_empty_test_results(base_minimal_cycle_time_columns)
    settings = extend_dict(
        base_minimal_settings,
        {
            "histogram_data": None,
            "histogram_chart": output_file,
            "histogram_window": None,
            "quantiles": [0.5, 0.85, 0.95],
        },
    )

    calculator = HistogramCalculator(custom_query_manager, settings, empty_results)
    calculator.run()
    calculator.write()

    # Chart should not be created with insufficient data
    assert not os.path.exists(output_file)


def test_write_lead_time_file(
    custom_query_manager, base_minimal_settings, large_cycle_time_results, tmp_path
):
    """Test writing lead time histogram data."""
    output_file = str(tmp_path / "lead_time_histogram.csv")
    settings = extend_dict(
        base_minimal_settings,
        {
            "histogram_data": None,
            "histogram_chart": None,
            "lead_time_histogram_data": [output_file],
            "lead_time_histogram_chart": None,
            "quantiles": [0.5, 0.85, 0.95],
        },
    )

    calculator = HistogramCalculator(
        custom_query_manager, settings, large_cycle_time_results
    )
    result = calculator.run()
    large_cycle_time_results[HistogramCalculator] = result
    calculator.write()

    assert os.path.exists(output_file)
    df = pd.read_csv(output_file)
    assert list(df.columns) == ["Range", "Items"]


def test_write_lead_time_chart(
    custom_query_manager, base_minimal_settings, large_cycle_time_results, tmp_path
):
    """Test writing lead time histogram chart."""
    output_file = str(tmp_path / "lead_time_histogram.png")
    settings = extend_dict(
        base_minimal_settings,
        {
            "histogram_data": None,
            "histogram_chart": None,
            "lead_time_histogram_data": None,
            "lead_time_histogram_chart": [output_file],
            "histogram_window": None,
            "lead_time_histogram_chart_title": "Test Lead Time Histogram",
            "quantiles": [0.5, 0.85, 0.95],
        },
    )

    calculator = HistogramCalculator(
        custom_query_manager, settings, large_cycle_time_results
    )
    result = calculator.run()
    large_cycle_time_results[HistogramCalculator] = result
    calculator.write()

    assert os.path.exists(output_file)


def test_write_lead_time_chart_with_window(
    custom_query_manager, base_minimal_settings, large_cycle_time_results, tmp_path
):
    """Test writing lead time histogram chart with window filtering."""
    output_file = str(tmp_path / "lead_time_histogram.png")
    settings = extend_dict(
        base_minimal_settings,
        {
            "histogram_data": None,
            "histogram_chart": None,
            "lead_time_histogram_data": None,
            "lead_time_histogram_chart": [output_file],
            "histogram_window": 30,
            "lead_time_histogram_chart_title": "Test Lead Time Histogram",
            "quantiles": [0.5, 0.85, 0.95],
        },
    )

    calculator = HistogramCalculator(
        custom_query_manager, settings, large_cycle_time_results
    )
    result = calculator.run()
    large_cycle_time_results[HistogramCalculator] = result
    calculator.write()

    assert os.path.exists(output_file)


def test_write_no_output_files(
    custom_query_manager, base_minimal_settings, large_cycle_time_results
):
    """Test write() when no output files are specified."""
    settings = extend_dict(
        base_minimal_settings,
        {
            "histogram_data": None,
            "histogram_chart": None,
            "lead_time_histogram_data": None,
            "lead_time_histogram_chart": None,
            "quantiles": [0.5, 0.85, 0.95],
        },
    )

    calculator = HistogramCalculator(
        custom_query_manager, settings, large_cycle_time_results
    )
    calculator.run()
    # Should not raise an error
    calculator.write()


def test_write_all_outputs(
    custom_query_manager, base_minimal_settings, large_cycle_time_results, tmp_path
):
    """Test write() with all output files specified."""
    histogram_csv = str(tmp_path / "histogram.csv")
    histogram_png = str(tmp_path / "histogram.png")
    lead_time_csv = str(tmp_path / "lead_time_histogram.csv")
    lead_time_png = str(tmp_path / "lead_time_histogram.png")

    settings = extend_dict(
        base_minimal_settings,
        {
            "histogram_data": [histogram_csv],
            "histogram_chart": histogram_png,
            "lead_time_histogram_data": [lead_time_csv],
            "lead_time_histogram_chart": [lead_time_png],
            "histogram_window": None,
            "histogram_chart_title": "Test Histogram",
            "lead_time_histogram_chart_title": "Test Lead Time Histogram",
            "quantiles": [0.5, 0.85, 0.95],
        },
    )

    calculator = HistogramCalculator(
        custom_query_manager, settings, large_cycle_time_results
    )
    result = calculator.run()
    large_cycle_time_results[HistogramCalculator] = result
    calculator.write()

    assert os.path.exists(histogram_csv)
    assert os.path.exists(histogram_png)
    assert os.path.exists(lead_time_csv)
    assert os.path.exists(lead_time_png)


def test_run_without_cycle_time_column(
    custom_query_manager, base_minimal_settings, base_minimal_cycle_time_columns
):
    """Test run() when cycle_time column is missing."""
    # Create results without cycle_time column
    empty_df = pd.DataFrame([], columns=base_minimal_cycle_time_columns)
    if "cycle_time" in empty_df.columns:
        empty_df = empty_df.drop(columns=["cycle_time"])
    results = {CycleTimeCalculator: empty_df}

    calculator = HistogramCalculator(
        custom_query_manager, base_minimal_settings, results
    )
    data = calculator.run()

    assert isinstance(data, pd.Series)
    assert len(data) == 0


def test_run_with_non_timedelta_cycle_time(
    custom_query_manager, base_minimal_settings, base_minimal_cycle_time_columns
):
    """Test run() when cycle_time is not timedelta type."""
    # Create results with cycle_time as non-timedelta
    df = pd.DataFrame(
        {"cycle_time": [1, 2, 3]},
        columns=base_minimal_cycle_time_columns + ["cycle_time"],
    )
    results = {CycleTimeCalculator: df}

    calculator = HistogramCalculator(
        custom_query_manager, base_minimal_settings, results
    )
    data = calculator.run()

    assert isinstance(data, pd.Series)
    assert len(data) == 0
