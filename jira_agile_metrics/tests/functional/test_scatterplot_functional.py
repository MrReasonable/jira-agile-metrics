"""Functional tests for scatterplot calculator."""

import os

import pandas as pd

from jira_agile_metrics.calculator import run_calculators
from jira_agile_metrics.calculators.cycletime import CycleTimeCalculator
from jira_agile_metrics.calculators.scatterplot import ScatterplotCalculator


def _parse_days(value):
    """Parse 'X days' format to integer, returning original value on failure."""
    if isinstance(value, str) and "days" in value:
        try:
            return int(str(value).split()[0])
        except (ValueError, IndexError):
            return value
    return value


def test_scatterplot_generates_expected_csv(
    query_manager, simple_cycle_settings, tmp_path
):
    """Test that scatterplot calculator generates CSV matching expected fixture."""
    settings, _ = simple_cycle_settings

    output_csv = tmp_path / "scatterplot.csv"
    settings = {
        **settings,
        "scatterplot_data": [str(output_csv)],
        "scatterplot_chart": None,
        "scatterplot_window": 0,
        "scatterplot_chart_title": None,
        "quantiles": [0.5, 0.85, 0.95],
        "date_format": "%Y-%m-%d",
    }

    run_calculators(
        [CycleTimeCalculator, ScatterplotCalculator], query_manager, settings
    )

    # Read both CSVs and compare as DataFrames
    # (handles date parsing and ordering)
    actual_df = pd.read_csv(output_csv, parse_dates=["completed_date"])
    expected_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "fixtures",
        "expected",
        "scatterplot.csv",
    )
    expected_df = pd.read_csv(expected_path, parse_dates=["completed_date"])

    # Sort by completed_date for consistent comparison
    actual_df = actual_df.sort_values("completed_date").reset_index(drop=True)
    expected_df = expected_df.sort_values("completed_date").reset_index(drop=True)

    # Compare - use check_dtype=False since lead_time might be string vs timedelta
    pd.testing.assert_frame_equal(
        actual_df.drop(columns=["lead_time"]),
        expected_df.drop(columns=["lead_time"]),
        check_dtype=False,
    )

    # Compare lead_time separately
    # (handles "19 days" vs timedelta representation)
    actual_lead_times = actual_df["lead_time"].apply(_parse_days)
    expected_lead_times = expected_df["lead_time"].apply(_parse_days)
    pd.testing.assert_series_equal(
        actual_lead_times, expected_lead_times, check_dtype=False
    )
