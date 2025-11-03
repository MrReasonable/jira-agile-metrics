"""Functional tests for scatterplot calculator."""

from pathlib import Path

import pandas as pd

from jira_agile_metrics.calculator import run_calculators
from jira_agile_metrics.calculators.cycletime import CycleTimeCalculator
from jira_agile_metrics.calculators.scatterplot import ScatterplotCalculator


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
    expected_path = (
        Path(__file__).parent.parent / "fixtures" / "expected" / "scatterplot.csv"
    )
    expected_df = pd.read_csv(expected_path, parse_dates=["completed_date"])

    # Sort by completed_date for consistent comparison
    actual_df = actual_df.sort_values("completed_date").reset_index(drop=True)
    expected_df = expected_df.sort_values("completed_date").reset_index(drop=True)

    # Compare entire DataFrame - lead_time is now always an integer
    pd.testing.assert_frame_equal(actual_df, expected_df, check_dtype=False)
