"""Functional tests for histogram calculator."""

from pathlib import Path

import pandas as pd

from jira_agile_metrics.calculator import run_calculators
from jira_agile_metrics.calculators.cycletime import CycleTimeCalculator
from jira_agile_metrics.calculators.histogram import HistogramCalculator
from jira_agile_metrics.tests.helpers.dataframe_utils import normalize_dataframe


def test_histogram_generates_expected_csv(
    query_manager, simple_cycle_settings, tmp_path
):
    """Test that histogram calculator generates CSV matching expected fixture."""
    settings, _ = simple_cycle_settings

    output_csv = tmp_path / "histogram.csv"
    settings = {
        **settings,
        "histogram_data": [str(output_csv)],
        "histogram_chart": None,
        "histogram_window": 0,
        "histogram_chart_title": None,
        "lead_time_histogram_data": None,
        "lead_time_histogram_chart": None,
        "quantiles": [0.5, 0.85, 0.95],
    }

    run_calculators([CycleTimeCalculator, HistogramCalculator], query_manager, settings)

    # Compare CSVs using shared DataFrame normalization
    expected_path = (
        Path(__file__).parent.parent / "fixtures" / "expected" / "histogram.csv"
    )

    expected_df = pd.read_csv(expected_path, encoding="utf-8")
    actual_df = pd.read_csv(output_csv, encoding="utf-8")

    expected_df = normalize_dataframe(expected_df)
    actual_df = normalize_dataframe(actual_df)

    pd.testing.assert_frame_equal(expected_df, actual_df, check_dtype=False)
