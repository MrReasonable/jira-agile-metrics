"""Functional tests for throughput calculator."""

from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from jira_agile_metrics.calculator import run_calculators
from jira_agile_metrics.calculators.cycletime import CycleTimeCalculator
from jira_agile_metrics.calculators.throughput import ThroughputCalculator
from jira_agile_metrics.tests.helpers.dataframe_utils import normalize_dataframe


def test_throughput_generates_expected_csv(
    query_manager, simple_cycle_settings, tmp_path
):
    """Test that throughput calculator generates CSV matching expected fixture."""
    settings, _ = simple_cycle_settings

    output_csv = tmp_path / "throughput.csv"
    settings = {
        **settings,
        "throughput_data": [str(output_csv)],
        "throughput_chart": None,
        "throughput_frequency": "D",
        "throughput_window": 0,
        "throughput_chart_title": None,
        "date_format": "%Y-%m-%d",
    }

    run_calculators(
        [CycleTimeCalculator, ThroughputCalculator], query_manager, settings
    )

    # Compare CSVs using DataFrame equality (robust to formatting/whitespace)
    expected_path = (
        Path(__file__).resolve().parents[1] / "fixtures" / "expected" / "throughput.csv"
    )

    expected_df = pd.read_csv(expected_path, encoding="utf-8")
    actual_df = pd.read_csv(output_csv, encoding="utf-8")

    expected_df = normalize_dataframe(expected_df)
    actual_df = normalize_dataframe(actual_df)

    assert_frame_equal(actual_df, expected_df, check_dtype=False)
