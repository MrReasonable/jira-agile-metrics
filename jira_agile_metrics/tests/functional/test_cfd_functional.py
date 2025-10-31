"""Functional tests for CFD calculator."""

from pathlib import Path

import pandas as pd

from jira_agile_metrics.calculator import run_calculators
from jira_agile_metrics.calculators.cfd import CFDCalculator
from jira_agile_metrics.calculators.cycletime import CycleTimeCalculator


def test_cfd_generates_expected_csv(query_manager, simple_cycle_settings, tmp_path):
    """Test that CFD calculator generates CSV matching expected fixture."""
    settings, _ = simple_cycle_settings

    # Add CFD outputs to settings
    output_csv = tmp_path / "cfd.csv"
    settings = {
        **settings,
        "cfd_data": [str(output_csv)],
        "cfd_chart": None,
        "cfd_window": 0,
        "backlog_column": "Backlog",
        "cfd_chart_title": None,
    }

    run_calculators([CycleTimeCalculator, CFDCalculator], query_manager, settings)

    # Read both actual and expected, comparing as DataFrames
    # for proper floating point handling
    actual_df = pd.read_csv(output_csv, index_col=0, parse_dates=True)
    expected_path = (
        Path(__file__).resolve().parents[1] / "fixtures" / "expected" / "cfd.csv"
    )
    expected_df = pd.read_csv(expected_path, index_col=0, parse_dates=True)

    # Compare DataFrames - they should be identical
    pd.testing.assert_frame_equal(actual_df, expected_df, check_dtype=False)
