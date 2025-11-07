"""Functional tests for percentiles calculator."""

from pathlib import Path

import pandas as pd

from jira_agile_metrics.calculator import run_calculators
from jira_agile_metrics.calculators.cycletime import CycleTimeCalculator
from jira_agile_metrics.calculators.percentiles import PercentilesCalculator


def test_percentiles_generates_expected_csv(
    query_manager, simple_cycle_settings, tmp_path
):
    """Test that percentiles calculator generates CSV matching expected fixture."""
    settings, _ = simple_cycle_settings

    output_csv = tmp_path / "percentiles.csv"
    settings = {
        **settings,
        "percentiles_data": [str(output_csv)],
        "quantiles": [0.5, 0.85, 0.95],
    }

    run_calculators(
        [CycleTimeCalculator, PercentilesCalculator], query_manager, settings
    )

    # Read both CSVs and compare
    expected_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "expected"
        / "percentiles.csv"
    )
    expected_df = pd.read_csv(expected_path)
    actual_df = pd.read_csv(output_csv)

    # Compare DataFrames - they should match exactly
    pd.testing.assert_frame_equal(
        actual_df, expected_df, check_dtype=False, check_names=True
    )
