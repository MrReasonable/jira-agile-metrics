"""Functional tests for cycle time calculator."""

from pathlib import Path

import pandas as pd

from jira_agile_metrics.calculator import run_calculators
from jira_agile_metrics.calculators.cycletime import CycleTimeCalculator


def test_cycletime_functional_generates_expected_csv(
    query_manager, simple_cycle_settings
):
    """Test that cycle time calculator generates CSV matching expected fixture."""
    settings, output_csv = simple_cycle_settings

    run_calculators([CycleTimeCalculator], query_manager, settings)

    expected_path = (
        Path(__file__).resolve().parents[1] / "fixtures" / "expected" / "cycletime.csv"
    )
    expected_df = pd.read_csv(expected_path)
    actual_df = pd.read_csv(output_csv)

    # Normalize to consistent column order
    expected_df = expected_df.sort_index(axis=1)
    actual_df = actual_df.sort_index(axis=1)

    # Reset index if necessary
    expected_df = expected_df.reset_index(drop=True)
    actual_df = actual_df.reset_index(drop=True)

    pd.testing.assert_frame_equal(expected_df, actual_df, check_dtype=False)
