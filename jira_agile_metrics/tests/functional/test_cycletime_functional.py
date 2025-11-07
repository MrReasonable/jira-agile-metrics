"""Functional tests for cycle time calculator."""

from pathlib import Path

import pandas as pd

from jira_agile_metrics.calculator import run_calculators
from jira_agile_metrics.calculators.cycletime import CycleTimeCalculator
from jira_agile_metrics.tests.helpers.dataframe_utils import normalize_dataframe


def test_cycletime_functional_generates_expected_csv(
    query_manager, simple_cycle_settings
):
    """Test that cycle time calculator generates CSV matching expected fixture."""
    settings, output_csv = simple_cycle_settings

    # Include impediment_flags to match expected CSV generation
    settings = {**settings, "impediment_flags": ["Impediment", "Awaiting input"]}

    run_calculators([CycleTimeCalculator], query_manager, settings)

    expected_path = (
        Path(__file__).resolve().parents[1] / "fixtures" / "expected" / "cycletime.csv"
    )
    expected_df = pd.read_csv(expected_path)
    actual_df = pd.read_csv(output_csv)

    # Normalize DataFrames for stable comparison
    expected_df = normalize_dataframe(expected_df)
    actual_df = normalize_dataframe(actual_df)

    pd.testing.assert_frame_equal(expected_df, actual_df, check_dtype=False)
