"""Functional tests for impediments calculator."""

from pathlib import Path

import pandas as pd

from jira_agile_metrics.calculator import run_calculators
from jira_agile_metrics.calculators.cycletime import CycleTimeCalculator
from jira_agile_metrics.calculators.impediments import ImpedimentsCalculator
from jira_agile_metrics.tests.helpers.dataframe_utils import normalize_dataframe


def test_impediments_generates_expected_csv(
    query_manager, simple_cycle_settings, tmp_path
):
    """Test that impediments calculator generates CSV matching expected fixture."""
    settings, _ = simple_cycle_settings

    output_csv = tmp_path / "impediments.csv"
    settings = {
        **settings,
        "impediments_data": [str(output_csv)],
        "impediments_chart": None,
        "impediments_days_chart": None,
        "impediments_status_chart": None,
        "impediments_status_days_chart": None,
        "impediments_window": 0,
        "impediment_flags": ["Impediment", "Awaiting input"],
    }

    run_calculators(
        [CycleTimeCalculator, ImpedimentsCalculator], query_manager, settings
    )

    # Read both CSVs and compare
    expected_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "expected"
        / "impediments.csv"
    )

    # If no impediments data exists, the file might not be created
    if not output_csv.exists():
        # Check if expected file also doesn't exist or is empty
        if not expected_path.exists():
            # Both missing - this is acceptable if no impediments in test data
            return
        expected_df = pd.read_csv(expected_path)
        if expected_df.empty:
            # Both empty - acceptable
            return
        # Actual missing but expected exists - this is a failure
        raise AssertionError(
            f"Expected impediments CSV but file was not created: {output_csv}"
        )

    # Guard: check that expected_path exists before reading
    if not expected_path.exists():
        raise AssertionError(
            f"Expected fixture file is missing: {expected_path}. "
            f"Output CSV was created at {output_csv}, but expected fixture "
            f"is not present."
        )

    expected_df = pd.read_csv(expected_path, parse_dates=["start", "end"])
    actual_df = pd.read_csv(output_csv, parse_dates=["start", "end"])

    # Normalize DataFrames for stable comparisons (whitespace, precision, types)
    expected_df = normalize_dataframe(expected_df)
    actual_df = normalize_dataframe(actual_df)

    # Sort by key and start for consistent comparison
    expected_df = expected_df.sort_values(["key", "start"]).reset_index(drop=True)
    actual_df = actual_df.sort_values(["key", "start"]).reset_index(drop=True)

    # Compare DataFrames - they should match exactly
    pd.testing.assert_frame_equal(
        expected_df, actual_df, check_dtype=False, check_names=True
    )
