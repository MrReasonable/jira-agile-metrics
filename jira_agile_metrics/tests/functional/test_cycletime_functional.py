"""Functional tests for cycle time calculator."""

import os

from jira_agile_metrics.calculator import run_calculators
from jira_agile_metrics.calculators.cycletime import CycleTimeCalculator


def test_cycletime_functional_generates_expected_csv(
    query_manager, simple_cycle_settings
):
    """Test that cycle time calculator generates CSV matching expected fixture."""
    settings, output_csv = simple_cycle_settings

    run_calculators([CycleTimeCalculator], query_manager, settings)

    expected_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "fixtures",
        "expected",
        "cycletime.csv",
    )
    with open(expected_path, "r", encoding="utf-8") as f:
        expected_text = f.read().strip()
    with open(output_csv, "r", encoding="utf-8") as f:
        actual_text = f.read().strip()
    assert actual_text == expected_text
