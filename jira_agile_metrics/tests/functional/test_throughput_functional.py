"""Functional tests for throughput calculator."""

from pathlib import Path

from jira_agile_metrics.calculator import run_calculators
from jira_agile_metrics.calculators.cycletime import CycleTimeCalculator
from jira_agile_metrics.calculators.throughput import ThroughputCalculator


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

    # Compare full CSV content against expected fixture
    expected_path = (
        Path(__file__).resolve().parents[1] / "fixtures" / "expected" / "throughput.csv"
    )
    expected_text = expected_path.read_text(encoding="utf-8").strip()
    actual_text = Path(output_csv).read_text(encoding="utf-8").strip()
    assert actual_text == expected_text
