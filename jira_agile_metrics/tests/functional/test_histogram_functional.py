"""Functional tests for histogram calculator."""

from pathlib import Path

from jira_agile_metrics.calculator import run_calculators
from jira_agile_metrics.calculators.cycletime import CycleTimeCalculator
from jira_agile_metrics.calculators.histogram import HistogramCalculator


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

    # Compare full CSV content against expected fixture
    expected_path = (
        Path(__file__).parent.parent / "fixtures" / "expected" / "histogram.csv"
    )
    with expected_path.open(encoding="utf-8") as f:
        expected_text = f.read().strip()
    with Path(output_csv).open(encoding="utf-8") as f:
        actual_text = f.read().strip()
    assert actual_text == expected_text
