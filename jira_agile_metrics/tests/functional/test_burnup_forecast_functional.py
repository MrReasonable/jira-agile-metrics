"""Functional tests for burnup forecast calculator."""

import random

import pandas as pd

from jira_agile_metrics.calculator import run_calculators
from jira_agile_metrics.calculators.burnup import BurnupCalculator
from jira_agile_metrics.calculators.cfd import CFDCalculator
from jira_agile_metrics.calculators.cycletime import CycleTimeCalculator
from jira_agile_metrics.calculators.forecast import BurnupForecastCalculator
from jira_agile_metrics.tests.functional.conftest import get_burnup_base_settings


def test_burnup_forecast_generates_expected_data_structure(
    query_manager, simple_cycle_settings
):
    """Test that burnup forecast calculator generates expected data structure."""
    settings, _ = simple_cycle_settings

    # Set fixed random seed for reproducible results
    random.seed(42)

    # Configure forecast settings with minimal trials for faster execution
    settings = get_burnup_base_settings(settings)
    settings.update(
        {
            # Dummy path to allow validation to pass
            "burnup_forecast_chart": "/dev/null",
            "burnup_forecast_chart_trials": 20,  # More trials for better validation
            "burnup_forecast_chart_throughput_window": 30,
            "burnup_forecast_chart_throughput_window_end": None,
            "burnup_forecast_chart_target": None,
            "burnup_forecast_chart_deadline": None,
            "burnup_forecast_chart_deadline_confidence": 0.85,
        }
    )

    results = run_calculators(
        [
            CycleTimeCalculator,
            CFDCalculator,
            BurnupCalculator,
            BurnupForecastCalculator,
        ],
        query_manager,
        settings,
    )

    forecast_result = results[BurnupForecastCalculator]

    # Verify forecast produces a DataFrame (or None if insufficient data)
    # With our minimal fixture, we should get a DataFrame with trial columns
    assert forecast_result is not None, (
        "Forecast should produce a result with valid data"
    )
    assert isinstance(forecast_result, pd.DataFrame), (
        "Forecast result should be a DataFrame"
    )

    # Verify structure
    assert len(forecast_result.columns) > 0, "Forecast should have trial columns"
    assert isinstance(forecast_result.index, pd.DatetimeIndex), (
        "Forecast index should be datetime"
    )

    # Verify forecast extends beyond the last burnup date
    # Last date from fixtures is 2021-01-25, forecast should extend forward
    assert forecast_result.index.min() >= pd.Timestamp("2021-01-25"), (
        "Forecast should start from or after last burnup date"
    )

    # Verify all trial columns have numeric data
    for col in forecast_result.columns:
        assert pd.api.types.is_numeric_dtype(forecast_result[col]), (
            f"Trial column {col} should contain numeric data"
        )

    # Verify forecast has reasonable progression
    # (done values should be non-decreasing or stable)
    # Check a few trial columns for basic sanity
    sample_trials = list(forecast_result.columns)[:3]  # Check first 3 trials
    for trial_col in sample_trials:
        trial_values = forecast_result[trial_col].dropna()
        if len(trial_values) > 1:
            # Values should be non-negative and generally increasing
            # (or at least not dramatically decreasing)
            assert (trial_values >= 0).all(), (
                f"Trial {trial_col} should have non-negative values"
            )
            # Initial value should match completed items from fixtures (2 items done)
            assert trial_values.iloc[0] >= 2, (
                f"Trial {trial_col} should start at least at 2 (completed items)"
            )
