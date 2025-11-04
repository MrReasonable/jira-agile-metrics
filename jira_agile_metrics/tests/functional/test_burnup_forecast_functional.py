"""Functional tests for burnup forecast calculator."""

import random

import numpy as np
import pandas as pd

from jira_agile_metrics.calculators.forecast import BurnupForecastCalculator
from jira_agile_metrics.tests.functional.conftest import (
    get_burnup_base_settings,
    get_default_forecast_settings,
    run_forecast_calculators,
    validate_forecast_result_structure,
    validate_forecast_trial_values,
)


def test_burnup_forecast_generates_expected_data_structure(
    query_manager, simple_cycle_settings
):
    """Test that burnup forecast calculator generates expected data structure."""
    settings, _ = simple_cycle_settings

    # Set fixed random seed for reproducible results
    random.seed(42)
    np.random.seed(42)

    # Configure forecast settings with minimal trials for faster execution
    settings = get_burnup_base_settings(settings)
    settings.update(get_default_forecast_settings())
    settings.update(
        {
            # Dummy path to allow validation to pass
            "burnup_forecast_chart": "/dev/null",
            "burnup_forecast_chart_trials": 20,  # More trials for better validation
        }
    )

    results = run_forecast_calculators(query_manager, settings)
    forecast_result = results[BurnupForecastCalculator]

    # Verify forecast produces a DataFrame (or None if insufficient data)
    # With our minimal fixture, we should get a DataFrame with trial columns
    assert (
        forecast_result is not None
    ), "Forecast should produce a result with valid data"

    # Verify structure
    validate_forecast_result_structure(forecast_result)

    # Verify forecast extends beyond the last burnup date
    # Last date from fixtures is 2021-01-25, forecast should extend forward
    assert forecast_result.index.min() >= pd.Timestamp(
        "2021-01-25"
    ), "Forecast should start from or after last burnup date"

    # Verify all trial columns have numeric data
    for col in forecast_result.columns:
        assert pd.api.types.is_numeric_dtype(
            forecast_result[col]
        ), f"Trial column {col} should contain numeric data"

    # Verify forecast has reasonable progression
    validate_forecast_trial_values(forecast_result)

    # Additional check: initial completed-items count should be >= 2
    # Fixture note: `jira_client` loads `tests/fixtures/jira/search_issues.json`
    # where the initial completed-items count is 2
    sample_trials = list(forecast_result.columns)[:3]
    for trial_col in sample_trials:
        trial_values = forecast_result[trial_col].dropna()
        if len(trial_values) > 0:
            assert (
                trial_values.iloc[0] >= 2
            ), f"Trial {trial_col} should start at least at 2 (completed items)"
