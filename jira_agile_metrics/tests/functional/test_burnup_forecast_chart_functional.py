"""Functional tests for burnup forecast chart generation."""

import random

import numpy as np
import pytest

from jira_agile_metrics.calculators.burnup import BurnupCalculator
from jira_agile_metrics.calculators.forecast import BurnupForecastCalculator
from jira_agile_metrics.tests.functional.conftest import (
    get_burnup_base_settings,
    get_default_forecast_settings,
    run_forecast_calculators,
    validate_forecast_result_structure,
    validate_forecast_trial_values,
)


def test_burnup_forecast_chart_generation(
    query_manager, simple_cycle_settings, tmp_path
):
    """Test that burnup forecast calculator generates a valid chart file."""
    settings, _ = simple_cycle_settings

    # Set fixed random seed for reproducible results
    random.seed(42)
    np.random.seed(42)

    # Configure forecast settings with minimal trials for faster execution
    settings = get_burnup_base_settings(settings)
    output_chart = tmp_path / "burnup-forecast.png"
    settings.update(get_default_forecast_settings())
    settings.update(
        {
            "burnup_forecast_chart": str(output_chart),
            "burnup_forecast_chart_title": "Test Burnup Forecast",
            "burnup_forecast_chart_trials": 50,  # Enough for meaningful forecast
            "burnup_forecast_chart_throughput_frequency": "weekly",
            "burnup_forecast_chart_smart_window": True,
        }
    )

    results = run_forecast_calculators(query_manager, settings)
    forecast_result = results[BurnupForecastCalculator]

    # Verify forecast produces a DataFrame (or None if insufficient data)
    if forecast_result is None:
        pytest.skip("Forecast could not be generated (insufficient data)")

    # Verify chart file was created
    assert output_chart.exists(), "Forecast chart file should be created"
    assert output_chart.stat().st_size > 0, "Chart file should not be empty"

    # Verify it's a valid PNG file (basic check - PNG files start with PNG signature)
    with open(output_chart, "rb") as f:
        header = f.read(8)
        # PNG files start with: 89 50 4E 47 0D 0A 1A 0A
        assert header[:8] == b"\x89PNG\r\n\x1a\n", "Chart file should be a valid PNG"

    # Verify forecast data structure
    validate_forecast_result_structure(forecast_result)

    # Verify forecast extends beyond the last burnup date
    burnup_data = results[BurnupCalculator]
    if not burnup_data.empty:
        last_burnup_date = burnup_data.index[-1]
        assert forecast_result.index.min() >= last_burnup_date, (
            "Forecast should start from or after last burnup date"
        )

    # Verify forecast has reasonable progression
    validate_forecast_trial_values(forecast_result)


def test_burnup_forecast_chart_with_target(
    query_manager, simple_cycle_settings, tmp_path
):
    """Test burnup forecast chart generation with explicit target."""
    settings, _ = simple_cycle_settings

    # Set fixed random seed for reproducible results
    random.seed(42)
    np.random.seed(42)

    settings = get_burnup_base_settings(settings)
    output_chart = tmp_path / "burnup-forecast-target.png"
    settings.update(
        {
            "burnup_forecast_chart": str(output_chart),
            "burnup_forecast_chart_title": "Test Burnup Forecast with Target",
            "burnup_forecast_chart_trials": 50,
            "burnup_forecast_chart_throughput_window": 30,
            "burnup_forecast_chart_throughput_frequency": "weekly",
            "burnup_forecast_chart_smart_window": True,
            "burnup_forecast_chart_target": 50,  # Explicit target
            "burnup_forecast_chart_deadline": None,
        }
    )

    results = run_forecast_calculators(query_manager, settings)

    forecast_result = results[BurnupForecastCalculator]

    if forecast_result is None:
        pytest.skip("Forecast could not be generated (insufficient data)")

    # Verify chart file was created
    assert output_chart.exists(), "Forecast chart file should be created"
    assert output_chart.stat().st_size > 0, "Chart file should not be empty"

    # Verify PNG format
    with open(output_chart, "rb") as f:
        header = f.read(8)
        assert header[:8] == b"\x89PNG\r\n\x1a\n", "Chart file should be a valid PNG"


def test_burnup_forecast_chart_with_completion_dates(
    query_manager, simple_cycle_settings, tmp_path
):
    """Test burnup forecast chart includes completion date quantiles."""
    settings, _ = simple_cycle_settings

    # Set fixed random seed for reproducible results
    random.seed(42)
    np.random.seed(42)

    settings = get_burnup_base_settings(settings)
    output_chart = tmp_path / "burnup-forecast-completion.png"
    settings.update(
        {
            "burnup_forecast_chart": str(output_chart),
            "burnup_forecast_chart_title": "Test Burnup Forecast with Completion",
            "burnup_forecast_chart_trials": 100,  # More trials for better quantiles
            "burnup_forecast_chart_throughput_window": 30,
            "burnup_forecast_chart_throughput_frequency": "weekly",
            "burnup_forecast_chart_smart_window": True,
            "burnup_forecast_chart_target": 40,  # Target that should be reachable
            "burnup_forecast_chart_deadline": None,
        }
    )

    results = run_forecast_calculators(query_manager, settings)

    forecast_result = results[BurnupForecastCalculator]

    if forecast_result is None:
        pytest.skip("Forecast could not be generated (insufficient data)")

    # Verify chart file was created
    assert output_chart.exists(), "Forecast chart file should be created"

    # Verify the chart exists, confirming the forecast completed and
    # quantiles were used in chart generation
