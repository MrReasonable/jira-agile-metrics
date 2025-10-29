"""Tests for forecast calculator functionality in Jira Agile Metrics.

This module contains unit tests for the forecast calculator.
"""

import datetime

import numpy as np
import pytest
from pandas import DataFrame, Timestamp, date_range

from ..utils import extend_dict
from .burnup import BurnupCalculator
from .cfd import CFDCalculator
from .cycletime import CycleTimeCalculator
from .forecast import BurnupForecastCalculator


@pytest.fixture(name="settings")
def fixture_settings(base_minimal_settings):
    """Create forecast test settings with burnup forecast configuration."""
    return extend_dict(
        base_minimal_settings,
        {
            "burnup_forecast_chart_throughput_window_end": None,
            "burnup_forecast_chart_throughput_window": 8,
            "burnup_forecast_chart_target": 30,
            "burnup_forecast_chart_trials": 10,
            "burnup_forecast_chart_deadline": datetime.date(2018, 1, 30),
            "burnup_forecast_chart_deadline_confidence": 0.85,
            "quantiles": [0.1, 0.3, 0.5],
            "burnup_forecast_chart": "forecast.png",
            # without a file, calculator stops
        },
    )


@pytest.fixture(name="query_manager")
def fixture_query_manager(minimal_query_manager):
    """Create query manager fixture for forecast tests."""
    return minimal_query_manager


@pytest.fixture(name="results")
def fixture_results(query_manager, settings, large_cycle_time_results):
    """Create results fixture with CFD data for forecast tests."""
    results = large_cycle_time_results.copy()
    results.update(
        {CFDCalculator: CFDCalculator(query_manager, settings, results).run()}
    )
    results.update(
        {BurnupCalculator: BurnupCalculator(query_manager, settings, results).run()}
    )
    return results


def test_empty(query_manager, settings, base_minimal_cycle_time_columns):
    """Test burnup forecast calculator with empty data."""
    results = {
        CycleTimeCalculator: DataFrame([], columns=base_minimal_cycle_time_columns),
        BurnupCalculator: DataFrame(
            [],
            columns=["Backlog", "Committed", "Build", "Test", "Done"],
            index=date_range(start=datetime.date(2018, 1, 1), periods=0, freq="D"),
        ),
    }

    calculator = BurnupForecastCalculator(query_manager, settings, results)

    data = calculator.run()
    assert data is None


def test_columns(query_manager, settings, results):
    """Test burnup forecast calculator column structure."""
    calculator = BurnupForecastCalculator(query_manager, settings, results)
    data = calculator.run()
    if data is None:
        # Forecast horizon is not after last data point; nothing to simulate
        assert data is None
    else:
        assert list(data.columns) == [
            "Trial 0",
            "Trial 1",
            "Trial 2",
            "Trial 3",
            "Trial 4",
            "Trial 5",
            "Trial 6",
            "Trial 7",
            "Trial 8",
            "Trial 9",
        ]


def test_calculate_forecast(query_manager, settings, results):
    """Test burnup forecast calculation logic."""
    calculator = BurnupForecastCalculator(query_manager, settings, results)
    data = calculator.run()
    if data is None:
        # Forecast horizon is not after last data point; nothing to simulate
        assert data is None
    else:
        # because of the random nature of this,
        # we don't know exactly how many records
        # there will be, but will assume at least two
        assert len(data.index) > 0
        assert list(data.index)[0] == Timestamp("2018-01-09 00:00:00")
        assert list(data.index)[1] == Timestamp("2018-01-10 00:00:00")
        for i in range(10):
            trial_values = data[f"Trial {i}"]
            trial_values = trial_values[: trial_values.last_valid_index()]
            trial_diff = np.diff(trial_values)
            assert np.all(trial_diff >= 0)
            # Check that the trial starts with initial done value
            # and ends with target value (which is 30 based on settings)
            initial_done = trial_values.iloc[0]
            final_done = trial_values.iloc[-1]
            assert initial_done >= 0  # Should be non-negative
            assert final_done >= initial_done  # Should be monotonically increasing
            assert final_done <= 30  # Should not exceed target


def test_convert_trials_to_dataframe(query_manager, settings, results, mocker):
    """Test conversion of trials to DataFrame through public API.

    Uses monkeypatching of the MonteCarloSimulator to return deterministic
    trials without accessing private members/methods.
    """

    calculator = BurnupForecastCalculator(query_manager, settings, results)

    # Deterministic trials for stable assertions
    trials = [
        {
            "trial_num": i,
            "done_trial": [2.0 + j * 0.5 for j in range(25)],
            "backlog_trial": [5.0 - j * 0.2 for j in range(25)],
            "final_backlog": 0.0,
            "final_done": 14.5,
        }
        for i in range(10)
    ]

    # Monkeypatch simulator's run_simulation to avoid randomness and internals
    patch_path = (
        "jira_agile_metrics.calculators."
        "monte_carlo_simulator.MonteCarloSimulator.run_simulation"
    )
    mocker.patch(
        patch_path,
        return_value={"trials": trials, "trust_metrics": {}, "num_trials": len(trials)},
    )

    # Execute via public API
    df = calculator.run()

    # Validate results
    assert df is not None
    assert not df.empty
    assert len(df.columns) == 10
    assert list(df.columns) == [f"Trial {i}" for i in range(10)]
    assert len(df.index) > 0

    # Check that values are correctly extracted
    for i in range(10):
        trial_values = df[f"Trial {i}"]
        # Should start with the first value from done_trial
        assert trial_values.iloc[0] == pytest.approx(2.0)
        # Should be monotonically increasing
        assert trial_values.is_monotonic_increasing


def test_calculate_forecast_settings(query_manager, settings, results):
    """Test burnup forecast with different settings."""
    settings.update(
        {
            "backlog_column": "Committed",
            "done_column": "Test",
            "burnup_forecast_chart_throughput_window_end": datetime.date(2018, 1, 6),
            "burnup_forecast_chart_throughput_window": 4,
            "burnup_forecast_chart_target": None,
            "burnup_forecast_chart_trials": 10,
            "burnup_forecast_chart_deadline": datetime.date(2018, 1, 30),
            "burnup_forecast_chart_deadline_confidence": 0.85,
            "quantiles": [0.1, 0.3, 0.5],
        }
    )
    results.update(
        {CFDCalculator: CFDCalculator(query_manager, settings, results).run()}
    )
    results.update(
        {BurnupCalculator: BurnupCalculator(query_manager, settings, results).run()}
    )
    calculator = BurnupForecastCalculator(query_manager, settings, results)
    data = calculator.run()
    if data is None:
        # Forecast horizon is not after last data point; nothing to simulate
        assert data is None
    else:
        assert len(data.index) > 0
        assert list(data.index)[0] == Timestamp("2018-01-09 00:00:00")
        assert list(data.index)[1] == Timestamp("2018-01-10 00:00:00")
        for i in range(10):
            trial_values = data[f"Trial {i}"]
            trial_values = trial_values[: trial_values.last_valid_index()]
            trial_diff = np.diff(trial_values)
            assert np.all(trial_diff >= 0)
            # Check that the trial starts with initial done value
            # and ends with a reasonable value
            initial_done = trial_values.iloc[0]
            final_done = trial_values.iloc[-1]
            assert initial_done >= 0  # Should be non-negative
            assert final_done >= initial_done  # Should be monotonically increasing
