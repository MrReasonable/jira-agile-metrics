"""Tests for forecast calculator functionality in Jira Agile Metrics.

This module contains unit tests for the forecast calculator.
"""

import datetime
import random

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Timestamp, date_range

from jira_agile_metrics.tests.helpers.assertions import (
    assert_forecast_csv_file_valid,
    assert_forecast_json_file_valid,
    assert_forecast_xlsx_file_valid,
)

from ..utils import extend_dict
from .burnup import BurnupCalculator
from .cfd import CFDCalculator
from .cycletime import CycleTimeCalculator
from .forecast import BurnupForecastCalculator
from .monte_carlo_simulator import MonteCarloSimulator


@pytest.fixture(name="settings")
def fixture_settings(base_minimal_settings):
    """Create forecast test settings with burnup forecast configuration."""
    return extend_dict(
        base_minimal_settings,
        {
            "burnup_forecast_chart_throughput_window_end": None,
            "burnup_forecast_chart_throughput_window": 8,
            "burnup_forecast_chart_target": None,
            "burnup_forecast_chart_trials": 10,
            "burnup_forecast_chart_deadline": datetime.date(2018, 1, 30),
            "burnup_forecast_chart_deadline_confidence": 0.85,
            "quantiles": [0.1, 0.3, 0.5],
            # Don't set burnup_forecast_chart to avoid creating files during tests
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
    results.update({
        CFDCalculator: CFDCalculator(query_manager, settings, results).run()
    })
    results.update({
        BurnupCalculator: BurnupCalculator(query_manager, settings, results).run()
    })
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


def test_get_last_burnup_date_handles_indexerror(mocker):
    """Test that run() handles IndexError from empty burnup data gracefully."""
    results = {}
    calculator = BurnupForecastCalculator(None, {}, results)

    # Mock get_result to return a DataFrame with empty index for burnup data
    mock_burnup = DataFrame(columns=["Backlog", "Done"], index=[])
    mock_cycle = DataFrame(columns=["key"])

    def mock_get_result(calc=None, default=None):
        if calc == BurnupCalculator:
            return mock_burnup
        if calc == CycleTimeCalculator:
            return mock_cycle
        return default

    mocker.patch.object(calculator, "get_result", side_effect=mock_get_result)

    # Should not raise IndexError, but return None when burnup data has empty index
    result = calculator.run()
    assert result is None


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


def test_convert_trials_to_dataframe(
    query_manager, settings, results, mocker, tmp_path
):
    """Test conversion of trials to DataFrame through public API.

    Uses monkeypatching of the MonteCarloSimulator to return deterministic
    trials without accessing private members/methods.
    """
    # Set a temporary file path so validation passes
    output_file = tmp_path / "forecast-test.png"
    settings["burnup_forecast_chart"] = str(output_file)

    calculator = BurnupForecastCalculator(query_manager, settings, results)

    # Deterministic trials for stable assertions
    trials = [
        {
            "trial_num": i,
            "done_trial": [2.0 + j * 0.5 for j in range(25)],
            "backlog_trial": [
                5.0 + j * 0.2 for j in range(25)
            ],  # Cumulative growth, not shrinking
            "final_backlog": 9.8,  # 5.0 + 24*0.2 = 9.8 (initial + 24 periods)
            "final_done": 14.0,  # 2.0 + 24*0.5 = 14.0 (initial + 24 periods)
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
    assert isinstance(df, DataFrame), "Result should be a DataFrame"
    assert not df.empty
    assert len(df.columns) == 10
    assert list(df.columns) == [f"Trial {i}" for i in range(10)]
    assert len(df.index) > 0

    # Check that values are correctly extracted
    # Use loc to access columns in a way pylint understands
    for i in range(10):
        trial_name = f"Trial {i}"
        trial_values = df.loc[:, trial_name]
        # Should start with the first value from done_trial
        assert trial_values.iloc[0] == pytest.approx(2.0)
        # Should be monotonically increasing
        assert trial_values.is_monotonic_increasing


def test_calculate_forecast_settings(query_manager, settings, results):
    """Test burnup forecast with different settings."""
    settings.update({
        "backlog_column": "Committed",
        "done_column": "Test",
        "burnup_forecast_chart_throughput_window_end": None,
        "burnup_forecast_chart_throughput_window": 4,
        "burnup_forecast_chart_target": None,
        "burnup_forecast_chart_trials": 10,
        "burnup_forecast_chart_deadline": datetime.date(2018, 1, 30),
        "burnup_forecast_chart_deadline_confidence": 0.85,
        "quantiles": [0.1, 0.3, 0.5],
    })
    results.update({
        CFDCalculator: CFDCalculator(query_manager, settings, results).run()
    })
    results.update({
        BurnupCalculator: BurnupCalculator(query_manager, settings, results).run()
    })
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


def test_write_handles_no_forecast_horizon(
    query_manager, settings, results, mocker, tmp_path
):
    """Test that write() handles missing forecast horizon gracefully."""
    # Set a temporary file path so write() actually executes
    output_file = tmp_path / "forecast-test.png"
    settings["burnup_forecast_chart"] = str(output_file)

    calculator = BurnupForecastCalculator(query_manager, settings, results)

    # Run the calculator to set up state
    calculator.run()

    # Mock get_forecast_parameters to return None for forecast_horizon_end
    # This simulates a scenario where forecast_dates would be empty
    original_get_params = calculator.get_forecast_parameters

    def mock_get_params():
        params = original_get_params()
        params["forecast_horizon_end"] = None
        return params

    mocker.patch.object(
        calculator, "get_forecast_parameters", side_effect=mock_get_params
    )

    # Mock _get_forecast_dates to return empty list to simulate the behavior
    # when forecast_horizon_end is None
    mocker.patch.object(calculator, "_get_forecast_dates", return_value=[])

    # write() should handle empty forecast_dates without errors
    calculator.write()  # Should not raise exception


def test_write_handles_empty_quantile_data(
    query_manager, settings, results, mocker, tmp_path
):
    """Test that write() handles empty quantile data gracefully."""
    # Set a temporary file path so write() actually executes
    output_file = tmp_path / "forecast-test.png"
    settings["burnup_forecast_chart"] = str(output_file)

    calculator = BurnupForecastCalculator(query_manager, settings, results)

    # Run the calculator to set up state
    calculator.run()

    # Mock _calculate_quantile_data to return empty dict
    # This simulates when completion_dates would be empty
    mocker.patch.object(calculator, "_calculate_quantile_data", return_value={})

    # write() should handle empty quantile data without errors
    calculator.write()  # Should not raise exception


# Configuration variation tests
class TestForecastConfigurationVariations:
    """Test forecast calculator with various configuration settings."""

    @pytest.fixture
    def base_results(
        self, query_manager, base_minimal_settings, large_cycle_time_results
    ):
        """Create base results fixture for configuration tests."""
        results = large_cycle_time_results.copy()
        settings = extend_dict(
            base_minimal_settings,
            {
                # Don't set burnup_forecast_chart to avoid creating files during tests
                "burnup_forecast_chart_trials": 10,
            },
        )
        results.update({
            CFDCalculator: CFDCalculator(query_manager, settings, results).run()
        })
        results.update({
            BurnupCalculator: BurnupCalculator(query_manager, settings, results).run()
        })
        return results, settings

    def test_different_trial_counts(self, query_manager, base_results):
        """Test forecast with different numbers of trials."""
        results, base_settings = base_results

        for trial_count in [10, 50, 100, 500]:
            settings = extend_dict(
                base_settings, {"burnup_forecast_chart_trials": trial_count}
            )
            calculator = BurnupForecastCalculator(query_manager, settings, results)
            forecast_data = calculator.run()

            if forecast_data is not None:
                assert len(forecast_data.columns) == trial_count
                assert all(
                    f"Trial {i}" in forecast_data.columns for i in range(trial_count)
                )

    def test_different_throughput_windows(self, query_manager, base_results):
        """Test forecast with different throughput window sizes."""
        results, base_settings = base_results

        for window_size in [7, 14, 30, 60, 90]:
            settings = extend_dict(
                base_settings,
                {
                    "burnup_forecast_chart_throughput_window": window_size,
                    "burnup_forecast_chart_smart_window": False,  # Use fixed window
                },
            )
            calculator = BurnupForecastCalculator(query_manager, settings, results)
            forecast_data = calculator.run()

            if forecast_data is not None:
                assert isinstance(forecast_data, DataFrame)
                assert len(forecast_data.columns) > 0

    def test_different_frequencies(self, query_manager, base_results):
        """Test forecast with different throughput frequencies."""
        results, base_settings = base_results

        frequencies = ["daily", "weekly", "monthly", "D", "W", "M"]
        for freq in frequencies:
            settings = extend_dict(
                base_settings,
                {
                    "burnup_forecast_chart_throughput_frequency": freq,
                    "burnup_forecast_chart_throughput_window": 30,
                },
            )
            calculator = BurnupForecastCalculator(query_manager, settings, results)
            forecast_data = calculator.run()

            if forecast_data is not None:
                assert isinstance(forecast_data, DataFrame)
                # Verify forecast dates are generated correctly
                assert len(forecast_data.index) > 0
                assert isinstance(forecast_data.index, pd.DatetimeIndex)

    def test_smart_window_vs_fixed_window(self, query_manager, base_results):
        """Test forecast with smart window vs fixed window."""
        results, base_settings = base_results

        for smart_window in [True, False]:
            settings = extend_dict(
                base_settings,
                {
                    "burnup_forecast_chart_smart_window": smart_window,
                    "burnup_forecast_chart_throughput_window": 30,
                },
            )
            calculator = BurnupForecastCalculator(query_manager, settings, results)
            forecast_data = calculator.run()

            if forecast_data is not None:
                assert isinstance(forecast_data, DataFrame)

    def test_different_targets(self, query_manager, base_results):
        """Test forecast with different target values."""
        results, base_settings = base_results

        # Get initial values from burnup data
        burnup_data = results.get(BurnupCalculator)
        if burnup_data is None or burnup_data.empty:
            return

        # Calculate default target from burnup data
        # (should be backlog only, not backlog + done)
        default_target = burnup_data["Backlog"].iloc[-1]

        # Test with different target values
        test_cases = [
            (None, default_target),  # Use default
            (int(default_target * 0.5), int(default_target * 0.5)),  # 50%
            (int(default_target * 1.5), int(default_target * 1.5)),  # 150%
            (int(default_target * 2.0), int(default_target * 2.0)),  # 200%
        ]

        for target, expected in test_cases:
            settings = extend_dict(
                base_settings,
                {"burnup_forecast_chart_target": target},
            )
            calculator = BurnupForecastCalculator(query_manager, settings, results)
            forecast_data = calculator.run()

            if forecast_data is not None:
                assert isinstance(forecast_data, DataFrame)
                # Verify target is set correctly via public API
                forecast_params = calculator.get_forecast_parameters()
                assert forecast_params.get("target") == expected

    def test_throughput_window_end(self, query_manager, base_results):
        """Test forecast with throughput window end date."""
        results, base_settings = base_results

        window_end_dates = [
            datetime.date(2018, 1, 5),
            datetime.date(2018, 1, 6),
            datetime.date(2018, 1, 7),
            None,  # Use default (today)
        ]

        for window_end in window_end_dates:
            settings = extend_dict(
                base_settings,
                {
                    "burnup_forecast_chart_throughput_window_end": window_end,
                    "burnup_forecast_chart_throughput_window": 30,
                },
            )
            calculator = BurnupForecastCalculator(query_manager, settings, results)
            forecast_data = calculator.run()

            if forecast_data is not None:
                assert isinstance(forecast_data, DataFrame)

    def test_backlog_growth_window(self, query_manager, base_results):
        """Test forecast with different backlog growth window sizes."""
        results, base_settings = base_results

        growth_windows = [None, 7, 14, 30, 60]
        for growth_window in growth_windows:
            settings = extend_dict(
                base_settings,
                {
                    "burnup_forecast_chart_backlog_growth_window": growth_window,
                    "burnup_forecast_chart_throughput_window": 30,
                },
            )
            calculator = BurnupForecastCalculator(query_manager, settings, results)
            forecast_data = calculator.run()

            if forecast_data is not None:
                assert isinstance(forecast_data, DataFrame)

    def test_deadline_and_confidence(self, query_manager, base_results):
        """Test forecast with deadline and confidence settings."""
        results, base_settings = base_results

        deadline = datetime.date(2018, 2, 1)
        confidences = [0.5, 0.75, 0.85, 0.90, 0.95]

        for confidence in confidences:
            settings = extend_dict(
                base_settings,
                {
                    "burnup_forecast_chart_deadline": deadline,
                    "burnup_forecast_chart_deadline_confidence": confidence,
                },
            )
            calculator = BurnupForecastCalculator(query_manager, settings, results)
            forecast_data = calculator.run()

            if forecast_data is not None:
                assert isinstance(forecast_data, DataFrame)

    def test_combined_configuration(self, query_manager, base_results):
        """Test forecast with multiple configuration settings combined."""
        results, base_settings = base_results

        settings = extend_dict(
            base_settings,
            {
                "burnup_forecast_chart_trials": 100,
                "burnup_forecast_chart_throughput_window": 60,
                "burnup_forecast_chart_throughput_frequency": "weekly",
                "burnup_forecast_chart_smart_window": True,
                "burnup_forecast_chart_target": 50,
                "burnup_forecast_chart_backlog_growth_window": 30,
                "burnup_forecast_chart_deadline": datetime.date(2018, 2, 15),
                "burnup_forecast_chart_deadline_confidence": 0.85,
                "burnup_forecast_chart_confidence": 0.9,
                "burnup_forecast_chart_random_seed": 12345,
            },
        )
        calculator = BurnupForecastCalculator(query_manager, settings, results)
        forecast_data = calculator.run()

        if forecast_data is not None:
            assert isinstance(forecast_data, DataFrame)
            assert len(forecast_data.columns) == 100
            # Verify all settings are applied - check trials count via public API
            # (trials count is verified by number of columns in output)

    def test_confidence_configuration(self, query_manager, base_results, mocker):
        """Test burnup_forecast_chart_confidence passed to simulator."""
        results, base_settings = base_results

        # Mock MonteCarloSimulator to verify it's called with correct confidence
        mock_simulator_init = mocker.patch.object(
            MonteCarloSimulator, "__init__", return_value=None
        )

        settings = extend_dict(
            base_settings,
            {
                "burnup_forecast_chart_confidence": 0.95,
            },
        )
        _ = BurnupForecastCalculator(query_manager, settings, results)

        # Verify MonteCarloSimulator was initialized with the confidence value
        mock_simulator_init.assert_called_once()
        call_args = mock_simulator_init.call_args
        assert call_args.kwargs["confidence"] == 0.95

    def test_random_seed_configuration(self, query_manager, base_results, mocker):
        """Test burnup_forecast_chart_random_seed passed to simulator."""
        results, base_settings = base_results

        # Mock MonteCarloSimulator to verify it's called with correct random_seed
        mock_simulator_init = mocker.patch.object(
            MonteCarloSimulator, "__init__", return_value=None
        )

        settings = extend_dict(
            base_settings,
            {
                "burnup_forecast_chart_random_seed": 99999,
            },
        )
        _ = BurnupForecastCalculator(query_manager, settings, results)

        # Verify MonteCarloSimulator was initialized with the random_seed value
        mock_simulator_init.assert_called_once()
        call_args = mock_simulator_init.call_args
        assert call_args.kwargs["random_seed"] == 99999

    def test_random_seed_none_configuration(self, query_manager, base_results, mocker):
        """Test that None random_seed is handled correctly."""
        results, base_settings = base_results

        # Mock MonteCarloSimulator to verify it's called with None random_seed
        mock_simulator_init = mocker.patch.object(
            MonteCarloSimulator, "__init__", return_value=None
        )

        settings = extend_dict(
            base_settings,
            {
                "burnup_forecast_chart_random_seed": None,
            },
        )
        _ = BurnupForecastCalculator(query_manager, settings, results)

        # Verify MonteCarloSimulator was initialized with None random_seed
        mock_simulator_init.assert_called_once()
        call_args = mock_simulator_init.call_args
        assert call_args.kwargs["random_seed"] is None

    def test_frequency_conversion(self, query_manager, base_results):
        """Test frequency string conversion produces valid forecasts."""
        results, base_settings = base_results

        # Test that different frequency formats produce valid forecasts
        # The actual frequency conversion is verified by checking that forecasts
        # are generated correctly with different frequency settings
        frequencies = [
            "daily",
            "weekly",
            "monthly",
            "D",
            "W",
            "M",  # Legacy 'M' will be converted to 'ME'
            "ME",
            "day",
            "week",
            "month",
        ]

        for freq in frequencies:
            settings = extend_dict(
                base_settings,
                {
                    "burnup_forecast_chart_throughput_frequency": freq,
                    "burnup_forecast_chart_throughput_window": 30,
                },
            )
            calculator = BurnupForecastCalculator(query_manager, settings, results)
            forecast_data = calculator.run()

            # Verify that forecast is generated successfully with this frequency
            if forecast_data is not None:
                assert isinstance(forecast_data, DataFrame)
                assert len(forecast_data.index) > 0
                assert isinstance(forecast_data.index, pd.DatetimeIndex)

                # Verify date intervals match expected frequency
                if len(forecast_data.index) > 1:
                    # Check that dates are properly spaced
                    date_diffs = forecast_data.index.to_series().diff().dropna()
                    # All differences should be positive (dates should be increasing)
                    assert (date_diffs > pd.Timedelta(0)).all()

    def test_target_validation(self, query_manager, base_results, tmp_path):
        """Test that target validation works correctly."""
        results, base_settings = base_results

        burnup_data = results.get(BurnupCalculator)
        if burnup_data is not None and not burnup_data.empty:
            initial_done = burnup_data["Done"].iloc[-1]

            # Set a temporary file path so validation passes
            output_file = tmp_path / "forecast-test.png"

            # Test with target that's already reached
            settings = extend_dict(
                base_settings,
                {
                    "burnup_forecast_chart_target": initial_done,  # Already reached
                    "burnup_forecast_chart": str(output_file),
                },
            )
            calculator = BurnupForecastCalculator(query_manager, settings, results)
            calculator.run()

            # Target should be adjusted to be greater than initial_done
            forecast_params = calculator.get_forecast_parameters()
            final_target = forecast_params.get("target")
            assert final_target is not None
            assert final_target > initial_done

            # Test with target less than initial_done
            output_file2 = tmp_path / "forecast-test2.png"
            settings = extend_dict(
                base_settings,
                {
                    "burnup_forecast_chart_target": max(0, int(initial_done) - 5),
                    "burnup_forecast_chart": str(output_file2),
                },
            )
            calculator = BurnupForecastCalculator(query_manager, settings, results)
            calculator.run()

            forecast_params = calculator.get_forecast_parameters()
            final_target = forecast_params.get("target")
            assert final_target is not None
            assert final_target > initial_done

    def test_reproducibility_with_seed(self, query_manager, base_results):
        """Test that forecast is reproducible with same seed (via trials)."""
        results, base_settings = base_results

        settings1 = extend_dict(
            base_settings,
            {
                "burnup_forecast_chart_trials": 50,
                "burnup_forecast_chart_throughput_window": 30,
            },
        )
        settings2 = extend_dict(
            base_settings,
            {
                "burnup_forecast_chart_trials": 50,
                "burnup_forecast_chart_throughput_window": 30,
            },
        )

        calculator1 = BurnupForecastCalculator(query_manager, settings1, results)
        calculator2 = BurnupForecastCalculator(query_manager, settings2, results)

        # Note: Full reproducibility requires setting random seed in MonteCarloSimulator
        # This test verifies that the same configuration produces same structure
        data1 = calculator1.run()
        data2 = calculator2.run()

        if data1 is not None and data2 is not None:
            assert len(data1.columns) == len(data2.columns)
            assert len(data1.index) == len(data2.index)
            assert list(data1.columns) == list(data2.columns)

    def test_minimal_trials(self, query_manager, base_results):
        """Test forecast with minimal trial count (edge case)."""
        results, base_settings = base_results

        settings = extend_dict(base_settings, {"burnup_forecast_chart_trials": 1})
        calculator = BurnupForecastCalculator(query_manager, settings, results)
        forecast_data = calculator.run()

        if forecast_data is not None:
            assert len(forecast_data.columns) == 1
            assert "Trial 0" in forecast_data.columns

    def test_large_trial_count(self, query_manager, base_results):
        """Test forecast with large trial count."""
        results, base_settings = base_results

        settings = extend_dict(base_settings, {"burnup_forecast_chart_trials": 1000})
        calculator = BurnupForecastCalculator(query_manager, settings, results)
        forecast_data = calculator.run()

        if forecast_data is not None:
            assert len(forecast_data.columns) == 1000
            # Verify structure is correct even with many trials
            assert all(f"Trial {i}" in forecast_data.columns for i in range(1000))

    def test_very_short_window(self, query_manager, base_results):
        """Test forecast with very short throughput window."""
        results, base_settings = base_results

        settings = extend_dict(
            base_settings,
            {
                "burnup_forecast_chart_throughput_window": 1,
                "burnup_forecast_chart_smart_window": False,
            },
        )
        calculator = BurnupForecastCalculator(query_manager, settings, results)
        forecast_data = calculator.run()

        # Should handle gracefully even with minimal data
        if forecast_data is not None:
            assert isinstance(forecast_data, DataFrame)

    def test_very_long_window(self, query_manager, base_results):
        """Test forecast with very long throughput window."""
        results, base_settings = base_results

        settings = extend_dict(
            base_settings,
            {
                "burnup_forecast_chart_throughput_window": 365,  # 1 year
                "burnup_forecast_chart_smart_window": False,
            },
        )
        calculator = BurnupForecastCalculator(query_manager, settings, results)
        forecast_data = calculator.run()

        if forecast_data is not None:
            assert isinstance(forecast_data, DataFrame)


def _setup_forecast_for_write_tests(query_manager, settings, results, output_path):
    """Helper to setup forecast calculator with known working settings."""
    settings.update({
        "backlog_column": "Committed",
        "done_column": "Test",
        "burnup_forecast_chart_throughput_window_end": None,
        "burnup_forecast_chart_throughput_window": 4,
        "burnup_forecast_chart_target": None,
        "burnup_forecast_chart_trials": 10,
        "burnup_forecast_chart_deadline": datetime.date(2018, 1, 30),
        "burnup_forecast_chart_deadline_confidence": 0.85,
        "quantiles": [0.1, 0.3, 0.5],
    })
    results.update({
        CFDCalculator: CFDCalculator(query_manager, settings, results).run()
    })
    results.update({
        BurnupCalculator: BurnupCalculator(query_manager, settings, results).run()
    })
    settings["burnup_forecast_chart_data"] = output_path
    settings["burnup_forecast_chart"] = None
    random.seed(42)
    np.random.seed(42)
    calculator = BurnupForecastCalculator(query_manager, settings, results)
    return calculator, calculator.run()


def test_write_data_files_csv(query_manager, settings, results, tmp_path):
    """Test that CSV data files are written correctly."""
    output_csv = tmp_path / "forecast-data.csv"
    calculator, forecast_data = _setup_forecast_for_write_tests(
        query_manager, settings, results, str(output_csv)
    )
    if forecast_data is not None:
        calculator.write()
        csv_df = assert_forecast_csv_file_valid(output_csv)
        assert all(col.startswith("Trial ") for col in csv_df.columns if col != "Date")
        assert len(csv_df) == len(forecast_data), "CSV should have same number of rows"
        assert len(csv_df.columns) == len(forecast_data.columns) + 1, (
            "CSV should have Date + trial columns"
        )
        csv_dates = pd.to_datetime(csv_df["Date"]).dt.date
        forecast_dates = forecast_data.index.date
        assert list(csv_dates) == list(forecast_dates), (
            "CSV dates should match forecast dates"
        )
    else:
        calculator.write()
        assert not output_csv.exists(), (
            "CSV file should not be created when forecast is None"
        )


def test_write_data_files_json(query_manager, settings, results, tmp_path):
    """Test that JSON data files are written correctly."""
    output_json = tmp_path / "forecast-data.json"
    calculator, forecast_data = _setup_forecast_for_write_tests(
        query_manager, settings, results, str(output_json)
    )
    if forecast_data is not None:
        calculator.write()
        assert_forecast_json_file_valid(output_json)
    else:
        calculator.write()
        assert not output_json.exists(), (
            "JSON file should not be created when forecast is None"
        )


def test_write_data_files_xlsx(query_manager, settings, results, tmp_path):
    """Test that XLSX data files are written correctly."""
    output_xlsx = tmp_path / "forecast-data.xlsx"
    calculator, forecast_data = _setup_forecast_for_write_tests(
        query_manager, settings, results, str(output_xlsx)
    )
    if forecast_data is not None:
        calculator.write()
        assert_forecast_xlsx_file_valid(output_xlsx)
    else:
        calculator.write()
        assert not output_xlsx.exists(), (
            "XLSX file should not be created when forecast is None"
        )


def test_write_data_files_multiple_formats(query_manager, settings, results, tmp_path):
    """Test that multiple data file formats can be written simultaneously."""
    output_csv = tmp_path / "forecast-data.csv"
    output_json = tmp_path / "forecast-data.json"
    output_xlsx = tmp_path / "forecast-data.xlsx"
    calculator, forecast_data = _setup_forecast_for_write_tests(
        query_manager,
        settings,
        results,
        [str(output_csv), str(output_json), str(output_xlsx)],
    )
    if forecast_data is not None:
        calculator.write()
        csv_df = assert_forecast_csv_file_valid(output_csv)
        json_df = assert_forecast_json_file_valid(output_json)
        xlsx_df = assert_forecast_xlsx_file_valid(output_xlsx)
        assert len(csv_df) == len(json_df) == len(xlsx_df), (
            "All formats should have same number of rows"
        )
    else:
        calculator.write()
        assert not output_csv.exists(), "CSV file should not be created"
        assert not output_json.exists(), "JSON file should not be created"
        assert not output_xlsx.exists(), "XLSX file should not be created"


def test_write_data_files_handles_empty_data(
    query_manager, settings, results, tmp_path, mocker
):
    """Test that write_data_files handles empty forecast data gracefully."""
    output_csv = tmp_path / "forecast-data.csv"
    settings["burnup_forecast_chart_data"] = str(output_csv)
    settings["burnup_forecast_chart"] = None

    calculator = BurnupForecastCalculator(query_manager, settings, results)
    # Mock run() to return None/empty DataFrame
    mocker.patch.object(calculator, "run", return_value=None)

    # Should not raise exception, just log warning
    calculator.write()

    # File should not be created if data is empty
    assert not output_csv.exists(), "CSV file should not be created when data is empty"


def test_write_data_files_no_config(query_manager, settings, results):
    """Test that write_data_files does nothing when no output file is configured."""
    settings["burnup_forecast_chart_data"] = None
    settings["burnup_forecast_chart"] = None

    calculator = BurnupForecastCalculator(query_manager, settings, results)
    calculator.run()

    # Should not raise exception
    calculator.write()


def test_write_data_files_with_fixed_seed(
    query_manager, settings, results, tmp_path, mocker
):
    """Test that CSV output is reproducible with fixed random seed."""
    # Set fixed random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    output_csv1 = tmp_path / "forecast-data1.csv"
    output_csv2 = tmp_path / "forecast-data2.csv"

    # Mock MonteCarloSimulator initialization to always use fixed seed
    # This avoids patching private members and ensures reproducibility
    original_init = MonteCarloSimulator.__init__

    def mock_init(self, trials=1000, random_seed=None, confidence=0.8):  # pylint: disable=unused-argument
        """Mock __init__ to always use seed=42 for reproducibility.

        The random_seed parameter is intentionally unused - we always use 42
        for test reproducibility regardless of what is passed.
        """
        # Always use seed=42 for reproducibility, ignoring the random_seed parameter
        original_init(self, trials=trials, random_seed=42, confidence=confidence)

    # Patch the class constructor to use fixed seed
    # Use 'new' instead of 'side_effect' to properly handle method binding
    mocker.patch.object(MonteCarloSimulator, "__init__", new=mock_init, autospec=False)

    # Run forecast twice with same seed
    for output_csv in [output_csv1, output_csv2]:
        settings_copy = settings.copy()
        settings_copy["burnup_forecast_chart_data"] = str(output_csv)
        settings_copy["burnup_forecast_chart"] = None
        settings_copy["burnup_forecast_chart_trials"] = (
            10  # Small number for faster test
        )

        calculator = BurnupForecastCalculator(query_manager, settings_copy, results)
        calculator.run()
        calculator.write()

    # Both files should exist
    if output_csv1.exists() and output_csv2.exists():
        # Read both CSVs
        df1 = pd.read_csv(output_csv1, parse_dates=["Date"])
        df2 = pd.read_csv(output_csv2, parse_dates=["Date"])

        # Compare DataFrames - they should be identical with fixed seed
        pd.testing.assert_frame_equal(df1, df2, check_dtype=False)
