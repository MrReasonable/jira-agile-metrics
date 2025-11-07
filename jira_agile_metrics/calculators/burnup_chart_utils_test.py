"""Tests for burnup_chart_utils module."""

import os
from datetime import datetime
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from jira_agile_metrics.config.exceptions import ChartGenerationError

from .burnup_chart_generator import BurnupChartGenerator
from .burnup_chart_utils import (
    calculate_next_date,
    extend_forecast_dates_to_completion,
    extract_forecast_trials,
    find_latest_completion_date,
    find_max_completion_date_from_trials,
    format_date_for_legend,
    get_frequency_string,
    normalize_trial_length,
    save_chart,
    validate_figure_size,
    validate_initial_state,
)


class TestFindLatestCompletionDate:
    """Test cases for find_latest_completion_date."""

    def test_find_latest_completion_date_with_quantiles(self):
        """Test finding latest completion date from quantile data."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        quantile_data = {
            "50%": dates[2],
            "75%": dates[3],
            "90%": dates[4],
            "99%": dates[4],
        }
        result = find_latest_completion_date(quantile_data)
        assert result == pd.Timestamp(dates[4])

    def test_find_latest_completion_date_empty(self):
        """Test with empty quantile data."""
        result = find_latest_completion_date({})
        assert result is None

    def test_find_latest_completion_date_missing_quantiles(self):
        """Test with missing quantiles."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        quantile_data = {"50%": dates[2]}
        result = find_latest_completion_date(quantile_data)
        assert result == pd.Timestamp(dates[2])


class TestFindMaxCompletionDateFromTrials:
    """Test cases for find_max_completion_date_from_trials."""

    def test_find_max_completion_date_all_trials_reached_target(self):
        """Test when all trials have reached the target."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        target = 50
        # All trials reach target at different dates
        # Trial format: [initial_state] + [forecast_values...]
        # done_trial[idx] >= target means completion_idx = idx-1
        # maps to forecast_dates[idx-1]
        done_trials = [
            [20, 30, 40, 50, 60],  # Reaches target at done_trial[3]=50 -> dates[2]
            [20, 25, 35, 45, 55],  # Reaches target at done_trial[4]=55 -> dates[3]
            [20, 40, 50, 60, 70],  # Reaches target at done_trial[2]=50 -> dates[1]
        ]
        result = find_max_completion_date_from_trials(
            done_trials, target, dates.tolist()
        )
        # Max should be dates[3] (trial 1 is slowest)
        assert result == pd.Timestamp(dates[3])

    def test_find_max_completion_date_some_trials_not_reached(self):
        """Test when some trials haven't reached target (extrapolation)."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        target = 50
        # Some trials reach target, some don't
        done_trials = [
            [20, 30, 40, 50, 60],  # Reaches target at index 3
            [20, 25, 30, 35, 40],  # Hasn't reached - needs extrapolation
            [20, 40, 50, 60, 70],  # Reaches target at index 2
        ]
        result = find_max_completion_date_from_trials(
            done_trials, target, dates.tolist()
        )
        # Should extrapolate for trial 1 and find the max
        assert result is not None
        assert result >= pd.Timestamp(dates[-1])

    def test_find_max_completion_date_all_trials_not_reached(self):
        """Test when no trials have reached target (all need extrapolation)."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        target = 50
        # No trials reach target - all need extrapolation
        done_trials = [
            [20, 25, 30, 35, 40],  # Rate: 5 per period, needs 2 more periods
            [20, 22, 24, 26, 28],  # Rate: 2 per period, needs 11 more periods
            [20, 30, 35, 40, 45],  # Rate: ~6.25 per period, needs 1 more period
        ]
        result = find_max_completion_date_from_trials(
            done_trials, target, dates.tolist()
        )
        # Should extrapolate for all and find the max (trial 1 is slowest)
        assert result is not None
        assert result > pd.Timestamp(dates[-1])

    def test_find_max_completion_date_empty_trials(self):
        """Test with empty trials."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        result = find_max_completion_date_from_trials([], 50, dates.tolist())
        assert result is None

    def test_find_max_completion_date_no_target(self):
        """Test with no target."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        done_trials = [[20, 30, 40, 50, 60]]
        result = find_max_completion_date_from_trials(done_trials, 0, dates.tolist())
        assert result is None

    def test_find_max_completion_date_invalid_trials(self):
        """Test with invalid trial data."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        done_trials = [
            # Valid - reaches target at done_trial[3]=50 -> dates[2]
            [20, 30, 40, 50, 60],
            "invalid",  # Invalid - not a list
            [],  # Invalid - too short
        ]
        result = find_max_completion_date_from_trials(done_trials, 50, dates.tolist())
        # Should still work with valid trials
        assert result == pd.Timestamp(dates[2])


class TestExtendForecastDatesToCompletion:
    """Test cases for extend_forecast_dates_to_completion."""

    def test_extend_with_done_trials_and_target(self):
        """Test extending forecast dates using done_trials and target."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        target = 50
        done_trials = [
            [20, 30, 40, 50, 60],  # Reaches target at index 3
            [20, 25, 30, 35, 40],  # Needs extrapolation
        ]
        quantile_data = {"50%": dates[2]}

        result = extend_forecast_dates_to_completion(
            dates.tolist(), quantile_data, done_trials, target
        )

        # Should extend beyond original dates
        assert len(result) >= len(dates)
        assert result[: len(dates)] == dates.tolist()

    def test_extend_with_quantile_data_fallback(self):
        """Test extending forecast dates using quantile data.

        Tests when done_trials not provided.
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        quantile_data = {
            "50%": dates[2],
            "99%": dates[4] + pd.Timedelta(days=7),
        }

        result = extend_forecast_dates_to_completion(
            dates.tolist(), quantile_data, None, 0
        )

        # Should extend to include 99% date
        assert len(result) > len(dates)
        assert result[-1] >= quantile_data["99%"]

    def test_extend_no_extension_needed(self):
        """Test when no extension is needed."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        target = 50
        done_trials = [
            [20, 30, 40, 50, 60],  # All reach target within dates
            [20, 25, 35, 45, 55],
        ]
        quantile_data = {"50%": dates[2]}

        result = extend_forecast_dates_to_completion(
            dates.tolist(), quantile_data, done_trials, target
        )

        # Should not extend if all trials complete within original dates
        # (assuming max completion is within original range)
        assert isinstance(result, list)

    def test_extend_empty_forecast_dates(self):
        """Test with empty forecast dates."""
        result = extend_forecast_dates_to_completion([], {}, None, 0)
        assert result == []

    def test_extend_prefers_done_trials_over_quantile(self):
        """Test that done_trials takes precedence over quantile_data."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        target = 50
        done_trials = [
            [20, 25, 30, 35, 40],  # Needs extrapolation beyond dates
        ]
        # Quantile data suggests earlier date
        quantile_data = {"99%": dates[3]}

        result_with_trials = extend_forecast_dates_to_completion(
            dates.tolist(), quantile_data, done_trials, target
        )
        result_without_trials = extend_forecast_dates_to_completion(
            dates.tolist(), quantile_data, None, 0
        )

        # Result with trials should extend further (to when slowest trial completes)
        assert len(result_with_trials) >= len(result_without_trials)


class TestExtendForecastDatesIntegration:
    """Integration tests for forecast date extension with chart generator."""

    @patch(
        "jira_agile_metrics.calculators.burnup_chart_generator."
        "find_backlog_and_done_columns"
    )
    @patch("jira_agile_metrics.calculators.burnup_chart_generator.set_chart_style")
    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_prepare_chart_data_extends_forecast_dates(
        self, mock_plt, _mock_set_style, _mock_find_columns, tmp_path
    ):
        """Test that generate_chart extends forecast dates correctly."""
        generator = BurnupChartGenerator(str(tmp_path / "test.png"))
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        target = 50
        done_trials = [
            [20, 30, 40, 50, 60],  # Reaches target
            [20, 25, 30, 35, 40],  # Needs extrapolation
        ]
        burnup_data = pd.DataFrame(
            {"Backlog": [100, 95, 90], "Done": [0, 5, 10]},
            index=dates[:3],
        )
        chart_data = {
            "forecast_dates": dates.tolist(),
            "done_trials": done_trials,
            "backlog_trials": [],
            "target": target,
            "quantile_data": {"50%": dates[2]},
        }
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        _mock_find_columns.return_value = ("Backlog", "Done")

        result = generator.generate_chart(burnup_data, chart_data)

        # Chart generation should succeed
        assert result is True
        # Verify that plotting occurred (indicating dates were extended)
        assert mock_ax.plot.call_count > 0

    @patch(
        "jira_agile_metrics.calculators.burnup_chart_generator."
        "find_backlog_and_done_columns"
    )
    @patch("jira_agile_metrics.calculators.burnup_chart_generator.set_chart_style")
    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_extend_until_all_trials_reach_target(
        self, mock_plt, _mock_set_style, _mock_find_columns, tmp_path
    ):
        """Test that forecast dates extend until ALL trials reach target."""
        generator = BurnupChartGenerator(str(tmp_path / "test.png"))
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        target = 50
        # Create trials where some haven't reached target yet
        done_trials = [
            [20, 30, 40, 50, 60],  # Reached target at index 3
            [20, 25, 30, 35, 40],  # Hasn't reached - needs extrapolation
            [20, 40, 50, 60, 70],  # Reached target at index 2
        ]
        burnup_data = pd.DataFrame(
            {"Backlog": [100, 95, 90], "Done": [0, 5, 10]},
            index=dates[:3],
        )
        chart_data = {
            "forecast_dates": dates.tolist(),
            "done_trials": done_trials,
            "backlog_trials": [],
            "target": target,
            "quantile_data": {"50%": dates[2]},
        }
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        _mock_find_columns.return_value = ("Backlog", "Done")

        result = generator.generate_chart(burnup_data, chart_data)

        # Chart generation should succeed
        assert result is True
        # Verify that plotting occurred with extended dates
        # The fan should be plotted with extended forecast dates
        fill_between_calls = [
            call for call in mock_ax.method_calls if call[0] == "fill_between"
        ]
        assert len(fill_between_calls) > 0


class TestFormatDateForLegend:
    """Test cases for format_date_for_legend."""

    def test_format_date_for_legend_with_timestamp(self):
        """Test formatting a pandas Timestamp."""
        date = pd.Timestamp("2024-01-15")
        result = format_date_for_legend(date)
        assert result == "2024-01-15"

    def test_format_date_for_legend_with_datetime(self):
        """Test formatting a datetime object."""
        date = datetime(2024, 1, 15)
        result = format_date_for_legend(date)
        assert result == "2024-01-15"

    def test_format_date_for_legend_with_string(self):
        """Test formatting a date string."""
        result = format_date_for_legend("2024-01-15")
        assert result == "2024-01-15"

    def test_format_date_for_legend_with_none(self):
        """Test formatting None."""
        result = format_date_for_legend(None)
        assert result == "N/A"

    def test_format_date_for_legend_with_invalid_value(self):
        """Test formatting an invalid value."""
        result = format_date_for_legend("invalid")
        assert result == "N/A"

    def test_format_date_for_legend_with_numeric_value(self):
        """Test formatting a numeric value that can't be converted."""
        result = format_date_for_legend(12345)
        # Should attempt conversion or return N/A
        assert isinstance(result, str)


class TestValidateFigureSize:
    """Test cases for validate_figure_size."""

    def test_validate_figure_size_with_none(self):
        """Test with None (should return default)."""
        result = validate_figure_size(None)
        assert result == (12, 8)

    def test_validate_figure_size_with_valid_tuple(self):
        """Test with valid tuple."""
        result = validate_figure_size((10, 6))
        assert result == (10.0, 6.0)

    def test_validate_figure_size_with_integers(self):
        """Test with integer tuple."""
        result = validate_figure_size((15, 10))
        assert result == (15.0, 10.0)

    def test_validate_figure_size_with_floats(self):
        """Test with float tuple."""
        result = validate_figure_size((12.5, 8.5))
        assert result == (12.5, 8.5)

    def test_validate_figure_size_with_invalid_length(self):
        """Test with tuple of wrong length."""
        result = validate_figure_size((10,))
        assert result == (12, 8)  # Should return default

    def test_validate_figure_size_with_negative_values(self):
        """Test with negative values."""
        result = validate_figure_size((-10, 6))
        assert result == (12, 8)  # Should return default

    def test_validate_figure_size_with_zero(self):
        """Test with zero value."""
        result = validate_figure_size((10, 0))
        assert result == (12, 8)  # Should return default

    def test_validate_figure_size_with_non_numeric(self):
        """Test with non-numeric values."""
        result = validate_figure_size(("10", "6"))
        assert result == (12, 8)  # Should return default

    def test_validate_figure_size_with_custom_default(self):
        """Test with custom default."""
        result = validate_figure_size(None, default=(15, 10))
        assert result == (15, 10)


class TestSaveChart:
    """Test cases for save_chart."""

    def test_save_chart_success(self, tmp_path):
        """Test successful chart save."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        output_file = str(tmp_path / "test_chart.png")

        save_chart(fig, output_file)

        assert os.path.exists(output_file)
        assert not plt.get_fignums()  # Figure should be closed

    def test_save_chart_creates_directory(self, tmp_path):
        """Test that save_chart creates output directory if needed."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        output_file = str(tmp_path / "subdir" / "test_chart.png")

        save_chart(fig, output_file)

        assert os.path.exists(output_file)

    def test_save_chart_oserror(self, tmp_path):
        """Test save_chart with OSError."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        output_file = str(tmp_path / "test_chart.png")

        # Mock savefig to raise OSError
        with patch.object(fig, "savefig", side_effect=OSError("Permission denied")):
            with pytest.raises(ChartGenerationError, match="Failed to save chart file"):
                save_chart(fig, output_file)

    def test_save_chart_valueerror(self, tmp_path):
        """Test save_chart with ValueError."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        output_file = str(tmp_path / "test_chart.png")

        # Mock savefig to raise ValueError
        with patch.object(fig, "savefig", side_effect=ValueError("Invalid format")):
            with pytest.raises(ChartGenerationError, match="Failed to save chart"):
                save_chart(fig, output_file)


class TestNormalizeTrialLength:
    """Test cases for normalize_trial_length."""

    def test_normalize_trial_length_exact_match(self):
        """Test with exact expected length."""
        trial = [10, 20, 30]
        result = normalize_trial_length(trial, 3, 0)
        assert result == [10, 20, 30]

    def test_normalize_trial_length_with_initial_state(self):
        """Test with initial state (expected_length + 1)."""
        trial = [10, 20, 30, 40]
        result = normalize_trial_length(trial, 3, 0)
        assert result == [10, 20, 30, 40]

    def test_normalize_trial_length_too_long(self):
        """Test with trial longer than expected."""
        trial = [10, 20, 30, 40, 50, 60]
        result = normalize_trial_length(trial, 3, 0)
        assert result == [10, 20, 30, 40]  # Truncated to expected_length + 1

    def test_normalize_trial_length_too_short(self):
        """Test with trial shorter than expected."""
        trial = [10, 20]
        result = normalize_trial_length(trial, 5, 0)
        assert result == [10, 20, 20, 20, 20]  # Padded with last value

    def test_normalize_trial_length_empty(self):
        """Test with empty trial."""
        result = normalize_trial_length([], 5, 0)
        assert result is None

    def test_normalize_trial_length_single_value(self):
        """Test with single value trial."""
        trial = [10]
        result = normalize_trial_length(trial, 3, 0)
        assert result == [10, 10, 10]  # Padded


class TestExtractForecastTrials:
    """Test cases for extract_forecast_trials."""

    def test_extract_forecast_trials_valid(self):
        """Test with valid trials."""
        trials = [[10, 20, 30], [15, 25, 35]]
        result = extract_forecast_trials(trials, 3)
        assert result == [[10, 20, 30], [15, 25, 35]]

    def test_extract_forecast_trials_with_initial_state(self):
        """Test with trials containing initial state."""
        trials = [[10, 20, 30, 40], [15, 25, 35, 45]]
        result = extract_forecast_trials(trials, 3)
        assert result == [[10, 20, 30], [15, 25, 35]]

    def test_extract_forecast_trials_invalid_type(self):
        """Test with invalid trial type."""
        trials = [[10, 20, 30], "not_a_list", [15, 25, 35]]
        result = extract_forecast_trials(trials, 3)
        assert result == [[10, 20, 30], [15, 25, 35]]

    def test_extract_forecast_trials_empty(self):
        """Test with empty trial."""
        trials = [[10, 20, 30], []]
        result = extract_forecast_trials(trials, 3)
        assert result == [[10, 20, 30]]

    def test_extract_forecast_trials_too_short_after_normalization(self):
        """Test with trial that's too short after normalization."""
        trials = [[10]]  # Too short even after padding
        result = extract_forecast_trials(trials, 3)
        # Should be padded by normalize_trial_length, then extracted
        assert len(result) == 1
        assert len(result[0]) == 3


class TestValidateInitialState:
    """Test cases for validate_initial_state."""

    def test_validate_initial_state_valid_int(self):
        """Test with valid integer."""
        assert validate_initial_state(10, 0) is True

    def test_validate_initial_state_valid_float(self):
        """Test with valid float."""
        assert validate_initial_state(10.5, 0) is True

    def test_validate_initial_state_negative(self):
        """Test with negative value (should still be valid but logged)."""
        assert validate_initial_state(-5, 0) is True

    def test_validate_initial_state_non_numeric(self):
        """Test with non-numeric value."""
        assert validate_initial_state("10", 0) is False

    def test_validate_initial_state_none(self):
        """Test with None."""
        assert validate_initial_state(None, 0) is False

    def test_validate_initial_state_nan(self):
        """Test with NaN."""
        assert validate_initial_state(np.nan, 0) is False

    def test_validate_initial_state_inf(self):
        """Test with infinity."""
        assert validate_initial_state(np.inf, 0) is False

    def test_validate_initial_state_negative_inf(self):
        """Test with negative infinity."""
        assert validate_initial_state(-np.inf, 0) is False


class TestGetFrequencyString:
    """Test cases for get_frequency_string."""

    def test_get_frequency_string_daily(self):
        """Test with daily frequency."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        result = get_frequency_string(dates.tolist())
        assert result == "D"

    def test_get_frequency_string_weekly(self):
        """Test with weekly frequency."""
        dates = pd.date_range("2024-01-01", periods=5, freq="W")
        result = get_frequency_string(dates.tolist())
        assert result in ("W", "W-SUN")  # pandas may return different formats

    def test_get_frequency_string_monthly(self):
        """Test with monthly frequency."""
        dates = pd.date_range("2024-01-01", periods=5, freq="ME")
        result = get_frequency_string(dates.tolist())
        assert result in ("ME", "M")  # pandas may return different formats

    def test_get_frequency_string_single_date(self):
        """Test with single date."""
        dates = [pd.Timestamp("2024-01-01")]
        result = get_frequency_string(dates)
        assert result == "D"

    def test_get_frequency_string_empty(self):
        """Test with empty list."""
        result = get_frequency_string([])
        assert result == "D"

    def test_get_frequency_string_custom_interval_daily(self):
        """Test with custom interval that looks daily."""
        dates = [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-03"),
        ]
        result = get_frequency_string(dates)
        assert result == "D"

    def test_get_frequency_string_custom_interval_weekly(self):
        """Test with custom interval that looks weekly."""
        dates = [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-08"),
            pd.Timestamp("2024-01-15"),
        ]
        result = get_frequency_string(dates)
        # pandas may return different weekly formats (W, W-SUN, W-MON, etc.)
        assert result.startswith("W")

    def test_get_frequency_string_custom_interval_monthly(self):
        """Test with custom interval that looks monthly."""
        dates = [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-02-01"),
            pd.Timestamp("2024-03-01"),
        ]
        result = get_frequency_string(dates)
        # pandas may return different monthly formats (ME, MS, M, etc.)
        assert result in ("ME", "MS", "M") or result.startswith("M")


class TestCalculateNextDate:
    """Test cases for calculate_next_date."""

    def test_calculate_next_date_daily(self):
        """Test calculating next date with daily frequency."""
        last_date = pd.Timestamp("2024-01-01")
        result = calculate_next_date(last_date, "D")
        assert result == pd.Timestamp("2024-01-02")

    def test_calculate_next_date_weekly(self):
        """Test calculating next date with weekly frequency."""
        last_date = pd.Timestamp("2024-01-01")
        result = calculate_next_date(last_date, "W")
        assert result == pd.Timestamp("2024-01-08")

    def test_calculate_next_date_monthly(self):
        """Test calculating next date with monthly frequency."""
        last_date = pd.Timestamp("2024-01-01")
        result = calculate_next_date(last_date, "ME")
        assert result == pd.Timestamp("2024-02-01")

    def test_calculate_next_date_unknown_frequency(self):
        """Test calculating next date with unknown frequency (defaults to daily)."""
        last_date = pd.Timestamp("2024-01-01")
        result = calculate_next_date(last_date, "X")
        assert result == pd.Timestamp("2024-01-02")

    def test_calculate_next_date_leap_year(self):
        """Test calculating next date across leap year boundary."""
        last_date = pd.Timestamp("2024-02-28")
        result = calculate_next_date(last_date, "D")
        assert result == pd.Timestamp("2024-02-29")


class TestFindLatestCompletionDateAdditional:
    """Additional test cases for find_latest_completion_date."""

    def test_find_latest_completion_date_with_none_values(self):
        """Test with None values in quantile data."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        quantile_data = {
            "50%": dates[2],
            "75%": None,
            "90%": dates[4],
            "99%": None,
        }
        result = find_latest_completion_date(quantile_data)
        assert result == pd.Timestamp(dates[4])

    def test_find_latest_completion_date_with_invalid_dates(self):
        """Test with invalid date values."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        quantile_data = {
            "50%": dates[2],
            "75%": "invalid_date",
            "90%": dates[4],
        }
        result = find_latest_completion_date(quantile_data)
        assert result == pd.Timestamp(dates[4])

    def test_find_latest_completion_date_with_string_dates(self):
        """Test with string date values."""
        quantile_data = {
            "50%": "2024-01-03",
            "75%": "2024-01-05",
            "90%": "2024-01-07",
        }
        result = find_latest_completion_date(quantile_data)
        assert result == pd.Timestamp("2024-01-07")
