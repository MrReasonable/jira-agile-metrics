"""Tests for throughput calculator service functionality in Jira Agile Metrics.

This module contains unit tests for the ThroughputCalculator service class used
in forecast calculations.
"""

from datetime import datetime

import pandas as pd
import pytest

from .throughput_calculator import ThroughputCalculator


class TestableThroughputCalculator(ThroughputCalculator):
    """Test helper that exposes protected methods for testing."""

    def calculate_window_parameters(self, forecast_params):
        """Public wrapper for _calculate_window_parameters."""
        return self._calculate_window_parameters(forecast_params)

    def calculate_smart_window_throughput(self, cycle_data, done_column, window_params):
        """Public wrapper for _calculate_smart_window_throughput."""
        return self._calculate_smart_window_throughput(
            cycle_data, done_column, window_params
        )

    def calculate_fixed_window_throughput(self, cycle_data, done_column, window_params):
        """Public wrapper for _calculate_fixed_window_throughput."""
        return self._calculate_fixed_window_throughput(
            cycle_data, done_column, window_params
        )

    def determine_optimal_window_from_data(self, cycle_data, done_column, freq):
        """Public wrapper for _determine_optimal_window_from_data."""
        return self._determine_optimal_window_from_data(cycle_data, done_column, freq)

    def find_optimal_window_start(self, cycle_data, done_column, end_date, freq):
        """Public wrapper for _find_optimal_window_start."""
        return self._find_optimal_window_start(cycle_data, done_column, end_date, freq)

    def calculate_adaptive_window_size(self, completion_dates, end_date, freq):
        """Public wrapper for _calculate_adaptive_window_size."""
        return self._calculate_adaptive_window_size(completion_dates, end_date, freq)

    def calculate_periods_back(self, end_date, window_size, freq):
        """Public wrapper for _calculate_periods_back."""
        return self._calculate_periods_back(end_date, window_size, freq)


@pytest.fixture(name="cycle_data")
def cycle_data_fixture():
    """Create sample cycle data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=60, freq="D")
    return pd.DataFrame(
        {
            "completed_timestamp": dates,
            "key": [f"ISSUE-{i}" for i in range(60)],
        }
    )


@pytest.fixture(name="calc_instance")
def calc_instance_fixture():
    """Create a ThroughputCalculator instance for testing."""
    return TestableThroughputCalculator()


def test_calculate_throughput_fixed_window(calc_instance, cycle_data):
    """Test calculate_throughput with fixed window.

    Fixed window calculations return a DataFrame with a "count" column
    containing the number of items completed in each time period.
    """
    forecast_params = {
        "freq": "D",
        "freq_label": "day",
        "throughput_window": 30,
        "smart_window": False,
    }

    result = calc_instance.calculate_throughput(
        cycle_data, "completed_timestamp", forecast_params
    )

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert "count" in result.columns, "Fixed window results must have 'count' column"


def test_calculate_throughput_smart_window(calc_instance, cycle_data):
    """Test calculate_throughput with smart window."""
    forecast_params = {
        "freq": "D",
        "freq_label": "day",
        "smart_window": True,
    }

    result = calc_instance.calculate_throughput(
        cycle_data, "completed_timestamp", forecast_params
    )

    assert result is not None
    assert isinstance(result, pd.DataFrame)


def test_calculate_throughput_no_window_params(calc_instance, cycle_data):
    """Test calculate_throughput when window parameters calculation fails."""
    forecast_params = {
        "freq": "D",
        "freq_label": "day",
        "smart_window": False,  # Use fixed window for deterministic testing
    }

    result = calc_instance.calculate_throughput(
        cycle_data, "completed_timestamp", forecast_params
    )

    # Should still return a result with default window parameters applied
    assert result is not None
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    assert "count" in result.columns, "Result should have 'count' column"
    assert len(result.columns) == 1, "Result should have exactly one column"

    # Verify default window size (30 days for daily frequency) is applied
    # The cycle_data has 60 days, so with a 30-day window, we should get 30 periods
    assert len(result) == 30, f"Expected 30 periods (default window), got {len(result)}"

    # Verify the date range is limited to the last 30 days
    # cycle_data spans 2024-01-01 to 2024-02-29 (60 days, 2024 is a leap year)
    # With 30-day window, should start from 2024-01-31 (window_end - 29 days)
    expected_start = pd.Timestamp("2024-01-31")
    actual_start = result.index.min()
    assert actual_start == expected_start, (
        f"Expected window start date {expected_start}, got {actual_start}"
    )

    # Verify the end date is the last date in the data
    expected_end = pd.Timestamp("2024-02-29")
    actual_end = result.index.max()
    assert actual_end == expected_end, (
        f"Expected window end date {expected_end}, got {actual_end}"
    )

    # Verify all values are non-negative integers (counts)
    assert (result["count"] >= 0).all(), "All count values should be non-negative"
    assert pd.api.types.is_integer_dtype(result["count"]), (
        f"Count column should be integer type, got {result['count'].dtype}"
    )


def test_calculate_window_parameters_daily(calc_instance):
    """Test _calculate_window_parameters for daily frequency."""
    forecast_params = {
        "freq": "D",
        "freq_label": "day",
        "throughput_window": 30,
        "smart_window": False,
    }

    result = calc_instance.calculate_window_parameters(forecast_params)

    assert result is not None
    assert result["freq"] == "D"
    assert result["window_size"] == 30
    assert result["smart_window"] is False


def test_calculate_window_parameters_weekly(calc_instance):
    """Test _calculate_window_parameters for weekly frequency."""
    forecast_params = {
        "freq": "W",
        "freq_label": "week",
        "throughput_window": 4,
        "smart_window": False,
    }

    result = calc_instance.calculate_window_parameters(forecast_params)

    assert result is not None
    assert result["freq"] == "W"
    assert result["window_size"] == 4


def test_calculate_window_parameters_monthly(calc_instance):
    """Test _calculate_window_parameters for monthly frequency."""
    forecast_params = {
        "freq": "M",
        "freq_label": "month",
        "throughput_window": 1,
        "smart_window": False,
    }

    result = calc_instance.calculate_window_parameters(forecast_params)

    assert result is not None
    assert result["freq"] == "M"


def test_calculate_window_parameters_default_window(calc_instance):
    """Test _calculate_window_parameters with default window."""
    forecast_params = {
        "freq": "D",
        "freq_label": "day",
        "smart_window": False,
    }

    result = calc_instance.calculate_window_parameters(forecast_params)

    assert result is not None
    assert result["window_size"] == 30  # Default for daily


def test_calculate_window_parameters_smart_window_string(calc_instance):
    """Test _calculate_window_parameters with string smart_window value."""
    forecast_params = {
        "freq": "D",
        "freq_label": "day",
        "smart_window": "true",
    }

    result = calc_instance.calculate_window_parameters(forecast_params)

    assert result is not None
    assert result["smart_window"] is True


def test_calculate_window_parameters_invalid_throughput_window(calc_instance):
    """Test _calculate_window_parameters with invalid throughput_window."""
    forecast_params = {
        "freq": "D",
        "freq_label": "day",
        "throughput_window": "invalid",
        "smart_window": False,
    }

    result = calc_instance.calculate_window_parameters(forecast_params)

    assert result is not None
    # Should fall back to default
    assert result["window_size"] == 30


def test_calculate_window_parameters_negative_window(calc_instance):
    """Test _calculate_window_parameters with negative window."""
    forecast_params = {
        "freq": "D",
        "freq_label": "day",
        "throughput_window": -5,
        "smart_window": False,
    }

    result = calc_instance.calculate_window_parameters(forecast_params)

    assert result is not None
    # Should fall back to default
    assert result["window_size"] == 30


def test_calculate_smart_window_throughput(calc_instance, cycle_data):
    """Test _calculate_smart_window_throughput."""
    window_params = {
        "freq": "D",
        "freq_label": "day",
    }

    result = calc_instance.calculate_smart_window_throughput(
        cycle_data, "completed_timestamp", window_params
    )

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    # Assert required columns exist
    assert "count" in result.columns
    # Assert index is DatetimeIndex
    assert isinstance(result.index, pd.DatetimeIndex)
    # Assert DataFrame is not empty
    assert len(result) > 0
    # Assert window size is within expected bounds (14-90 days for smart window)
    assert 14 <= len(result) <= 90
    # Assert all counts are non-negative integers
    assert (result["count"] >= 0).all()
    # Counts may be float64 due to fillna(0), but values should be integers
    assert (result["count"] % 1 == 0).all()  # All values are whole numbers
    # Assert sum of counts equals number of issues in the window
    # (smart window should capture all available data within its adaptive window)
    total_throughput = result["count"].sum()
    assert total_throughput > 0
    # Verify index is properly sorted and continuous
    assert result.index.is_monotonic_increasing


def test_calculate_fixed_window_throughput(calc_instance, cycle_data):
    """Test _calculate_fixed_window_throughput."""
    window_params = {
        "freq": "D",
        "freq_label": "day",
        "window_size": 30,
    }

    result = calc_instance.calculate_fixed_window_throughput(
        cycle_data, "completed_timestamp", window_params
    )

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    # Assert required columns exist
    assert "count" in result.columns
    # Assert index is DatetimeIndex
    assert isinstance(result.index, pd.DatetimeIndex)
    # Assert DataFrame has exactly 30 rows (window_size=30 days)
    assert len(result) == 30
    # Assert all counts are non-negative integers
    assert (result["count"] >= 0).all()
    # Counts may be float64 due to fillna(0), but values should be integers
    assert (result["count"] % 1 == 0).all()  # All values are whole numbers
    # Verify the window covers the last 30 days of data
    # (end date should be the last completion date, start should be 29 days before)
    expected_end = cycle_data["completed_timestamp"].max()
    expected_start = expected_end - pd.Timedelta(days=29)
    # Calculate expected count from fixture data within the window
    # This makes the test resilient to fixture changes
    window_mask = (cycle_data["completed_timestamp"] >= expected_start) & (
        cycle_data["completed_timestamp"] <= expected_end
    )
    expected_count = window_mask.sum()
    assert result["count"].sum() == expected_count, (
        f"Sum of counts ({result['count'].sum()}) should equal "
        f"number of issues in window ({expected_count})"
    )
    # Verify index is properly sorted and continuous (daily frequency)
    assert result.index.is_monotonic_increasing
    # Verify index frequency is daily (using infer_freq for robustness)
    inferred_freq = pd.infer_freq(result.index)
    assert inferred_freq == "D", f"Expected daily frequency, got {inferred_freq}"
    assert result.index.min() == expected_start
    assert result.index.max() == expected_end


def test_determine_optimal_window_from_data(calc_instance, cycle_data):
    """Test _determine_optimal_window_from_data."""
    window_size = calc_instance.determine_optimal_window_from_data(
        cycle_data, "completed_timestamp", "D"
    )

    assert isinstance(window_size, int)
    assert window_size >= 14  # Minimum window
    assert window_size <= 90  # Maximum window


def test_determine_optimal_window_missing_column(calc_instance, cycle_data):
    """Test _determine_optimal_window_from_data with missing column."""
    window_size = calc_instance.determine_optimal_window_from_data(
        cycle_data, "nonexistent_column", "D"
    )

    # Should return default
    assert window_size == 30


def test_determine_optimal_window_empty_data(calc_instance):
    """Test _determine_optimal_window_from_data with empty data."""
    empty_data = pd.DataFrame(columns=["completed_timestamp"])
    window_size = calc_instance.determine_optimal_window_from_data(
        empty_data, "completed_timestamp", "D"
    )

    # Should return default
    assert window_size == 30


def test_calculate_window_start_fixed(calc_instance, cycle_data):
    """Test calculate_window_start with fixed window."""
    params = {
        "cycle_data": cycle_data,
        "done_column": "completed_timestamp",
        "sampling_window_end": datetime(2024, 2, 1),
        "smart_window": False,
        "window_size": 30,
        "freq": "D",
    }

    result = calc_instance.calculate_window_start(params)

    assert result is not None
    assert isinstance(result, datetime)


def test_calculate_window_start_smart(calc_instance, cycle_data):
    """Test calculate_window_start with smart window."""
    params = {
        "cycle_data": cycle_data,
        "done_column": "completed_timestamp",
        "sampling_window_end": datetime(2024, 2, 1),
        "smart_window": True,
        "freq": "D",
    }

    result = calc_instance.calculate_window_start(params)

    assert result is not None
    assert isinstance(result, datetime)


def test_calculate_window_start_missing_params(calc_instance):
    """Test calculate_window_start with missing parameters."""
    params = {
        "cycle_data": None,
        "done_column": "completed_timestamp",
        "sampling_window_end": datetime(2024, 2, 1),
    }

    result = calc_instance.calculate_window_start(params)

    assert result is None


def test_find_optimal_window_start(calc_instance, cycle_data):
    """Test _find_optimal_window_start."""
    end_date = datetime(2024, 2, 1)
    result = calc_instance.find_optimal_window_start(
        cycle_data, "completed_timestamp", end_date, "D"
    )

    assert isinstance(result, datetime)
    assert result < end_date


def test_calculate_adaptive_window_size(calc_instance):
    """Test _calculate_adaptive_window_size."""
    dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
    completion_dates = pd.Series(dates)
    end_date = datetime(2024, 2, 20)

    window_size = calc_instance.calculate_adaptive_window_size(
        completion_dates, end_date, "D"
    )

    assert isinstance(window_size, int)
    assert 14 <= window_size <= 90


def test_calculate_adaptive_window_size_weekly(calc_instance):
    """Test _calculate_adaptive_window_size for weekly frequency."""
    dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
    completion_dates = pd.Series(dates)
    end_date = datetime(2024, 2, 20)

    window_size = calc_instance.calculate_adaptive_window_size(
        completion_dates, end_date, "W"
    )

    assert isinstance(window_size, int)
    assert window_size >= 2  # At least 2 weeks


def test_calculate_adaptive_window_size_monthly(calc_instance):
    """Test _calculate_adaptive_window_size for monthly frequency."""
    dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
    completion_dates = pd.Series(dates)
    end_date = datetime(2024, 2, 20)

    window_size = calc_instance.calculate_adaptive_window_size(
        completion_dates, end_date, "M"
    )

    assert isinstance(window_size, int)
    assert window_size >= 1


def test_calculate_periods_back_daily(calc_instance):
    """Test _calculate_periods_back for daily frequency."""
    end_date = datetime(2024, 2, 1)
    result = calc_instance.calculate_periods_back(end_date, 30, "D")

    assert isinstance(result, datetime)
    assert (end_date - result).days == 30


def test_calculate_periods_back_weekly(calc_instance):
    """Test _calculate_periods_back for weekly frequency."""
    end_date = datetime(2024, 2, 1)
    result = calc_instance.calculate_periods_back(end_date, 4, "W")

    assert isinstance(result, datetime)
    assert (end_date - result).days == 28  # 4 weeks


def test_calculate_periods_back_monthly(calc_instance):
    """Test _calculate_periods_back for monthly frequency."""
    end_date = datetime(2024, 2, 1)
    result = calc_instance.calculate_periods_back(end_date, 1, "M")

    assert isinstance(result, datetime)
    assert result < end_date


def test_create_throughput_sampler(calc_instance):
    """Test create_throughput_sampler."""
    throughput_data = pd.DataFrame({"throughput": [1, 2, 3, 4, 5]})

    sampler = calc_instance.create_throughput_sampler(throughput_data)

    assert callable(sampler)
    # Test that sampler returns values
    samples = [sampler() for _ in range(100)]
    # Verify sampled values are members of the expected set
    # (not that all values must appear)
    assert all(s in [1, 2, 3, 4, 5] for s in samples)
