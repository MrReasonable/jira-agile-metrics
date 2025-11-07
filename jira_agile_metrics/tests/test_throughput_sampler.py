"""Tests for throughput sampler creation and functionality."""

import pandas as pd

from jira_agile_metrics.utils import create_throughput_sampler


def test_create_throughput_sampler_with_count_column():
    """Test throughput sampler works with 'count' column (actual format)."""
    # Create throughput data with 'count' column
    # (as returned by calculate_throughput)
    throughput_data = pd.DataFrame(
        {"count": [1, 2, 3, 2, 1, 4, 3, 2]},
        index=pd.date_range("2024-01-01", periods=8, freq="D"),
    )

    sampler = create_throughput_sampler(throughput_data)
    assert sampler is not None
    assert callable(sampler)

    # Sample multiple times to verify it works
    samples = [sampler() for _ in range(100)]
    assert all(s >= 0 for s in samples), "All samples should be non-negative"
    assert any(s > 0 for s in samples), "At least some samples should be positive"
    assert all(s in [1, 2, 3, 4] for s in samples), "Samples should come from data"


def test_create_throughput_sampler_with_throughput_column():
    """Test throughput sampler works with 'throughput' column (compatibility)."""
    # Create throughput data with 'throughput' column (alternative format)
    throughput_data = pd.DataFrame(
        {"throughput": [1, 2, 3, 2, 1, 4, 3, 2]},
        index=pd.date_range("2024-01-01", periods=8, freq="D"),
    )

    sampler = create_throughput_sampler(throughput_data)
    assert sampler is not None
    assert callable(sampler)

    # Sample multiple times to verify it works
    samples = [sampler() for _ in range(100)]
    assert all(s >= 0 for s in samples), "All samples should be non-negative"
    assert any(s > 0 for s in samples), "At least some samples should be positive"
    # Verify samples are drawn from input data
    unique_values = set(throughput_data["throughput"])
    assert set(samples).issubset(unique_values), (
        "Samples should only contain values from input data"
    )


def test_create_throughput_sampler_empty_data():
    """Test that throughput sampler handles empty data gracefully."""
    throughput_data = pd.DataFrame({"count": []}, index=pd.DatetimeIndex([]))

    sampler = create_throughput_sampler(throughput_data)
    assert sampler is not None
    assert callable(sampler)

    # Empty data should return 0
    assert sampler() == 0
    assert sampler() == 0


def test_create_throughput_sampler_none_data():
    """Test that throughput sampler handles None data gracefully."""
    sampler = create_throughput_sampler(None)
    assert sampler is not None
    assert callable(sampler)

    # None data should return 0
    assert sampler() == 0


def test_create_throughput_sampler_missing_column():
    """Test that throughput sampler handles missing column gracefully."""
    # DataFrame with wrong column name
    throughput_data = pd.DataFrame(
        {"wrong_column": [1, 2, 3]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )

    sampler = create_throughput_sampler(throughput_data)
    assert sampler is not None
    assert callable(sampler)

    # Missing column should return 0 (fallback behavior)
    assert sampler() == 0
