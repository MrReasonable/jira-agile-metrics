"""Tests for throughput calculator functionality in Jira Agile Metrics.

This module contains unit tests for the throughput calculator.
"""

import pytest

from ..test_utils import create_empty_test_results
from ..utils import extend_dict
from .throughput import ThroughputCalculator


@pytest.fixture(name="settings")
def throughput_settings(base_minimal_settings):
    """Provide settings fixture for throughput tests."""
    return extend_dict(
        base_minimal_settings,
        {
            "throughput_frequency": "D",
            "throughput_window": None,
        },
    )


@pytest.fixture(name="query_manager")
def throughput_query_manager(custom_query_manager):
    """Provide query manager fixture for throughput tests."""
    return custom_query_manager


@pytest.fixture(name="test_fixture")
def throughput_test_fixture(large_cycle_time_results):
    """Provide results fixture for throughput tests."""
    return large_cycle_time_results


def test_empty(query_manager, settings, base_minimal_cycle_time_columns):
    """Test throughput calculator with empty data."""
    test_results = create_empty_test_results(base_minimal_cycle_time_columns)
    calculator = ThroughputCalculator(query_manager, settings, test_results)

    data = calculator.run()
    assert list(data.columns) == ["count"]
    assert len(data.index) == 0


def test_columns(query_manager, settings, test_fixture):
    """Test throughput calculator column structure."""
    calculator = ThroughputCalculator(query_manager, settings, test_fixture)

    data = calculator.run()

    assert list(data.columns) == ["count"]


def test_calculate_throughput(query_manager, settings, test_fixture):
    """Test throughput calculation functionality."""
    # Expected counts based on large_cycle_time_results fixture (extended_data):
    # The fixture contains 5 issues (A-1 through A-5), but only A-3 and A-5 have
    # completed_timestamp values (2018-01-06 and 2018-01-08 respectively).
    # With no window specified, the date range spans from earliest to latest
    # completion (2018-01-06 to 2018-01-08), creating 3 daily periods:
    # 2018-01-06: 1 completion (A-3), 2018-01-07: 0 completions,
    # 2018-01-08: 1 completion (A-5)
    calculator = ThroughputCalculator(query_manager, settings, test_fixture)

    data = calculator.run()

    assert data.to_dict("records") == [
        {"count": 1},
        {"count": 0},
        {"count": 1},
    ]


def test_calculate_throughput_with_wider_window(query_manager, settings, test_fixture):
    """Test throughput calculation with wider window."""
    # Expected counts based on large_cycle_time_results fixture (extended_data):
    # With window=5 and frequency='D', the 5-day window ending on the latest
    # completion (2018-01-08) starts 4 days earlier (2018-01-04),
    # creating 5 daily periods:
    # 2018-01-04: 0 completions, 2018-01-05: 0 completions,
    # 2018-01-06: 1 completion (A-3), 2018-01-07: 0 completions,
    # 2018-01-08: 1 completion (A-5)
    test_settings = extend_dict(
        settings,
        {
            "throughput_frequency": "D",
            "throughput_window": 5,
        },
    )

    calculator = ThroughputCalculator(query_manager, test_settings, test_fixture)

    data = calculator.run()

    assert data.to_dict("records") == [
        {"count": 0.0},
        {"count": 0.0},
        {"count": 1.0},
        {"count": 0.0},
        {"count": 1.0},
    ]


def test_calculate_throughput_with_narrower_window(
    query_manager, settings, test_fixture
):
    """Test throughput calculation with narrower window."""
    # Expected counts based on large_cycle_time_results fixture (extended_data):
    # With window=2 and frequency='D', the 2-day window ending on the latest completion
    # (2018-01-08) starts 1 day earlier (2018-01-07), creating 2 daily periods:
    # 2018-01-07: 0 completions, 2018-01-08: 1 completion (A-5)
    # Note: A-3 (completed 2018-01-06) falls outside this narrower window
    test_settings = extend_dict(
        settings,
        {
            "throughput_frequency": "D",
            "throughput_window": 2,
        },
    )

    calculator = ThroughputCalculator(query_manager, test_settings, test_fixture)

    data = calculator.run()

    assert data.to_dict("records") == [{"count": 0}, {"count": 1}]
