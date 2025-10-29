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
    calculator = ThroughputCalculator(query_manager, settings, test_fixture)

    data = calculator.run()

    assert data.to_dict("records") == [
        {"count": 1},
        {"count": 0},
        {"count": 1},
    ]


def test_calculate_throughput_with_wider_window(query_manager, settings, test_fixture):
    """Test throughput calculation with wider window."""
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
