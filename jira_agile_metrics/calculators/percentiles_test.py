"""Tests for percentiles calculator functionality in Jira Agile Metrics.

This module contains unit tests for the percentiles calculator.
"""

import math

import pytest
from pandas import Timedelta

from ..test_utils import (
    create_empty_test_results,
)
from ..utils import extend_dict
from .percentiles import PercentilesCalculator


@pytest.fixture(name="settings")
def fixture_settings(base_minimal_settings):
    """Provide settings fixture for percentiles tests."""
    return extend_dict(base_minimal_settings, {"quantiles": [0.1, 0.5, 0.9]})


@pytest.fixture(name="query_manager")
def fixture_query_manager(custom_query_manager):
    """Provide query manager fixture for percentiles tests."""
    return custom_query_manager


@pytest.fixture(name="results")
def fixture_results(base_minimal_cycle_time_results):
    """Provide results fixture for percentiles tests."""
    return base_minimal_cycle_time_results


def test_empty(query_manager, settings, base_minimal_cycle_time_columns):
    """Test percentiles calculator with empty data."""
    results = create_empty_test_results(base_minimal_cycle_time_columns)

    calculator = PercentilesCalculator(query_manager, settings, results)

    data = calculator.run()

    assert list(data.index) == [0.1, 0.5, 0.9]
    assert math.isnan(list(data)[0])
    assert math.isnan(list(data)[1])
    assert math.isnan(list(data)[2])


def test_calculate_percentiles(query_manager, settings, results):
    """Test percentiles calculation functionality."""
    calculator = PercentilesCalculator(query_manager, settings, results)

    data = calculator.run()

    assert list(data.index) == [0.1, 0.5, 0.9]
    assert list(data) == [
        Timedelta("3 days 00:00:00"),
        Timedelta("3 days 00:00:00"),
        Timedelta("3 days 00:00:00"),
    ]
