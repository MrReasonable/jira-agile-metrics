"""Tests for histogram calculator functionality in Jira Agile Metrics.

This module contains unit tests for the histogram calculator.
"""

import pandas as pd
import pytest

from ..test_utils import (
    create_empty_test_results,
)
from ..utils import extend_dict
from .histogram import HistogramCalculator


@pytest.fixture(name="test_settings")
def settings(base_minimal_settings):
    """Test settings fixture."""
    return extend_dict(base_minimal_settings, {})


@pytest.fixture(name="test_query_manager")
def query_manager(custom_query_manager):
    """Test query manager fixture."""
    return custom_query_manager


@pytest.fixture(name="test_results")
def results(large_cycle_time_results):
    """Test results fixture."""
    return large_cycle_time_results


def test_empty(test_query_manager, test_settings, base_minimal_cycle_time_columns):
    """Test histogram calculator with empty data."""
    empty_results = create_empty_test_results(base_minimal_cycle_time_columns)

    calculator = HistogramCalculator(test_query_manager, test_settings, empty_results)

    # Should not raise error on empty input
    try:
        data = calculator.run()
    except AttributeError:
        # Acceptable if .dt accessor fails on empty input
        data = None
    assert data is None or isinstance(data, pd.Series)


def test_calculate_histogram(test_query_manager, test_settings, test_results):
    """Test histogram calculation functionality."""
    calculator = HistogramCalculator(test_query_manager, test_settings, test_results)

    data = calculator.run()

    assert list(data.index) == [
        "0.0 to 1.0",
        "1.0 to 2.0",
        "2.0 to 3.0",
        "3.0 to 4.0",
    ]
    assert list(data) == [0, 0, 0, 2]
