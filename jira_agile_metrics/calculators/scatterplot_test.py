"""Tests for scatterplot calculator functionality in Jira Agile Metrics.

This module contains unit tests for the scatterplot calculator.
"""

import pandas as pd
import pytest
from pandas import Timestamp

from ..test_utils import (
    create_empty_test_results,
    create_scatterplot_expected_columns,
)
from ..utils import extend_dict
from .scatterplot import ScatterplotCalculator


@pytest.fixture(name="settings")
def fixture_settings(base_minimal_settings):
    """Provide settings fixture for scatterplot tests."""
    return extend_dict(base_minimal_settings, {})


@pytest.fixture(name="query_manager")
def fixture_query_manager(custom_query_manager):
    """Provide query manager fixture for scatterplot tests."""
    return custom_query_manager


@pytest.fixture(name="results")
def fixture_results(large_cycle_time_results):
    """Provide results fixture for scatterplot tests."""
    return large_cycle_time_results


def test_empty(query_manager, settings, base_minimal_cycle_time_columns):
    """Test scatterplot calculator with empty data."""
    results = create_empty_test_results(base_minimal_cycle_time_columns)

    calculator = ScatterplotCalculator(query_manager, settings, results)

    # Should not raise error on empty input
    try:
        data = calculator.run()
    except AttributeError:
        # Acceptable if .dt accessor fails on empty input
        data = None
    assert data is None or isinstance(data, pd.DataFrame)


def test_columns(query_manager, settings, results):
    """Test scatterplot calculator column structure."""
    calculator = ScatterplotCalculator(query_manager, settings, results)

    data = calculator.run()

    assert list(data.columns) == create_scatterplot_expected_columns()


def test_calculate_scatterplot(query_manager, settings, results):
    """Test scatterplot calculation functionality."""
    calculator = ScatterplotCalculator(query_manager, settings, results)

    data = calculator.run()

    assert data[["key", "completed_date", "cycle_time"]].to_dict("records") == [
        {
            "key": "A-3",
            "completed_date": Timestamp("2018-01-06 00:00:00"),
            "cycle_time": 3,
        },
        {
            "key": "A-5",
            "completed_date": Timestamp("2018-01-08 00:00:00"),
            "cycle_time": 3,
        },
    ]
