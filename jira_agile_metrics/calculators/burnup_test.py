"""Tests for burnup calculator functionality in Jira Agile Metrics.

This module contains unit tests for the burnup calculator.
"""

import pytest
from pandas import DataFrame

from ..utils import extend_dict
from .burnup import BurnupCalculator
from .cfd import CFDCalculator
from .cfd_test import assert_cfd_timestamp_index


@pytest.fixture(name="settings")
def fixture_settings(base_minimal_settings):
    """Provide settings fixture for burnup tests."""
    return extend_dict(base_minimal_settings, {})


@pytest.fixture(name="query_manager")
def fixture_query_manager(minimal_query_manager):
    """Provide query manager fixture for burnup tests."""
    return minimal_query_manager


@pytest.fixture(name="results")
def fixture_results(minimal_cfd_results):
    """Provide results fixture for burnup tests."""
    return extend_dict(minimal_cfd_results, {})


def test_empty(query_manager, settings, base_cfd_columns):
    """Test burnup calculator with empty data."""
    results = {CFDCalculator: DataFrame([], columns=base_cfd_columns, index=[])}

    calculator = BurnupCalculator(query_manager, settings, results)

    data = calculator.run()
    assert len(data.index) == 0


def test_columns(query_manager, settings, results):
    """Test burnup calculator column structure."""
    calculator = BurnupCalculator(query_manager, settings, results)

    data = calculator.run()

    assert list(data.columns) == ["Backlog", "Done"]


def test_calculate_burnup(query_manager, settings, results):
    """Test burnup calculation functionality."""
    calculator = BurnupCalculator(query_manager, settings, results)

    data = calculator.run()

    assert_cfd_timestamp_index(data)

    assert data.to_dict("records") == [
        {"Backlog": 1.0, "Done": 0.0},
        {"Backlog": 2.0, "Done": 0.0},
        {"Backlog": 3.0, "Done": 0.0},
        {"Backlog": 4.0, "Done": 0.0},
        {"Backlog": 4.0, "Done": 0.0},
        {"Backlog": 4.0, "Done": 1.0},
        {"Backlog": 4.0, "Done": 1.0},
    ]


def test_calculate_burnup_with_different_columns(query_manager, settings, results):
    """Test burnup calculation with different column configuration."""
    settings.update({"backlog_column": "Committed", "done_column": "Test"})

    calculator = BurnupCalculator(query_manager, settings, results)

    data = calculator.run()

    assert_cfd_timestamp_index(data)

    assert data.to_dict("records") == [
        {"Committed": 0.0, "Test": 0.0},
        {"Committed": 0.0, "Test": 0.0},
        {"Committed": 2.0, "Test": 0.0},
        {"Committed": 3.0, "Test": 0.0},
        {"Committed": 3.0, "Test": 1.0},
        {"Committed": 3.0, "Test": 1.0},
        {"Committed": 3.0, "Test": 1.0},
    ]
