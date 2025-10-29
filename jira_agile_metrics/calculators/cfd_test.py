"""Tests for CFD calculator functionality in Jira Agile Metrics.

This module contains unit tests for the CFD (Cumulative Flow Diagram) calculator.
"""

import pytest
from pandas import DataFrame

from ..test_data import COMMON_CFD_DATA
from ..test_utils import (
    EXTENDED_TIMESTAMP_INDEX,
    create_common_cycle_status_list,
)
from ..utils import extend_dict
from .cfd import CFDCalculator
from .cycletime import CycleTimeCalculator


def assert_cfd_timestamp_index(data):
    """Assert that data has the CFD timestamp index."""
    # Use first 7 elements of EXTENDED_TIMESTAMP_INDEX
    expected_index = EXTENDED_TIMESTAMP_INDEX[:7]
    assert list(data.index) == expected_index


@pytest.fixture(name="test_query_manager")
def query_manager(minimal_query_manager):
    """Test query manager fixture."""
    return minimal_query_manager


@pytest.fixture(name="test_settings")
def settings(base_minimal_settings):
    """Test settings fixture."""
    return extend_dict(base_minimal_settings, {})


@pytest.fixture(name="test_columns")
def columns(base_minimal_cycle_time_columns):
    """Test columns fixture."""
    return base_minimal_cycle_time_columns


def test_empty(test_query_manager, test_settings, test_columns):
    """Test CFD calculator with empty data."""
    results = {CycleTimeCalculator: DataFrame([], columns=test_columns)}

    calculator = CFDCalculator(test_query_manager, test_settings, results)

    data = calculator.run()
    assert len(data.index) == 0


def test_cfd_columns(
    test_query_manager, test_settings, base_minimal_cycle_time_results
):
    """Test CFD calculator column handling."""
    calculator = CFDCalculator(
        test_query_manager, test_settings, base_minimal_cycle_time_results
    )

    data = calculator.run()

    assert list(data.columns) == create_common_cycle_status_list()


def test_calculate_cfd(
    test_query_manager, test_settings, base_minimal_cycle_time_results
):
    """Test CFD calculation functionality."""
    calculator = CFDCalculator(
        test_query_manager, test_settings, base_minimal_cycle_time_results
    )

    data = calculator.run()

    assert_cfd_timestamp_index(data)

    assert data.to_dict("records") == COMMON_CFD_DATA
