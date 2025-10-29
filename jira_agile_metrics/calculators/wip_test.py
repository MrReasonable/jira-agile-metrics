"""Tests for WIP calculator functionality in Jira Agile Metrics.

This module contains unit tests for the WIP calculator.
"""

import pytest

from ..test_utils import (
    assert_extended_timestamp_index,
    create_common_cfd_results_fixture,
    create_common_empty_cfd_results,
)
from ..utils import extend_dict
from .wip import WIPChartCalculator


@pytest.fixture(name="settings")
def fixture_settings(base_minimal_settings):
    """Provide settings fixture for WIP tests."""
    return extend_dict(base_minimal_settings, {})


@pytest.fixture(name="query_manager")
def fixture_query_manager(minimal_query_manager):
    """Provide query manager fixture for WIP tests."""
    return minimal_query_manager


@pytest.fixture(name="results")
def fixture_results(query_manager, settings, large_cycle_time_results):
    """Create CFD results fixture for WIP testing."""
    return create_common_cfd_results_fixture(
        query_manager, settings, large_cycle_time_results
    )


def test_empty(query_manager, settings, base_minimal_cycle_time_columns):
    """Test WIP calculator with empty data."""
    results = create_common_empty_cfd_results(base_minimal_cycle_time_columns)

    calculator = WIPChartCalculator(query_manager, settings, results)

    data = calculator.run()
    assert list(data.columns) == ["wip"]
    assert len(data.index) == 0


def test_columns(query_manager, settings, results):
    """Test WIP calculator column structure."""
    calculator = WIPChartCalculator(query_manager, settings, results)

    data = calculator.run()
    assert list(data.columns) == ["wip"]


def test_calculate_wip(query_manager, settings, results):
    """Test WIP calculation functionality."""
    calculator = WIPChartCalculator(query_manager, settings, results)

    data = calculator.run()

    assert_extended_timestamp_index(data)

    assert data.to_dict("records") == [
        {"wip": 0.0},
        {"wip": 0.0},
        {"wip": 2.0},
        {"wip": 3.0},
        {"wip": 4.0},
        {"wip": 3.0},
        {"wip": 3.0},
        {"wip": 2.0},
        {"wip": 2.0},
    ]


def test_calculate_wip_different_columns(query_manager, settings, results):
    """Test WIP calculation with different column structure."""
    settings.update(
        {
            "committed_column": "Build",
            "done_column": "Test",
        }
    )

    calculator = WIPChartCalculator(query_manager, settings, results)

    data = calculator.run()

    assert_extended_timestamp_index(data)

    assert data.to_dict("records") == [
        {"wip": 0.0},
        {"wip": 0.0},
        {"wip": 0.0},
        {"wip": 1.0},
        {"wip": 0.0},
        {"wip": 1.0},
        {"wip": 0.0},
        {"wip": 0.0},
        {"wip": 0.0},
    ]
