"""Tests for netflow calculator functionality in Jira Agile Metrics.

This module contains unit tests for the netflow calculator.
"""

import pytest

from ..test_utils import (
    assert_extended_timestamp_index,
    create_common_cfd_results_fixture,
    create_common_empty_cfd_results,
)
from ..utils import extend_dict
from .netflow import NetFlowChartCalculator


@pytest.fixture(name="settings")
def fixture_settings(base_minimal_settings):
    """Provide settings fixture for netflow tests."""
    return extend_dict(base_minimal_settings, {"net_flow_frequency": "D"})


@pytest.fixture(name="query_manager")
def fixture_query_manager(minimal_query_manager):
    """Provide query manager fixture for netflow tests."""
    return minimal_query_manager


@pytest.fixture(name="results")
def fixture_results(query_manager, settings, large_cycle_time_results):
    """Create CFD results fixture for net flow testing."""
    return create_common_cfd_results_fixture(
        query_manager, settings, large_cycle_time_results
    )


def test_empty(query_manager, settings, base_minimal_cycle_time_columns):
    """Test net flow calculator with empty data."""
    results = create_common_empty_cfd_results(base_minimal_cycle_time_columns)

    calculator = NetFlowChartCalculator(query_manager, settings, results)

    data = calculator.run()
    assert list(data.columns) == [
        "Committed",
        "Done",
        "arrivals",
        "departures",
        "net_flow",
        "positive",
    ]
    assert len(data.index) == 0


def test_columns(query_manager, settings, results):
    """Test netflow calculator column structure."""
    calculator = NetFlowChartCalculator(query_manager, settings, results)

    data = calculator.run()
    assert list(data.columns) == [
        "Committed",
        "Done",
        "arrivals",
        "departures",
        "net_flow",
        "positive",
    ]


def test_calculate_net_flow(query_manager, settings, results):
    """Test netflow calculation functionality."""
    calculator = NetFlowChartCalculator(query_manager, settings, results)

    data = calculator.run()

    assert_extended_timestamp_index(data)

    assert data[["arrivals", "departures", "net_flow", "positive"]].to_dict(
        "records"
    ) == [
        {
            "arrivals": 0.0,
            "departures": 0.0,
            "net_flow": 0.0,
            "positive": True,
        },
        {
            "arrivals": 0.0,
            "departures": 0.0,
            "net_flow": 0.0,
            "positive": True,
        },
        {
            "arrivals": 2.0,
            "departures": 0.0,
            "net_flow": 2.0,
            "positive": True,
        },
        {
            "arrivals": 1.0,
            "departures": 0.0,
            "net_flow": 1.0,
            "positive": True,
        },
        {
            "arrivals": 1.0,
            "departures": 0.0,
            "net_flow": 1.0,
            "positive": True,
        },
        {
            "arrivals": 0.0,
            "departures": 1.0,
            "net_flow": -1.0,
            "positive": False,
        },
        {
            "arrivals": 0.0,
            "departures": 0.0,
            "net_flow": 0.0,
            "positive": True,
        },
        {
            "arrivals": 0.0,
            "departures": 1.0,
            "net_flow": -1.0,
            "positive": False,
        },
        {
            "arrivals": 0.0,
            "departures": 0.0,
            "net_flow": 0.0,
            "positive": True,
        },
    ]


def test_calculate_net_flow_different_columns(query_manager, settings, results):
    """Test netflow calculation with different column configuration."""
    settings.update(
        {
            "committed_column": "Build",
            "done_column": "Test",
        }
    )

    calculator = NetFlowChartCalculator(query_manager, settings, results)

    data = calculator.run()

    assert_extended_timestamp_index(data)

    assert data[["arrivals", "departures", "net_flow", "positive"]].to_dict(
        "records"
    ) == [
        {
            "arrivals": 0.0,
            "departures": 0.0,
            "net_flow": 0.0,
            "positive": True,
        },
        {
            "arrivals": 0.0,
            "departures": 0.0,
            "net_flow": 0.0,
            "positive": True,
        },
        {
            "arrivals": 0.0,
            "departures": 0.0,
            "net_flow": 0.0,
            "positive": True,
        },
        {
            "arrivals": 1.0,
            "departures": 0.0,
            "net_flow": 1.0,
            "positive": True,
        },
        {
            "arrivals": 0.0,
            "departures": 1.0,
            "net_flow": -1.0,
            "positive": False,
        },
        {
            "arrivals": 1.0,
            "departures": 0.0,
            "net_flow": 1.0,
            "positive": True,
        },
        {
            "arrivals": 0.0,
            "departures": 1.0,
            "net_flow": -1.0,
            "positive": False,
        },
        {
            "arrivals": 0.0,
            "departures": 0.0,
            "net_flow": 0.0,
            "positive": True,
        },
        {
            "arrivals": 0.0,
            "departures": 0.0,
            "net_flow": 0.0,
            "positive": True,
        },
    ]
