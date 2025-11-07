"""Tests for WIP calculator functionality in Jira Agile Metrics.

This module contains unit tests for the WIP calculator.
"""

import os

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


def test_write_chart(query_manager, settings, results, tmp_path):
    """Test writing WIP chart."""
    output_file = str(tmp_path / "wip.png")
    settings_extended = extend_dict(
        settings,
        {
            "wip_chart": output_file,
            "wip_chart_title": "Test WIP Chart",
            "wip_frequency": "D",
            "wip_window": None,
            "date_format": "%Y-%m-%d",
        },
    )

    calculator = WIPChartCalculator(query_manager, settings_extended, results)
    result = calculator.run()
    results[WIPChartCalculator] = result
    calculator.write()

    assert os.path.exists(output_file)


def test_write_chart_with_window(query_manager, settings, results, tmp_path):
    """Test writing WIP chart with window filtering."""
    output_file = str(tmp_path / "wip.png")
    settings_extended = extend_dict(
        settings,
        {
            "wip_chart": output_file,
            "wip_chart_title": "Test WIP Chart",
            "wip_frequency": "D",
            "wip_window": 5,
            "date_format": "%Y-%m-%d",
        },
    )

    calculator = WIPChartCalculator(query_manager, settings_extended, results)
    result = calculator.run()
    results[WIPChartCalculator] = result
    calculator.write()

    assert os.path.exists(output_file)


def test_write_chart_empty_data(
    query_manager, settings, base_minimal_cycle_time_columns, tmp_path
):
    """Test writing WIP chart with empty data."""
    output_file = str(tmp_path / "wip.png")
    results = create_common_empty_cfd_results(base_minimal_cycle_time_columns)
    settings_extended = extend_dict(
        settings,
        {
            "wip_chart": output_file,
            "wip_frequency": "D",
            "wip_window": None,
            "date_format": "%Y-%m-%d",
        },
    )

    calculator = WIPChartCalculator(query_manager, settings_extended, results)
    result = calculator.run()
    results[WIPChartCalculator] = result
    calculator.write()

    # Chart should not be created with empty data
    assert not os.path.exists(output_file)


def test_write_no_output_file(query_manager, settings, results):
    """Test write() when no output file is specified."""
    settings_extended = extend_dict(
        settings,
        {
            "wip_chart": None,
            "wip_frequency": "D",
            "wip_window": None,
        },
    )

    calculator = WIPChartCalculator(query_manager, settings_extended, results)
    calculator.run()
    # Should not raise an error
    calculator.write()
