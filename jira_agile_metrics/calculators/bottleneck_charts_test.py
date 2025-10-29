"""Tests for bottleneck charts calculator functionality in Jira Agile Metrics.

This module contains unit tests for the bottleneck charts calculator.
"""

import pandas as pd
import pytest
from pandas import Timestamp

from ..querymanager import QueryManager
from ..test_classes import FauxJIRA as JIRA
from ..test_utils import (
    create_common_cycle_status_list,
    create_common_done_issue,
)
from ..utils import extend_dict
from .bottleneck_charts_calculator import (
    BottleneckChartsCalculator,
    calculate_column_durations,
    calculate_column_durations_per_column,
)
from .cycletime import CycleTimeCalculator


@pytest.fixture(name="test_jira")
def jira(base_custom_fields):
    """Test JIRA instance with custom fields."""
    return JIRA(
        fields=base_custom_fields,
        issues=[
            create_common_done_issue(
                "A-1",
                resolutiondate=Timestamp("2018-01-06 00:00:00"),
                created="2018-01-01 01:01:01",
            ),
            create_common_done_issue(
                "A-2",
                resolutiondate=Timestamp("2018-01-08 00:00:00"),
                created="2018-01-03 01:01:01",
            ),
        ],
    )


@pytest.fixture(name="query_manager")
def fixture_query_manager(test_jira):
    """Provide query manager fixture for bottleneck charts tests."""
    return QueryManager(
        test_jira,
        {
            "attributes": {},
            "known_values": {"Release": ["R1", "R3"]},
            "max_results": None,
            "verbose": False,
            "cycle": create_common_cycle_status_list(),
            "query_attribute": None,
            "queries": [{"jql": "(filter=123)", "value": None}],
            "backlog_column": "Backlog",
            "committed_column": "Next",
            "done_column": "Done",
        },
    )


@pytest.fixture(name="settings")
def fixture_settings(base_minimal_settings):
    """Provide settings fixture for bottleneck charts tests."""
    return extend_dict(
        base_minimal_settings,
        {
            "negative_duration_handling": "zero",
        },
    )


@pytest.fixture(name="cycle_time_results")
def fixture_cycle_time_results(query_manager, settings):
    """Create cycle time results fixture."""
    return {CycleTimeCalculator: CycleTimeCalculator(query_manager, settings, {}).run()}


def test_calculate_column_durations_empty():
    """Test calculate_column_durations with empty data."""
    cycle_data = pd.DataFrame(columns=["key"])
    cycle_names = ["Backlog", "Next", "Build", "Done"]

    result = calculate_column_durations(cycle_data, cycle_names)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    expected_cols = ["Backlog→Next", "Next→Build", "Build→Done"]
    assert list(result.columns) == expected_cols


def test_calculate_column_durations(cycle_time_results):
    """Test calculate_column_durations functionality."""
    cycle_data = cycle_time_results[CycleTimeCalculator]
    cycle_names = ["Backlog", "Next", "Build", "Done"]

    result = calculate_column_durations(cycle_data, cycle_names)

    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert "A-1" in result.index
    assert "A-2" in result.index

    # Check that the result has the expected transition columns
    expected_cols = ["Backlog→Next", "Next→Build", "Build→Done"]
    assert set(result.columns) == set(expected_cols)


def test_calculate_column_durations_per_column_empty():
    """Test calculate_column_durations_per_column with empty data."""
    cycle_data = pd.DataFrame(columns=["key"])
    cycle_names = ["Backlog", "Next", "Build", "Done"]

    result = calculate_column_durations_per_column(cycle_data, cycle_names)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    # Should have columns except the last one
    assert list(result.columns) == ["Backlog", "Next", "Build"]


def test_calculate_column_durations_per_column(cycle_time_results):
    """Test calculate_column_durations_per_column functionality."""
    cycle_data = cycle_time_results[CycleTimeCalculator]
    cycle_names = ["Backlog", "Next", "Build", "Done"]

    result = calculate_column_durations_per_column(cycle_data, cycle_names)

    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert "A-1" in result.index
    assert "A-2" in result.index

    # Should have columns except the last one
    assert set(result.columns) == set(["Backlog", "Next", "Build"])


def test_calculate_column_durations_per_column_negative_handling_zero():
    """Test negative duration handling with 'zero' option."""
    # Create test data with negative durations
    # A-1 goes backwards from Build to Next (negative Next->Build duration)
    cycle_data = pd.DataFrame(
        {
            "key": ["A-1"],
            "Backlog": [Timestamp("2018-01-01")],
            "Next": [Timestamp("2018-01-03")],
            "Build": [Timestamp("2018-01-02")],  # Goes backwards from Next
            "Done": [Timestamp("2018-01-04")],
        }
    )

    cycle_names = ["Backlog", "Next", "Build", "Done"]
    result = calculate_column_durations_per_column(
        cycle_data, cycle_names, negative_duration_handling="zero"
    )

    # A-1 should have 0 for "Next" column duration due to negative duration
    assert result.loc["A-1", "Next"] == 0


def test_calculate_column_durations_per_column_negative_handling_nan():
    """Test negative duration handling with 'nan' option."""
    cycle_data = pd.DataFrame(
        {
            "key": ["A-1"],
            "Backlog": [Timestamp("2018-01-01")],
            "Next": [Timestamp("2018-01-03")],
            "Build": [Timestamp("2018-01-02")],  # Goes backwards
            "Done": [Timestamp("2018-01-04")],
        }
    )

    cycle_names = ["Backlog", "Next", "Build", "Done"]
    result = calculate_column_durations_per_column(
        cycle_data, cycle_names, negative_duration_handling="nan"
    )

    # Should be NaN for negative duration (Next column has negative Build date)
    assert pd.isna(result.loc["A-1", "Next"])


def test_calculate_column_durations_per_column_negative_handling_abs():
    """Test negative duration handling with 'abs' option."""
    cycle_data = pd.DataFrame(
        {
            "key": ["A-1"],
            "Backlog": [Timestamp("2018-01-01")],
            "Next": [Timestamp("2018-01-05")],
            "Build": [Timestamp("2018-01-04")],  # Goes backwards
            "Done": [Timestamp("2018-01-06")],
        }
    )

    cycle_names = ["Backlog", "Next", "Build", "Done"]
    result = calculate_column_durations_per_column(
        cycle_data, cycle_names, negative_duration_handling="abs"
    )

    # Should be exact absolute value of the backwards duration: 1 day
    # Next->Build goes backwards (Next is 2018-01-05, Build is 2018-01-04)
    # So Next column duration is -1 day, which with abs becomes 1 day
    expected_abs_duration = pd.Timedelta(days=1)
    value = result.loc["A-1", "Next"]
    if isinstance(value, pd.Timedelta):
        assert value == expected_abs_duration
    else:
        # Function returns days as numeric (from .dt.days)
        assert value == expected_abs_duration.days


def test_bottleneck_calculator_empty(query_manager, settings):
    """Test bottleneck calculator with empty cycle time data."""
    empty_data = pd.DataFrame(columns=["key", "Backlog", "Next", "Build", "Done"])
    empty_results = {CycleTimeCalculator: empty_data}

    calculator = BottleneckChartsCalculator(query_manager, settings, empty_results)

    result = calculator.run()

    assert isinstance(result, dict)
    assert "transitions" in result
    assert "columns" in result
    assert len(result["transitions"]) == 0
    assert len(result["columns"]) == 0


def test_bottleneck_calculator_run(cycle_time_results, query_manager, settings):
    """Test bottleneck calculator run functionality."""
    calculator = BottleneckChartsCalculator(query_manager, settings, cycle_time_results)

    result = calculator.run()

    assert isinstance(result, dict)
    assert "transitions" in result
    assert "columns" in result
    assert isinstance(result["transitions"], pd.DataFrame)
    assert isinstance(result["columns"], pd.DataFrame)
    assert len(result["transitions"]) > 0
    assert len(result["columns"]) > 0


def test_bottleneck_calculator_different_negative_handling(
    cycle_time_results, query_manager, settings
):
    """Test bottleneck calculator with different negative duration handling."""
    settings.update({"negative_duration_handling": "nan"})

    calculator = BottleneckChartsCalculator(query_manager, settings, cycle_time_results)

    result = calculator.run()

    assert isinstance(result["columns"], pd.DataFrame)
    assert len(result["columns"]) >= 0


def test_calculate_column_durations_per_column_invalid_negative_handling():
    """Test that invalid negative_duration_handling raises ValueError."""
    cycle_data = pd.DataFrame(
        {
            "key": ["A-1"],
            "Backlog": [Timestamp("2018-01-01")],
            "Next": [Timestamp("2018-01-03")],
            "Build": [Timestamp("2018-01-02")],
            "Done": [Timestamp("2018-01-04")],
        }
    )

    cycle_names = ["Backlog", "Next", "Build", "Done"]

    with pytest.raises(ValueError, match="must be one of"):
        calculate_column_durations_per_column(
            cycle_data, cycle_names, negative_duration_handling="invalid_option"
        )
