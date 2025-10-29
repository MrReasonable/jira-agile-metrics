"""Tests for impediments calculator functionality in Jira Agile Metrics.

This module contains unit tests for the impediments calculator.
"""

import pytest
from pandas import DataFrame, NaT, Timestamp

from ..test_data_factory import create_impediments_cycle_time_results
from ..utils import extend_dict
from .cycletime import CycleTimeCalculator
from .impediments import ImpedimentsCalculator


def _ts(datestring, timestring="00:00:00"):
    return Timestamp(f"{datestring} {timestring}")


@pytest.fixture(name="query_manager")
def fixture_query_manager(minimal_query_manager):
    """Provide query manager fixture for impediments tests."""
    return minimal_query_manager


@pytest.fixture(name="settings")
def fixture_settings(base_minimal_settings):
    """Provide settings fixture for impediments tests."""
    return extend_dict(
        base_minimal_settings,
        {
            "impediments_data": "impediments.csv",
            "impediments_chart": "impediments.png",
            "impediments_days_chart": "impediments-days.png",
            "impediments_status_chart": "impediments-status.png",
            "impediments_status_days_chart": "impediments-status-days.png",
        },
    )


@pytest.fixture(name="columns")
def fixture_columns(base_minimal_cycle_time_columns):
    """Provide columns fixture for impediments tests."""
    return base_minimal_cycle_time_columns


@pytest.fixture(name="cycle_time_results")
def fixture_cycle_time_results(base_minimal_cycle_time_columns):
    """A results dict mimicing a minimal result
    from the CycleTimeCalculator."""
    return create_impediments_cycle_time_results(base_minimal_cycle_time_columns)


def test_only_runs_if_charts_set(query_manager, settings, cycle_time_results):
    """Test that impediments calculator only runs when charts are configured."""
    test_settings = extend_dict(
        settings,
        {
            "impediments_data": None,
            "impediments_chart": None,
            "impediments_days_chart": None,
            "impediments_status_chart": None,
            "impediments_status_days_chart": None,
        },
    )

    calculator = ImpedimentsCalculator(query_manager, test_settings, cycle_time_results)
    data = calculator.run()
    assert data is None

    test_settings = extend_dict(
        settings,
        {
            "impediments_data": "impediments.csv",
            "impediments_chart": None,
            "impediments_days_chart": None,
            "impediments_status_chart": None,
            "impediments_status_days_chart": None,
        },
    )

    calculator = ImpedimentsCalculator(query_manager, test_settings, cycle_time_results)
    data = calculator.run()
    assert data is not None

    test_settings = extend_dict(
        settings,
        {
            "impediments_data": None,
            "impediments_chart": "impediments.png",
            "impediments_days_chart": None,
            "impediments_status_chart": None,
            "impediments_status_days_chart": None,
        },
    )

    calculator = ImpedimentsCalculator(query_manager, test_settings, cycle_time_results)
    data = calculator.run()
    assert data is not None

    test_settings = extend_dict(
        settings,
        {
            "impediments_data": None,
            "impediments_chart": None,
            "impediments_days_chart": "days.png",
            "impediments_status_chart": None,
            "impediments_status_days_chart": None,
        },
    )

    calculator = ImpedimentsCalculator(query_manager, test_settings, cycle_time_results)
    data = calculator.run()
    assert data is not None

    test_settings = extend_dict(
        settings,
        {
            "impediments_data": None,
            "impediments_chart": None,
            "impediments_days_chart": None,
            "impediments_status_chart": "status.png",
            "impediments_status_days_chart": None,
        },
    )

    calculator = ImpedimentsCalculator(query_manager, test_settings, cycle_time_results)
    data = calculator.run()
    assert data is not None

    test_settings = extend_dict(
        settings,
        {
            "impediments_data": None,
            "impediments_chart": None,
            "impediments_days_chart": None,
            "impediments_status_chart": None,
            "impediments_status_days_chart": "status-days.png",
        },
    )

    calculator = ImpedimentsCalculator(query_manager, test_settings, cycle_time_results)
    data = calculator.run()
    assert data is not None


def test_empty(query_manager, settings, columns):
    """Test impediments calculator with empty data."""
    results = {CycleTimeCalculator: DataFrame([], columns=columns)}

    calculator = ImpedimentsCalculator(query_manager, settings, results)

    data = calculator.run()
    assert len(data.index) == 0


def test_columns(query_manager, settings, cycle_time_results):
    """Test impediments calculator column structure."""
    calculator = ImpedimentsCalculator(query_manager, settings, cycle_time_results)

    data = calculator.run()

    assert list(data.columns) == ["key", "status", "flag", "start", "end"]


def test_calculate_impediments(query_manager, settings, cycle_time_results):
    """Test impediments calculator functionality."""
    calculator = ImpedimentsCalculator(query_manager, settings, cycle_time_results)

    data = calculator.run()

    assert data.to_dict("records") == [
        {
            "key": "A-2",
            "status": "Committed",
            "flag": "Impediment",
            "start": _ts("2018-01-10"),
            "end": _ts("2018-01-12"),
        },
        {
            "key": "A-3",
            "status": "Build",
            "flag": "Impediment",
            "start": _ts("2018-01-04"),
            "end": _ts("2018-01-05"),
        },
        {
            "key": "A-4",
            "status": "Committed",
            "flag": "Awaiting input",
            "start": _ts("2018-01-05"),
            "end": NaT,
        },
    ]


def test_different_backlog_column(query_manager, settings, cycle_time_results):
    """Test impediments calculator with different backlog column."""
    settings = extend_dict(
        settings,
        {
            "backlog_column": "Committed",
            "committed_column": "Build",
        },
    )
    calculator = ImpedimentsCalculator(query_manager, settings, cycle_time_results)

    data = calculator.run()

    assert data.to_dict("records") == [
        {
            "key": "A-3",
            "status": "Build",
            "flag": "Impediment",
            "start": _ts("2018-01-04"),
            "end": _ts("2018-01-05"),
        },
    ]


def test_different_done_column(query_manager, settings, cycle_time_results):
    """Test impediments calculator with different done column."""
    settings = extend_dict(
        settings,
        {
            "done_column": "Build",
        },
    )
    calculator = ImpedimentsCalculator(query_manager, settings, cycle_time_results)

    data = calculator.run()

    assert data.to_dict("records") == [
        {
            "key": "A-2",
            "status": "Committed",
            "flag": "Impediment",
            "start": _ts("2018-01-10"),
            "end": _ts("2018-01-12"),
        },
        {
            "key": "A-4",
            "status": "Committed",
            "flag": "Awaiting input",
            "start": _ts("2018-01-05"),
            "end": NaT,
        },
    ]
