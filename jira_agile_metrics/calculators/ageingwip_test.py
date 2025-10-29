"""Tests for the ageing WIP chart calculator."""

import datetime

import pytest
from pandas import DataFrame

from ..conftest import FauxJIRA as JIRA
from ..querymanager import QueryManager
from ..test_classes import FauxChange as Change
from ..test_classes import FauxFieldValue as Value
from ..test_classes import FauxIssue as Issue
from ..utils import extend_dict
from .ageingwip import AgeingWIPChartCalculator
from .cycletime import CycleTimeCalculator


@pytest.fixture(name="settings")
def fixture_settings(base_minimal_settings):
    """Provide settings fixture for ageing WIP tests."""
    return extend_dict(
        base_minimal_settings,
        {
            "ageing_wip_chart": "ageingwip.png"
            # without a file to write the calculator will stop
        },
    )


@pytest.fixture(name="jira_with_skipped_columns")
def fixture_jira_with_skipped_columns(base_minimal_fields):
    """Provide JIRA fixture with skipped columns for ageing WIP tests."""
    return JIRA(
        fields=base_minimal_fields,
        issues=[
            Issue(
                "A-13",
                summary="No Gaps",
                issuetype=Value("Story", "story"),
                status=Value("Build", "build"),
                resolution=None,
                resolutiondate=None,
                created="2018-01-01 08:15:00",
                changes=[
                    Change(
                        "2018-01-02 08:15:00",
                        [
                            (
                                "status",
                                "Backlog",
                                "Next",
                            )
                        ],
                    ),
                    Change(
                        "2018-01-03 08:15:00",
                        [
                            (
                                "status",
                                "Next",
                                "Build",
                            )
                        ],
                    ),
                ],
            ),
            Issue(
                "A-14",
                summary="Gaps",
                issuetype=Value("Story", "story"),
                status=Value("Build", "build"),
                resolution=None,
                resolutiondate=None,
                created="2018-01-01 08:15:00",
                changes=[
                    Change(
                        "2018-01-02 08:15:00",
                        [
                            (
                                "status",
                                "Backlog",
                                "Build",
                            )
                        ],
                    ),  # skipping column Committed
                ],
            ),
            Issue(
                "A-15",
                summary="Gaps and withdrawn",
                issuetype=Value("Story", "story"),
                status=Value("Done", "done"),
                resolution=Value("Withdrawn", "withdrawn"),
                resolutiondate="2018-01-02 08:15:00",
                created="2018-01-01 08:15:00",
                changes=[
                    Change(
                        "2018-01-02 08:15:00",
                        [
                            (
                                "status",
                                "Backlog",
                                "Done",
                            ),
                            ("resolution", None, "Withdrawn"),
                        ],
                    ),  # skipping columns Committed, Build and Test
                ],
            ),
            Issue(
                "A-16",
                summary="Gap in first committed step",
                issuetype=Value("Story", "story"),
                status=Value("Build", "Build"),
                resolution=None,
                resolutiondate=None,
                created="2018-01-01 08:15:00",
                changes=[
                    Change(
                        "2018-01-03 08:15:00",
                        [
                            (
                                "status",
                                "Backlog",
                                "Build",
                            )
                        ],
                    ),  # skipping column Committed
                ],
            ),
        ],
    )


@pytest.fixture(name="query_manager")
def fixture_query_manager(minimal_query_manager):
    """Provide a query manager fixture for ageing WIP tests."""
    return minimal_query_manager


@pytest.fixture(name="results")
def fixture_results(large_cycle_time_results):
    """Provide results fixture for ageing WIP tests."""
    return extend_dict(large_cycle_time_results, {})


@pytest.fixture(name="today")
def fixture_today():
    """Provide today's date fixture for ageing WIP tests."""
    return datetime.date(2018, 1, 10)


@pytest.fixture(name="now")
def fixture_now(today):
    """Provide current datetime fixture for ageing WIP tests."""
    return datetime.datetime.combine(today, datetime.time(8, 30, 00))


def test_empty(query_manager, settings, base_minimal_cycle_time_columns, today):
    """Test ageing WIP calculator with empty data."""
    results = {
        CycleTimeCalculator: DataFrame(
            [], columns=base_minimal_cycle_time_columns, index=[]
        )
    }

    calculator = AgeingWIPChartCalculator(query_manager, settings, results)

    data = calculator.run(today)
    assert list(data.columns) == [
        "key",
        "summary",
        "status",
        "age",
        "Committed",
        "Build",
        "Test",
    ]
    assert len(data.index) == 0


def test_columns(query_manager, settings, results, today):
    """Test ageing WIP calculator column structure."""
    calculator = AgeingWIPChartCalculator(query_manager, settings, results)

    data = calculator.run(today)

    assert list(data.columns) == [
        "key",
        "summary",
        "status",
        "age",
        "Committed",
        "Build",
        "Test",
    ]


def test_calculate_ageing_wip(query_manager, settings, results, today):
    """Test ageing WIP calculation functionality."""
    calculator = AgeingWIPChartCalculator(query_manager, settings, results)

    data = calculator.run(today)

    assert data[["key", "status", "age"]].to_dict("records") == [
        {"key": "A-2", "status": "Committed", "age": 7.0},
        {"key": "A-4", "status": "Committed", "age": 6.0},
    ]


def test_calculate_ageing_wip_with_different_done_column(
    query_manager, settings, results, today
):
    """Test ageing WIP calculation with different done column."""
    settings.update(
        {
            "done_column": "Test",
        }
    )

    calculator = AgeingWIPChartCalculator(query_manager, settings, results)

    data = calculator.run(today)

    assert data[["key", "status", "age"]].to_dict("records") == [
        {"key": "A-2", "status": "Committed", "age": 7.0},
        {"key": "A-4", "status": "Committed", "age": 6.0},
    ]


def test_calculate_ageing_wip_with_skipped_columns(
    jira_with_skipped_columns, settings, today, now
):
    """Test ageing WIP calculation with skipped columns."""
    query_manager = QueryManager(jira_with_skipped_columns, settings)
    results = {}
    cycle_time_calc = CycleTimeCalculator(query_manager, settings, results)
    results[CycleTimeCalculator] = cycle_time_calc.run(now=now)
    ageing_wip_calc = AgeingWIPChartCalculator(query_manager, settings, results)
    data = ageing_wip_calc.run(today=today)

    assert data[["key", "status", "age"]].to_dict("records") == [
        {"key": "A-13", "status": "Build", "age": 8.0},
        {"key": "A-14", "status": "Build", "age": 8.0},
        {"key": "A-16", "status": "Build", "age": 7.0},
    ]
