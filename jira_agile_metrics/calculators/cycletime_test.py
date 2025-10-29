"""Tests for cycle time calculator functionality in Jira Agile Metrics.

This module contains unit tests for the cycle time calculator.
"""

import datetime

import pytest
from pandas import NaT, Timedelta, Timestamp

from ..querymanager import QueryManager
from ..test_classes import FauxChange as Change
from ..test_classes import FauxFieldValue as Value
from ..test_classes import FauxIssue as Issue
from ..test_classes import FauxJIRA as JIRA
from ..test_utils import (
    create_common_awaiting_input_changes,
    create_common_backlog_issue,
    create_common_backlog_to_next_change,
    create_common_cycle_status_list,
    create_common_cycle_time_columns,
    create_common_done_issue,
    create_common_impediment_changes,
    create_common_moved_back_changes,
    create_common_next_issue,
    create_common_next_to_build_change,
    create_common_status_transition_changes,
)
from .cycletime import CycleTimeCalculator


@pytest.fixture(name="test_jira")
def jira(base_custom_fields):
    """Test JIRA instance with custom fields."""
    return JIRA(
        fields=base_custom_fields,
        issues=[
            create_common_backlog_issue(
                "A-1",
                customfield_001="Team 1",
                customfield_002=Value(None, 10),
                customfield_003=Value(None, ["R2", "R3", "R4"]),
                customfield_100=None,
                changes=[],
            ),
            create_common_next_issue(
                "A-2",
                customfield_001="Team 1",
                customfield_002=Value(None, 20),
                customfield_003=Value(None, []),
                customfield_100=None,
                changes=create_common_impediment_changes()
                + [
                    Change(
                        "2018-01-03 01:00:00", [("Flagged", "Impediment", "")]
                    ),  # blocked 1 day in the backlog
                    # (doesn't count towards blocked days)
                    create_common_backlog_to_next_change(),
                    Change("2018-01-04 10:01:01", [("Flagged", "", "Impediment")]),
                    Change(
                        "2018-01-05 08:01:01", [("Flagged", "Impediment", "")]
                    ),  # was blocked 1 day
                    Change(
                        "2018-01-08 10:01:01", [("Flagged", "", "Impediment")]
                    ),  # stays blocked until today
                ],
            ),
            create_common_done_issue(
                "A-3",
                customfield_001="Team 1",
                customfield_002=Value(None, 30),
                customfield_003=Value(None, []),
                customfield_100=None,
                changes=create_common_status_transition_changes()
                + [
                    Change(
                        "2018-01-04 10:01:01",
                        [("Flagged", None, "Impediment")],
                    ),  # should clear two days later when issue resolved
                ],
            ),
            Issue(
                "A-4",
                summary="Moved back",
                issuetype=Value("Story", "story"),
                status=Value("Next", "next"),
                resolution=None,
                resolutiondate=None,
                created="2018-01-04 01:01:01",
                customfield_001="Team 1",
                customfield_002=Value(None, 30),
                customfield_003=Value(None, []),
                customfield_100=None,
                changes=[
                    create_common_backlog_to_next_change(),
                    create_common_next_to_build_change(),
                ]
                + create_common_moved_back_changes()
                + create_common_awaiting_input_changes(),
            ),
        ],
    )


@pytest.fixture(name="jira_client_skipped_columns")
def jira_client_fixture(
    base_custom_fields,
):
    """Create a JIRA fixture with issues that have skipped columns."""
    return JIRA(
        fields=base_custom_fields,
        issues=[
            Issue(
                "A-10",
                summary="Gaps",
                issuetype=Value("Story", "story"),
                status=Value("Done", "done"),
                resolution=Value("Done", "Done"),
                resolutiondate="2018-01-04 01:01:01",
                created="2018-01-01 01:01:01",
                customfield_001="Team 1",
                customfield_002=Value(None, 10),
                customfield_003=Value(None, []),
                customfield_100=None,
                changes=[
                    create_common_backlog_to_next_change(),
                    Change(
                        "2018-01-04 01:01:01",
                        [
                            (
                                "status",
                                "Next",
                                "Done",
                            ),
                            ("resolution", None, "done"),
                        ],
                    ),  # skipping columns Build and Test
                ],
            ),
            Issue(
                "A-11",
                summary="More Gaps",
                issuetype=Value("Story", "story"),
                status=Value("Done", "done"),
                resolution=Value("Done", "Done"),
                resolutiondate="2018-01-04 01:01:01",
                created="2018-01-01 01:01:01",
                customfield_001="Team 1",
                customfield_002=Value(None, 10),
                customfield_003=Value(None, []),
                customfield_100=None,
                changes=[
                    Change(
                        "2018-01-02 01:05:01",
                        [
                            (
                                "status",
                                "Backlog",
                                "Build",
                            )
                        ],
                    ),
                    Change(
                        "2018-01-04 01:01:01",
                        [
                            (
                                "status",
                                "Build",
                                "Done",
                            ),
                            ("resolution", None, "done"),
                        ],
                    ),  # skipping columns Build and Test
                ],
            ),
        ],
    )


@pytest.fixture(name="test_settings")
def settings(base_custom_settings):
    """Test settings configuration."""
    return base_custom_settings


def test_columns(test_jira, test_settings):
    """Test cycle time calculator column handling."""
    query_manager = QueryManager(test_jira, test_settings)
    results = {}
    calculator = CycleTimeCalculator(query_manager, test_settings, results)

    data = calculator.run()

    assert (
        list(data.columns)
        == create_common_cycle_time_columns()
        + [
            "Estimate",
            "Release",
            "Team",
        ]
        + create_common_cycle_status_list()
    )


def test_empty(base_custom_fields, test_settings):
    """Test cycle time calculator with empty data."""
    jira_client = JIRA(fields=base_custom_fields, issues=[])
    query_manager = QueryManager(jira_client, test_settings)
    results = {}
    calculator = CycleTimeCalculator(query_manager, test_settings, results)

    data = calculator.run()

    assert len(data.index) == 0


def test_movement(test_jira, test_settings):
    """Test cycle time calculator movement functionality."""
    query_manager = QueryManager(test_jira, test_settings)
    results = {}
    calculator = CycleTimeCalculator(query_manager, test_settings, results)

    data = calculator.run(now=datetime.datetime(2018, 1, 10, 15, 37, 0))

    assert data.to_dict("records") == [
        {
            "key": "A-1",
            "url": "https://example.org/browse/A-1",
            "issue_type": "Story",
            "summary": "Just created",
            "status": "Backlog",
            "resolution": None,
            "Estimate": 10,
            "Release": "R3",
            "Team": "Team 1",
            "completed_timestamp": NaT,
            "cycle_time": NaT,
            "lead_time": NaT,
            "blocked_days": 0,
            "impediments": [],
            "Backlog": Timestamp("2018-01-01 00:00:00"),
            "Committed": NaT,
            "Build": NaT,
            "Test": NaT,
            "Done": NaT,
        },
        {
            "key": "A-2",
            "url": "https://example.org/browse/A-2",
            "issue_type": "Story",
            "summary": "Started",
            "status": "Next",
            "resolution": None,
            "Estimate": 20,
            "Release": "None",
            "Team": "Team 1",
            "completed_timestamp": NaT,
            "cycle_time": NaT,
            "lead_time": NaT,
            "blocked_days": 3,
            "impediments": [
                {
                    "start": datetime.date(2018, 1, 2),
                    "end": datetime.date(2018, 1, 3),
                    "status": "Backlog",
                    "flag": "Impediment",
                },  # doesn't count towards blocked_days
                {
                    "start": datetime.date(2018, 1, 4),
                    "end": datetime.date(2018, 1, 5),
                    "status": "Committed",
                    "flag": "Impediment",
                },
                {
                    "start": datetime.date(2018, 1, 8),
                    "end": None,
                    "status": "Committed",
                    "flag": "Impediment",
                },
            ],
            "Backlog": Timestamp("2018-01-02 00:00:00"),
            "Committed": Timestamp("2018-01-03 00:00:00"),
            "Build": NaT,
            "Test": NaT,
            "Done": NaT,
        },
        {
            "key": "A-3",
            "url": "https://example.org/browse/A-3",
            "summary": "Completed",
            "issue_type": "Story",
            "status": "Done",
            "resolution": "Done",
            "Estimate": 30,
            "Release": "None",
            "Team": "Team 1",
            "completed_timestamp": Timestamp("2018-01-06 00:00:00"),
            "cycle_time": Timedelta("3 days 00:00:00"),
            "lead_time": Timedelta("3 days 00:00:00"),
            "blocked_days": 2,
            "impediments": [
                {
                    "start": datetime.date(2018, 1, 4),
                    "end": datetime.date(2018, 1, 6),
                    "status": "Build",
                    "flag": "Impediment",
                }
            ],
            "Backlog": Timestamp("2018-01-03 00:00:00"),
            "Committed": Timestamp("2018-01-03 00:00:00"),
            "Build": Timestamp("2018-01-04 00:00:00"),
            "Test": Timestamp("2018-01-06 00:00:00"),
            "Done": Timestamp("2018-01-06 00:00:00"),
        },
        {
            "key": "A-4",
            "url": "https://example.org/browse/A-4",
            "summary": "Moved back",
            "issue_type": "Story",
            "status": "Next",
            "resolution": None,
            "Estimate": 30,
            "Release": "None",
            "Team": "Team 1",
            "completed_timestamp": NaT,
            "cycle_time": NaT,
            "lead_time": NaT,
            "blocked_days": 3,
            "impediments": [
                {
                    "start": datetime.date(2018, 1, 7),
                    "end": datetime.date(2018, 1, 10),
                    "status": "Committed",
                    "flag": "Awaiting input",
                }
            ],
            "Backlog": Timestamp("2018-01-04 00:00:00"),
            "Committed": Timestamp("2018-01-03 00:00:00"),
            "Build": NaT,
            "Test": NaT,
            "Done": NaT,
        },
    ]


def test_movement_skipped_columns(jira_client_skipped_columns, test_settings):
    """Test cycle time calculator with skipped columns."""
    query_manager = QueryManager(jira_client_skipped_columns, test_settings)
    results = {}
    calculator = CycleTimeCalculator(query_manager, test_settings, results)

    data = calculator.run(now=datetime.datetime(2018, 1, 10, 15, 37, 0))

    assert data.to_dict("records") == [
        {
            "key": "A-10",
            "url": "https://example.org/browse/A-10",
            "issue_type": "Story",
            "summary": "Gaps",
            "status": "Done",
            "resolution": "Done",
            "Estimate": 10,
            "Release": "None",
            "Team": "Team 1",
            "completed_timestamp": Timestamp("2018-01-04 00:00:00"),
            "cycle_time": Timedelta("1 days 00:00:00"),
            "lead_time": Timedelta("3 days 00:00:00"),
            "blocked_days": 0,
            "impediments": [],
            "Backlog": Timestamp("2018-01-01 00:00:00"),
            "Committed": Timestamp("2018-01-03 00:00:00"),
            "Build": Timestamp("2018-01-04 00:00:00"),
            "Test": Timestamp("2018-01-04 00:00:00"),
            "Done": Timestamp("2018-01-04 00:00:00"),
        },
        {
            "key": "A-11",
            "url": "https://example.org/browse/A-11",
            "issue_type": "Story",
            "summary": "More Gaps",
            "status": "Done",
            "resolution": "Done",
            "Estimate": 10,
            "Release": "None",
            "Team": "Team 1",
            "completed_timestamp": Timestamp("2018-01-04 00:00:00"),
            "cycle_time": Timedelta("2 days 00:00:00"),
            "lead_time": Timedelta("3 days 00:00:00"),
            "blocked_days": 0,
            "impediments": [],
            "Backlog": Timestamp("2018-01-01 00:00:00"),
            "Committed": Timestamp("2018-01-02 00:00:00"),
            "Build": Timestamp("2018-01-02 00:00:00"),
            "Test": Timestamp("2018-01-04 00:00:00"),
            "Done": Timestamp("2018-01-04 00:00:00"),
        },
    ]
