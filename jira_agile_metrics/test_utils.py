"""Shared test utilities for Jira Agile Metrics tests.

This module provides common test utilities and fixtures to reduce duplication
across test files.
"""

import datetime as dt
import json
import os

import pandas as pd
import pytest
from pandas import DataFrame, Timestamp, date_range

from .calculators.cfd import CFDCalculator
from .calculators.cycletime import CycleTimeCalculator
from .common_constants import COMMON_CYCLE_TIME_FIELDS
from .querymanager import QueryManager
from .test_classes import (
    FauxChange as Change,
)
from .test_classes import (
    FauxFieldValue as Value,
)
from .test_classes import (
    FauxIssue as Issue,
)
from .test_classes import (
    FauxJIRA as JIRA,
)
from .utils import extend_dict


@pytest.fixture
def common_query_manager(minimal_query_manager):
    """Common query manager fixture used across multiple test files."""
    return minimal_query_manager


@pytest.fixture
def common_results(large_cycle_time_results):
    """Common results fixture used across multiple test files."""
    return extend_dict(large_cycle_time_results, {})


def create_empty_test_results(minimal_cycle_time_columns):
    """Create empty test results DataFrame for testing."""
    return {
        CycleTimeCalculator: DataFrame([], columns=minimal_cycle_time_columns, index=[])
    }


def create_default_epic_config():
    """Create a default epic configuration dict for testing."""
    return {
        "min_stories_field": None,
        "max_stories_field": None,
        "team_field": None,
        "deadline_field": None,
    }


def create_default_epic_data(**kwargs):
    """Create default epic data dict for testing with optional overrides."""
    defaults = {
        "key": "EPIC-1",
        "summary": "Test Epic",
        "status": "In Progress",
        "resolution": None,
        "resolution_date": None,
        "min_stories": 0,
        "max_stories": 10,
        "team_name": "Team A",
        "deadline": None,
        "story_query": None,
        "story_cycle_times": [],
        "stories_raised": 0,
        "stories_in_backlog": 0,
        "stories_in_progress": 0,
        "stories_done": 0,
        "first_story_started": None,
        "last_story_finished": None,
        "team": None,
        "outcome": None,
        "forecast": None,
    }
    defaults.update(kwargs)
    return defaults


def create_default_story_counts(**kwargs):
    """Create default story counts dict for testing with optional overrides."""
    defaults = {
        "stories_raised": 0,
        "stories_in_backlog": 0,
        "stories_in_progress": 0,
        "stories_done": 0,
        "first_story_started": None,
        "last_story_finished": None,
    }
    defaults.update(kwargs)
    return defaults


def create_timestamp_index(start_date="2018-01-01", periods=6):
    """Create a common timestamp index for tests."""
    return list(pd.date_range(start=start_date, periods=periods, freq="D"))


def create_extended_timestamp_index(start_date="2018-01-01", periods=9):
    """Create an extended timestamp index for tests."""
    return create_timestamp_index(start_date, periods)


# Common timestamp assertions used across multiple test files
COMMON_TIMESTAMP_INDEX = create_timestamp_index("2018-01-01", 6)

EXTENDED_TIMESTAMP_INDEX = create_extended_timestamp_index("2018-01-01", 9)


def assert_common_timestamp_index(data):
    """Assert that data has the common timestamp index."""
    assert list(data.index) == COMMON_TIMESTAMP_INDEX


def assert_extended_timestamp_index(data):
    """Assert that data has the extended timestamp index."""
    assert list(data.index) == EXTENDED_TIMESTAMP_INDEX


def create_common_test_issue(key="A-1", issue_data=None, **kwargs):
    """Create a common test issue with standard defaults.

    Args:
        key: Issue key
        issue_data: Dictionary containing issue data (summary, issuetype, status, etc.)
        **kwargs: Additional fields to add to the issue
    """
    if issue_data is None:
        issue_data = {
            "summary": "Just created",
            "issuetype": "Story",
            "status": "Backlog",
            "created": "2018-01-01 01:01:01",
            "updated": "2018-01-01 01:01:01",
            "resolution": None,
            "resolutiondate": None,
            "project": Value("PROJ", "PROJ"),
            "reporter": Value("user1", "User One", email_address="user1@example.com"),
            "assignee": None,
            "priority": Value("Medium", "Medium"),
            "labels": [],
            "components": [],
            "fixVersions": [],
        }

    # Handle custom fields and other kwargs
    issue_kwargs = {
        "summary": issue_data.get("summary", "Just created"),
        "issuetype": Value(
            issue_data.get("issuetype", "Story"),
            issue_data.get("issuetype", "Story").lower(),
        ),
        "status": Value(
            issue_data.get("status", "Backlog"),
            issue_data.get("status", "Backlog").lower(),
        ),
        "resolution": issue_data.get("resolution"),
        "resolutiondate": issue_data.get("resolutiondate"),
        "created": issue_data.get("created", "2018-01-01 01:01:01"),
        "updated": issue_data.get("updated", "2018-01-01 01:01:01"),
        "project": issue_data.get("project", Value("PROJ", "PROJ")),
        "reporter": issue_data.get(
            "reporter", Value("user1", "User One", email_address="user1@example.com")
        ),
        "assignee": issue_data.get("assignee"),
        "priority": issue_data.get("priority", Value("Medium", "Medium")),
        "labels": issue_data.get("labels", []),
        "components": issue_data.get("components", []),
        "fixVersions": issue_data.get("fixVersions", []),
    }

    # Add any additional fields passed in kwargs
    issue_kwargs.update(kwargs)

    # Extract changes from kwargs if present, otherwise use empty list
    changes = kwargs.get("changes", [])

    # Remove changes from kwargs to avoid duplicate argument
    issue_kwargs.pop("changes", None)

    return Issue(key, changes, **issue_kwargs)


def create_common_backlog_issue(
    key="A-1", summary="Just created", created="2018-01-01 01:01:01", **kwargs
):
    """Create a common backlog issue with standard defaults."""
    return create_common_test_issue(
        key=key,
        issue_data={
            "summary": summary,
            "issuetype": "Story",
            "status": "Backlog",
            "created": created,
        },
        **kwargs,
    )


def create_common_next_issue(
    key="A-2", summary="Started", created="2018-01-02 01:01:01", **kwargs
):
    """Create a common next status issue with standard defaults."""
    return create_common_test_issue(
        key=key,
        issue_data={
            "summary": summary,
            "issuetype": "Story",
            "status": "Next",
            "created": created,
        },
        **kwargs,
    )


def create_common_done_issue(
    key="A-3",
    summary="Completed",
    created="2018-01-03 01:01:01",
    resolutiondate="2018-01-06 01:01:01",
    **kwargs,
):
    """Create a common done issue with standard defaults."""
    return create_common_test_issue(
        key=key,
        issue_data={
            "summary": summary,
            "issuetype": "Story",
            "status": "Done",
            "created": created,
            "resolution": Value("Done", "Done"),
            "resolutiondate": resolutiondate,
        },
        **kwargs,
    )


def create_common_test_issue_with_changes(
    key="I-1",
    summary="Just created",
    issuetype="Story",
    status="Backlog",
    created="2018-01-01 01:01:01",
):
    """Create a common test issue with standard changes."""
    return create_common_test_issue(
        key=key,
        issue_data={
            "summary": summary,
            "issuetype": issuetype,
            "status": status,
            "created": created,
        },
        changes=[
            Change(
                "2018-01-02 10:01:01",
                [("Flagged", None, "Impediment")],
            ),
            Change(
                "2018-01-03 10:01:01",
                [("Flagged", "Impediment", None)],
            ),
        ],
    )


def create_common_defect_test_settings():
    """Create common defect test settings."""
    return {
        "defects_type_field": "Issue type",
        "defects_type_values": ["Config", "Data", "Code"],
        "defects_environment_field": "Environment",
        "defects_environment_values": ["SIT", "UAT", "PROD"],
        # Set output paths to None by default to prevent files from being
        # written to project root. Tests that need to write files should
        # override these with tmp_path-based paths.
        "defects_by_priority_chart": None,
        "defects_by_priority_chart_title": "Defects by priority",
        "defects_by_type_chart": None,
        "defects_by_type_chart_title": "Defects by type",
        "defects_by_environment_chart": None,
        "defects_by_environment_chart_title": "Defects by environment",
    }


def create_common_cycle_time_columns():
    """Create common cycle time columns."""
    return [
        "key",
        "url",
        "issue_type",
        "summary",
        "status",
        "resolution",
    ] + COMMON_CYCLE_TIME_FIELDS


def create_common_minimal_fields():
    """Create common minimal fields."""
    return [
        "key",
        "url",
        "issue_type",
        "summary",
        "status",
        "resolution",
    ]


def create_scatterplot_expected_columns():
    """Create expected columns list for scatterplot calculator tests."""
    return [
        "completed_date",
        "cycle_time",
        "blocked_days",
        "key",
        "url",
        "issue_type",
        "summary",
        "status",
        "resolution",
        "lead_time",
        "Backlog",
        "Committed",
        "Build",
        "Test",
        "Done",
    ]


def create_common_cfd_results_fixture(
    query_manager, settings, large_cycle_time_results
):
    """Create common CFD results fixture used across multiple test files."""
    return extend_dict(
        large_cycle_time_results,
        {
            CFDCalculator: CFDCalculator(
                query_manager, settings, large_cycle_time_results
            ).run()
        },
    )


def create_common_empty_cfd_results(_minimal_cycle_time_columns):
    """Create common empty CFD results for testing."""
    return {
        CFDCalculator: DataFrame(
            [],
            columns=["Backlog", "Committed", "Build", "Test", "Done"],
            index=date_range(start=dt.date(2018, 1, 1), periods=0, freq="D"),
        )
    }


def create_common_impediment_changes():
    """Create common impediment change patterns used across tests."""
    return [
        Change(
            "2018-01-02 10:01:01",
            [("Flagged", None, "Impediment")],
        ),
        Change(
            "2018-01-03 10:01:01",
            [("Flagged", "Impediment", None)],
        ),
    ]


def create_common_awaiting_input_changes():
    """Create common 'Awaiting input' impediment change patterns used across tests."""
    return [
        Change(
            "2018-01-07 01:01:01",
            [("Flagged", None, "Awaiting input")],
        ),
        Change(
            "2018-01-10 10:01:01",
            [("Flagged", "Awaiting input", "")],
        ),  # blocked 3 days
    ]


def create_common_status_transition_changes():
    """Create common status transition change patterns used across tests."""
    return [
        Change(
            "2018-01-03 01:01:01",
            [
                (
                    "status",
                    "Backlog",
                    "Next",
                )
            ],
        ),
        Change(
            "2018-01-04 01:01:01",
            [
                (
                    "status",
                    "Next",
                    "Build",
                )
            ],
        ),
        Change(
            "2018-01-05 01:01:01",
            [
                (
                    "status",
                    "Build",
                    "QA",
                )
            ],
        ),
        Change(
            "2018-01-06 01:01:01",
            [
                (
                    "status",
                    "QA",
                    "Done",
                )
            ],
        ),
    ]


def create_common_moved_back_changes():
    """Create common status transition changes with 'moved back' scenario."""
    return [
        Change(
            "2018-01-04 01:01:01",
            [
                (
                    "status",
                    "Backlog",
                    "Next",
                )
            ],
        ),
        Change(
            "2018-01-05 01:01:01",
            [
                (
                    "status",
                    "Next",
                    "Build",
                )
            ],
        ),
        Change(
            "2018-01-06 01:01:01",
            [
                (
                    "status",
                    "Build",
                    "Next",
                )
            ],
        ),
    ]


def create_common_issue_changes():
    """Create common issue change patterns used across tests."""
    return [
        Change(
            "2018-01-04 01:01:01",
            [
                (
                    "status",
                    "Backlog",
                    "Next",
                )
            ],
        ),
        Change(
            "2018-01-05 01:01:01",
            [
                (
                    "status",
                    "Next",
                    "Build",
                )
            ],
        ),
        Change(
            "2018-01-06 01:01:01",
            [
                (
                    "status",
                    "Build",
                    "QA",
                )
            ],
        ),
        Change(
            "2018-01-07 01:01:01",
            [
                (
                    "status",
                    "QA",
                    "Done",
                )
            ],
        ),
    ]


def create_common_backlog_to_next_change():
    """Create common Backlog to Next status change."""
    return Change(
        "2018-01-03 01:01:01",
        [
            (
                "status",
                "Backlog",
                "Next",
            )
        ],
    )


def create_common_next_to_build_change():
    """Create common Next to Build status change."""
    return Change(
        "2018-01-04 01:01:01",
        [
            (
                "status",
                "Next",
                "Build",
            )
        ],
    )


def create_common_build_to_qa_change():
    """Create common Build to QA status change."""
    return Change(
        "2018-01-05 01:01:01",
        [
            (
                "status",
                "Build",
                "QA",
            )
        ],
    )


def create_common_qa_to_done_change():
    """Create common QA to Done status change."""
    return Change(
        "2018-01-06 01:01:01",
        [
            (
                "status",
                "QA",
                "Done",
            )
        ],
    )


def create_common_test_issue_with_impediments(
    key="I-1",
    summary="Just created",
    issuetype="Story",
    status="Backlog",
    created="2018-01-01 01:01:01",
):
    """Create a common test issue with impediment changes."""
    return create_common_test_issue(
        key=key,
        issue_data={
            "summary": summary,
            "issuetype": issuetype,
            "status": status,
            "created": created,
        },
        changes=create_common_impediment_changes(),
    )


def create_common_test_issue_with_status_transitions(
    key="I-2",
    summary="Status transitions",
    issuetype="Story",
    status="Backlog",
    created="2018-01-01 01:01:01",
):
    """Create a common test issue with status transition changes."""
    return create_common_test_issue(
        key=key,
        issue_data={
            "summary": summary,
            "issuetype": issuetype,
            "status": status,
            "created": created,
        },
        changes=create_common_status_transition_changes(),
    )


def run_empty_calculator_test(
    calculator_class, fields, settings, expected_columns=None
):
    """Run a common empty calculator test pattern."""
    jira = JIRA(fields=fields, issues=[])
    query_manager = QueryManager(jira, settings)
    results = {}
    calculator = calculator_class(query_manager, settings, results)

    data = calculator.run()

    if expected_columns:
        assert list(data.columns) == expected_columns

    return data


def create_common_cycle_status_list():
    """Create common cycle status list used across multiple tests."""
    return [
        "Backlog",
        "Committed",
        "Build",
        "Test",
        "Done",
    ]


def create_waste_settings():
    """Create waste settings dictionary to eliminate duplication."""
    return {
        "waste_query": ("issueType = Story AND resolution IN (Withdrawn, Invalid)"),
        "waste_window": 3,
        "waste_frequency": "2W-WED",
        "waste_chart": "waste.png",
        "waste_chart_title": "Waste",
    }


def validate_defect_test_data(
    data, expected_columns, valid_keys=None, expected_count=None
):
    """Helper to validate defect test data and extract valid records.

    Args:
        data: DataFrame to validate
        expected_columns: List of expected column names
        valid_keys: Optional list of valid keys to filter,
            defaults to D-1 through D-6
        expected_count: Optional expected number of valid records.
            Defaults to len(valid_keys)

    Returns:
        Tuple of (records dict, valid_records list)
    """
    if valid_keys is None:
        valid_keys = ["D-1", "D-2", "D-3", "D-4", "D-5", "D-6"]

    # If expected_count not provided, default to the number of valid_keys
    if expected_count is None:
        expected_count = len(valid_keys)

    assert list(data.columns) == expected_columns
    records = data.to_dict("records")
    valid_records = [r for r in records if r["key"] in valid_keys]
    assert len(valid_records) == expected_count
    return records, valid_records


def assert_common_d1_d2_record_values(valid_records):
    """Common assertions for D-1 and D-2 records in defect/debt tests.

    This function checks the common fields that are identical across tests:
    - D-1: priority="High", created timestamp, resolved is NaN
    - D-2: priority="Medium", created timestamp, resolved timestamp

    Args:
        valid_records: List of valid records from validate_defect_test_data
    """
    d1_record = next(r for r in valid_records if r["key"] == "D-1")
    assert d1_record["priority"] == "High"
    assert d1_record["created"] == Timestamp("2018-01-01 01:01:01")
    assert pd.isna(d1_record["resolved"])

    d2_record = next(r for r in valid_records if r["key"] == "D-2")
    assert d2_record["priority"] == "Medium"
    assert d2_record["created"] == Timestamp("2018-01-01 01:01:01")
    assert d2_record["resolved"] == Timestamp("2018-03-20 02:02:02")


def assert_common_d1_d2_record_values_no_priority(valid_records):
    """Common assertions for D-1 and D-2 records when priority field is not configured.

    This function checks the common fields that are identical across tests:
    - D-1: priority=None, created timestamp, resolved is NaN
    - D-2: priority=None, created timestamp, resolved timestamp

    Args:
        valid_records: List of valid records from validate_defect_test_data
    """
    d1_record = next(r for r in valid_records if r["key"] == "D-1")
    assert d1_record["priority"] is None
    assert d1_record["created"] == Timestamp("2018-01-01 01:01:01")
    assert pd.isna(d1_record["resolved"])

    d2_record = next(r for r in valid_records if r["key"] == "D-2")
    assert d2_record["priority"] is None
    assert d2_record["created"] == Timestamp("2018-01-01 01:01:01")
    assert d2_record["resolved"] == Timestamp("2018-03-20 02:02:02")


def assert_calculator_wrote_json_file(calculator, output_file, results_dict):
    """Helper function to run calculator, write output, and verify JSON file.

    This function eliminates duplicate code across test files by providing
    a common pattern for testing calculator write operations to JSON files.

    Args:
        calculator: Calculator instance to run and write
        output_file: Path to the expected output JSON file
        results_dict: Dictionary to store calculator results (will be updated)
    """
    result = calculator.run()
    results_dict[type(calculator)] = result
    calculator.write()

    assert os.path.exists(output_file)
    with open(output_file, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    return data
