"""Tests for query manager functionality in Jira Agile Metrics.

This module contains unit tests for the QueryManager class and related functionality.
"""

import datetime

import pytest

from .querymanager import IssueSnapshot, QueryManager
from .test_classes import (
    FauxChange as Change,
)
from .test_classes import (
    FauxFieldValue as Value,
)
from .test_classes import (
    FauxIssue as Issue,
)
from .test_classes import FauxJIRA as JIRA
from .utils import extend_dict


@pytest.fixture(name="test_jira")
def jira(base_custom_fields):
    """Test JIRA instance with custom fields."""
    return JIRA(
        fields=base_custom_fields,
        issues=[
            Issue(
                "A-1",
                summary="Issue A-1",
                issuetype=Value("Story", "story"),
                status=Value("Backlotg", "backlog"),
                resolution=None,
                created="2018-01-01 01:01:01",
                customfield_001="Team 1",
                customfield_002=Value(None, 30),
                customfield_003=Value(None, ["R2", "R3", "R4"]),
                changes=[
                    # the changes are not in chrnological order, the first
                    # change is intentionally the third
                    # status change. This is intended to test that we manage
                    # get the correct first status change as
                    # the transition from Backlog to Next
                    Change(
                        "2018-01-03 01:01:01",
                        [
                            (
                                "resolution",
                                None,
                                "Closed",
                            ),
                            (
                                "status",
                                "Next",
                                "Done",
                            ),
                        ],
                    ),
                    Change(
                        "2018-01-02 01:01:01",
                        [
                            (
                                "status",
                                "Backlog",
                                "Next",
                            )
                        ],
                    ),
                    Change(
                        "2018-01-02 01:01:01",
                        [
                            (
                                "Team",
                                "Team 2",
                                "Team 1",
                            )
                        ],
                    ),
                    Change(
                        "2018-01-04 01:01:01",
                        [
                            (
                                "resolution",
                                "Closed",
                                None,
                            ),
                            (
                                "status",
                                "Done",
                                "QA",
                            ),
                        ],
                    ),
                ],
            )
        ],
    )


@pytest.fixture(name="test_settings")
def settings(base_custom_settings):
    """Test settings configuration."""
    return extend_dict(base_custom_settings, {})


def test_search(test_jira, test_settings):
    """Test search functionality."""
    qm = QueryManager(test_jira, test_settings)
    assert qm.attributes_to_fields == {
        "Team": "customfield_001",
        "Estimate": "customfield_002",
        "Release": "customfield_003",
    }

    issues = qm.find_issues("(filter=123)")
    assert issues == test_jira.search_issues("(filter=123)")


def test_resolve_attribute_value(test_jira, test_settings):
    """Test resolve_attribute_value functionality."""
    qm = QueryManager(test_jira, test_settings)
    issues = qm.find_issues("(filter=123)")

    assert qm.resolve_attribute_value(issues[0], "Team") == "Team 1"
    assert qm.resolve_attribute_value(issues[0], "Estimate") == 30
    assert (
        qm.resolve_attribute_value(issues[0], "Release") == "R3"
    )  # due to known_value


def test_resolve_field_value(test_jira, test_settings):
    """Test resolve_field_value functionality."""
    qm = QueryManager(test_jira, test_settings)
    issues = qm.find_issues("(filter=123)")

    assert qm.resolve_field_value(issues[0], "customfield_001") == "Team 1"
    assert qm.resolve_field_value(issues[0], "customfield_002") == 30
    assert (
        qm.resolve_field_value(issues[0], "customfield_003") == "R3"
    )  # due to known_value


def test_iter_changes(test_jira, test_settings):
    """Test iter_changes functionality."""
    qm = QueryManager(test_jira, test_settings)
    issues = qm.find_issues("(filter=123)")
    changes = list(qm.iter_changes(issues[0], ["status", "Team"]))

    assert changes == [
        IssueSnapshot(
            change="status",
            transition_data={
                "key": "A-1",
                "date": datetime.datetime(2018, 1, 1, 1, 1, 1),
                "from_string": None,
                "to_string": "Backlog",
            },
        ),
        IssueSnapshot(
            change="Team",
            transition_data={
                "key": "A-1",
                "date": datetime.datetime(2018, 1, 1, 1, 1, 1),
                "from_string": None,
                "to_string": "Team 2",
            },
        ),
        IssueSnapshot(
            change="status",
            transition_data={
                "key": "A-1",
                "date": datetime.datetime(2018, 1, 2, 1, 1, 1),
                "from_string": "Backlog",
                "to_string": "Next",
            },
        ),
        IssueSnapshot(
            change="Team",
            transition_data={
                "key": "A-1",
                "date": datetime.datetime(2018, 1, 2, 1, 1, 1),
                "from_string": "Team 2",
                "to_string": "Team 1",
            },
        ),
        IssueSnapshot(
            change="status",
            transition_data={
                "key": "A-1",
                "date": datetime.datetime(2018, 1, 3, 1, 1, 1),
                "from_string": "Next",
                "to_string": "Done",
            },
        ),
        IssueSnapshot(
            change="status",
            transition_data={
                "key": "A-1",
                "date": datetime.datetime(2018, 1, 4, 1, 1, 1),
                "from_string": "Done",
                "to_string": "QA",
            },
        ),
    ]
