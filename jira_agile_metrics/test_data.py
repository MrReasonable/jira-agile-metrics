"""Shared test data and fixtures for Jira Agile Metrics tests.

This module provides common test data structures to reduce duplication
across test files.
"""

from jira_agile_metrics.test_classes import (
    FauxFieldValue as Value,
)
from jira_agile_metrics.test_classes import (
    FauxIssue as Issue,
)

from .common_constants import BOTTLENECK_CHART_SETTINGS
from .test_utils import (
    create_common_cycle_status_list,
    create_common_cycle_time_columns,
    create_common_impediment_changes,
    create_common_minimal_fields,
)

# Common cycle configuration used across multiple tests
COMMON_CYCLE_CONFIG = [
    {"name": "Backlog", "statuses": ["Backlog"]},
    {"name": "Committed", "statuses": ["Next"]},
    {"name": "Build", "statuses": ["Build"]},
    {"name": "Test", "statuses": ["Code review", "QA"]},
    {"name": "Done", "statuses": ["Done"]},
]

# Common CFD columns
COMMON_CFD_COLUMNS = create_common_cycle_status_list()

# Use common field definitions from test_utils
COMMON_MINIMAL_FIELDS = create_common_minimal_fields()
COMMON_CYCLE_TIME_COLUMNS = create_common_cycle_time_columns()
COMMON_BOTTLENECK_CHART_SETTINGS = BOTTLENECK_CHART_SETTINGS

# Common CFD test data
COMMON_CFD_DATA = [
    {
        "Backlog": 1.0,
        "Committed": 0.0,
        "Build": 0.0,
        "Test": 0.0,
        "Done": 0.0,
    },
    {
        "Backlog": 2.0,
        "Committed": 0.0,
        "Build": 0.0,
        "Test": 0.0,
        "Done": 0.0,
    },
    {
        "Backlog": 3.0,
        "Committed": 2.0,
        "Build": 0.0,
        "Test": 0.0,
        "Done": 0.0,
    },
    {
        "Backlog": 4.0,
        "Committed": 3.0,
        "Build": 1.0,
        "Test": 0.0,
        "Done": 0.0,
    },
    {
        "Backlog": 4.0,
        "Committed": 3.0,
        "Build": 1.0,
        "Test": 1.0,
        "Done": 0.0,
    },
    {
        "Backlog": 4.0,
        "Committed": 3.0,
        "Build": 1.0,
        "Test": 1.0,
        "Done": 1.0,
    },
    {
        "Backlog": 4.0,
        "Committed": 3.0,
        "Build": 1.0,
        "Test": 1.0,
        "Done": 1.0,
    },
]

# Common test issue data


def create_test_issue(key, issue_data):
    """Create a test issue with common defaults.

    Args:
        key: Issue key
        issue_data: Dictionary containing issue data with keys:
            summary, issuetype, status, created, resolution,
            resolutiondate, priority, changes
    """
    changes = issue_data.get("changes", [])

    return Issue(
        key,
        summary=issue_data["summary"],
        issuetype=Value(issue_data["issuetype"], issue_data["issuetype"].lower()),
        status=Value(issue_data["status"], issue_data["status"].lower()),
        resolution=issue_data.get("resolution"),
        resolutiondate=issue_data.get("resolutiondate"),
        created=issue_data["created"],
        updated=issue_data.get("updated", issue_data["created"]),
        project=Value("PROJ", "PROJ"),
        reporter=Value("user1", "User One", email_address="user1@example.com"),
        assignee=None,
        priority=issue_data.get("priority"),
        labels=[],
        components=[],
        fixVersions=[],
        changes=changes,
    )


# Common test issues for debt and defects tests
COMMON_TEST_ISSUES = [
    create_test_issue(
        "D-1",
        {
            "summary": "Test Issue 1",
            "issuetype": "Tech Debt",
            "status": "Closed",
            "created": "2018-01-01 01:01:01",
            "resolution": "Done",
            "resolutiondate": "2018-03-20 02:02:02",
            "priority": Value("High", "High"),
        },
    ),
    create_test_issue(
        "D-2",
        {
            "summary": "Test Issue 2",
            "issuetype": "Tech Debt",
            "status": "Closed",
            "created": "2018-01-02 01:01:01",
            "resolution": "Done",
            "resolutiondate": "2018-01-20 02:02:02",
            "priority": Value("Medium", "Medium"),
        },
    ),
    create_test_issue(
        "D-3",
        {
            "summary": "Test Issue 3",
            "issuetype": "Tech Debt",
            "status": "Closed",
            "created": "2018-02-03 01:01:01",
            "resolution": "Done",
            "resolutiondate": "2018-03-20 02:02:02",
            "priority": Value("High", "High"),
        },
    ),
    create_test_issue(
        "D-4",
        {
            "summary": "Test Issue 4",
            "issuetype": "Tech Debt",
            "status": "Closed",
            "created": "2018-01-04 01:01:01",
            "resolution": None,
            "resolutiondate": None,
            "priority": Value("Medium", "Medium"),
        },
    ),
    create_test_issue(
        "D-5",
        {
            "summary": "Test Issue 5",
            "issuetype": "Tech Debt",
            "status": "Closed",
            "created": "2018-02-05 01:01:01",
            "resolution": None,
            "resolutiondate": None,
            "priority": Value("Medium", "Medium"),
        },
    ),
    create_test_issue(
        "D-6",
        {
            "summary": "Test Issue 6",
            "issuetype": "Tech Debt",
            "status": "Closed",
            "created": "2018-03-06 01:01:01",
            "resolution": None,
            "resolutiondate": None,
            "priority": Value("Medium", "Medium"),
        },
    ),
]

# Common test issues for defects tests (with custom fields)
COMMON_DEFECT_TEST_ISSUES = [
    Issue(
        "D-1",
        summary="Debt 1",
        issuetype=Value("Bug", "Bug"),
        status=Value("Closed", "closed"),
        created="2018-01-01 01:01:01",
        updated="2018-03-20 02:02:02",
        resolution="Done",
        resolutiondate="2018-03-20 02:02:02",
        priority=Value("High", "High"),
        project=Value("PROJ", "PROJ"),
        reporter=Value("user1", "User One", email_address="user1@example.com"),
        assignee=None,
        labels=[],
        components=[],
        fixVersions=[],
        customfield_001=Value(None, "PROD"),
        customfield_002=Value(None, "Config"),
        changes=[],
    ),
    Issue(
        "D-2",
        summary="Debt 2",
        issuetype=Value("Bug", "Bug"),
        status=Value("Closed", "closed"),
        created="2018-01-02 01:01:01",
        updated="2018-01-20 02:02:02",
        resolution="Done",
        resolutiondate="2018-01-20 02:02:02",
        priority=Value("Medium", "Medium"),
        project=Value("PROJ", "PROJ"),
        reporter=Value("user1", "User One", email_address="user1@example.com"),
        assignee=None,
        labels=[],
        components=[],
        fixVersions=[],
        customfield_001=Value(None, "SIT"),
        customfield_002=Value(None, "Config"),
        changes=[],
    ),
    Issue(
        "D-3",
        summary="Debt 3",
        issuetype=Value("Bug", "Bug"),
        status=Value("Closed", "closed"),
        created="2018-02-03 01:01:01",
        updated="2018-03-20 02:02:02",
        resolution="Done",
        resolutiondate="2018-03-20 02:02:02",
        priority=Value("High", "High"),
        project=Value("PROJ", "PROJ"),
        reporter=Value("user1", "User One", email_address="user1@example.com"),
        assignee=None,
        labels=[],
        components=[],
        fixVersions=[],
        customfield_001=Value(None, "UAT"),
        customfield_002=Value(None, "Config"),
        changes=[],
    ),
    Issue(
        "D-4",
        summary="Debt 4",
        issuetype=Value("Bug", "Bug"),
        status=Value("Closed", "closed"),
        created="2018-01-04 01:01:01",
        updated="2018-01-04 01:01:01",
        resolution=None,
        resolutiondate=None,
        priority=Value("Medium", "Medium"),
        project=Value("PROJ", "PROJ"),
        reporter=Value("user1", "User One", email_address="user1@example.com"),
        assignee=None,
        labels=[],
        components=[],
        fixVersions=[],
        customfield_001=Value(None, "PROD"),
        customfield_002=Value(None, "Data"),
        changes=[],
    ),
    Issue(
        "D-5",
        summary="Debt 5",
        issuetype=Value("Bug", "Bug"),
        status=Value("Closed", "closed"),
        created="2018-02-05 01:01:01",
        updated="2018-02-20 02:02:02",
        resolution="Done",
        resolutiondate="2018-02-20 02:02:02",
        priority=Value("High", "High"),
        project=Value("PROJ", "PROJ"),
        reporter=Value("user1", "User One", email_address="user1@example.com"),
        assignee=None,
        labels=[],
        components=[],
        fixVersions=[],
        customfield_001=Value(None, "SIT"),
        customfield_002=Value(None, "Data"),
        changes=[],
    ),
    Issue(
        "D-6",
        summary="Debt 6",
        issuetype=Value("Bug", "Bug"),
        status=Value("Closed", "closed"),
        created="2018-03-06 01:01:01",
        updated="2018-03-06 01:01:01",
        resolution=None,
        resolutiondate=None,
        priority=Value("Medium", "Medium"),
        project=Value("PROJ", "PROJ"),
        reporter=Value("user1", "User One", email_address="user1@example.com"),
        assignee=None,
        labels=[],
        components=[],
        fixVersions=[],
        customfield_001=Value(None, "UAT"),
        customfield_002=Value(None, "Data"),
        changes=[],
    ),
]

# Common test issue for cycle time tests
COMMON_CYCLE_TIME_TEST_ISSUE = create_test_issue(
    "I-1",
    {
        "summary": "Just created",
        "issuetype": "Story",
        "status": "Backlog",
        "created": "2018-01-01 01:01:01",
        "changes": create_common_impediment_changes(),
    },
)
