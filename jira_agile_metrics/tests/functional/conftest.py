"""Pytest fixtures for functional tests."""

from pathlib import Path

import pytest

from jira_agile_metrics.querymanager import QueryManager
from jira_agile_metrics.test_file_jira_client import FileJiraClient
from jira_agile_metrics.tests.e2e.e2e_config import _get_standard_cycle_config


def fixtures_path(*parts):
    """Return path to test fixtures directory."""
    return str(Path(__file__).resolve().parent.parent.joinpath("fixtures", *parts))


@pytest.fixture()
def jira_client():
    """Create a FileJiraClient fixture using test data."""
    return FileJiraClient(fixtures_path("jira"))


@pytest.fixture()
def query_manager(request):
    """Create a QueryManager fixture with minimal settings.

    Args:
        request: Pytest request object to access fixtures.
    """
    client = request.getfixturevalue("jira_client")
    return QueryManager(
        client,
        settings={
            "attributes": {},
            "known_values": {},
            "max_results": False,
        },
    )


@pytest.fixture()
def simple_cycle_settings(tmp_path):
    """Create simple cycle time settings for functional tests."""
    output_csv = tmp_path / "cycletime.csv"
    settings = {
        "cycle": _get_standard_cycle_config(),
        "committed_column": "Committed",
        "done_column": "Done",
        "attributes": {},
        "queries": [{"jql": "project=TEST"}],
        "query_attribute": None,
        "cycle_time_data": [str(output_csv)],
    }
    return settings, output_csv


def get_burnup_base_settings(base_settings):
    """Get common settings dictionary for burnup-related tests.

    Returns a dictionary with CFD and burnup chart settings disabled,
    suitable for use in burnup and burnup forecast tests.
    """
    return {
        **base_settings,
        "cfd_data": [],
        "cfd_chart": None,
        "cfd_window": 0,
        "backlog_column": "Backlog",
        "cfd_chart_title": None,
        "burnup_chart": None,
        "burnup_window": 0,
        "burnup_chart_title": None,
        "done_column": "Done",
    }


# end of file
