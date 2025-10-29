"""Test utilities and fixtures for progress report tests."""

import random
from datetime import timedelta

import pytest

from ..querymanager import QueryManager
from ..test_classes import FauxJIRA as JIRA


class ProgressReportTestBase:
    """Base class for progress report tests with common setup."""

    @staticmethod
    def create_test_field_lookup(fields):
        """Create field lookup for testing."""
        field_lookup = {}
        for field in fields:
            field_lookup[field["name"]] = field["key"]
        return field_lookup

    @staticmethod
    def create_test_query_functions(field_lookup):
        """Create test query functions."""

        def local_compare_value(i, clause):
            return compare_value(i, clause, field_lookup)

        def local_simple_ql(i, jql):
            return simple_ql(i, jql, field_lookup)

        return local_compare_value, local_simple_ql

    @staticmethod
    def create_test_teams_config():
        """Create test teams configuration."""
        return {
            "Team A": {
                "throughput_window_end": "2018-01-15",
                "throughput_window_start": "2018-01-01",
                "wip_window_end": "2018-01-15",
                "wip_window_start": "2018-01-01",
            },
            "Team B": {
                "throughput_window_end": "2018-01-15",
                "throughput_window_start": "2018-01-01",
                "wip_window_end": "2018-01-15",
                "wip_window_start": "2018-01-01",
            },
        }

    @staticmethod
    def create_test_outcomes_config(today):
        """Create test outcomes configuration."""
        return {
            "Outcome 1": {
                "deadline": (today + timedelta(days=30)).strftime("%Y-%m-%d"),
                "target": 100,
                "epics": ["Epic 1", "Epic 2"],
            },
            "Outcome 2": {
                "deadline": (today + timedelta(days=60)).strftime("%Y-%m-%d"),
                "target": 200,
                "epics": ["Epic 3", "Epic 4"],
            },
        }

    @staticmethod
    def create_test_epics_config(today):
        """Create test epics configuration."""
        return {
            "Epic 1": {
                "deadline": (today + timedelta(days=20)).strftime("%Y-%m-%d"),
                "target": 50,
                "team": "Team A",
            },
            "Epic 2": {
                "deadline": (today + timedelta(days=25)).strftime("%Y-%m-%d"),
                "target": 50,
                "team": "Team A",
            },
            "Epic 3": {
                "deadline": (today + timedelta(days=40)).strftime("%Y-%m-%d"),
                "target": 100,
                "team": "Team B",
            },
            "Epic 4": {
                "deadline": (today + timedelta(days=45)).strftime("%Y-%m-%d"),
                "target": 100,
                "team": "Team B",
            },
        }

    @staticmethod
    def create_test_stories_config():
        """Create test stories configuration."""
        return {
            "Story 1": {
                "epic": "Epic 1",
                "team": "Team A",
                "status": "Done",
                "created": "2018-01-01",
                "completed": "2018-01-10",
            },
            "Story 2": {
                "epic": "Epic 1",
                "team": "Team A",
                "status": "In Progress",
                "created": "2018-01-05",
            },
            "Story 3": {
                "epic": "Epic 2",
                "team": "Team A",
                "status": "Done",
                "created": "2018-01-02",
                "completed": "2018-01-12",
            },
            "Story 4": {
                "epic": "Epic 3",
                "team": "Team B",
                "status": "Done",
                "created": "2018-01-03",
                "completed": "2018-01-15",
            },
        }


def random_date_past(start, max_days):
    """Generate a random date in the past."""
    return start - timedelta(days=random.randint(1, max_days))


def random_date_future(start, max_days):
    """Generate a random date in the future."""
    return start + timedelta(days=random.randint(1, max_days))


@pytest.fixture
def fields_fixture(custom_fields):
    """Fixture for test fields."""
    return custom_fields


@pytest.fixture
def settings(custom_settings):
    """Fixture for test settings."""
    return custom_settings


@pytest.fixture
def query_manager(fields_fixture_param, test_settings):
    """Fixture for query manager."""
    # Use fields_fixture_param to avoid unused argument warning
    _ = fields_fixture_param
    return QueryManager(
        JIRA(fields=[], issues=[]),
        test_settings,
    )


@pytest.fixture
def results():
    """Fixture for test results."""
    return {}


def create_field_lookup(fields):
    """Create field lookup from fields fixture."""
    field_lookup = {}
    for field in fields:
        field_lookup[field["name"]] = field["key"]
    return field_lookup


def compare_value(i, clause, _field_lookup):
    """Compare value helper function."""
    if clause == "epic":
        return i.fields.epic
    if clause == "team":
        return i.fields.team
    if clause == "status":
        return i.fields.status
    if clause == "created":
        return i.fields.created
    if clause == "completed":
        return i.fields.completed
    return getattr(i.fields, clause, None)


def simple_ql(_i, _jql, _field_lookup):
    """Simple QL helper function."""
    # Simplified implementation for testing
    return True
