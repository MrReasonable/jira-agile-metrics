"""Tests for progressreport_queries module."""

import datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from ..test_utils import create_default_epic_config, create_default_story_counts
from .progressreport_queries import (
    _parse_date_field,
    _update_story_dates,
    _update_story_status_counts,
    date_value,
    find_epics,
    find_outcomes,
    int_or_none,
    update_story_counts,
)


class TestIntOrNone:
    """Test cases for int_or_none function."""

    def test_int_or_none_with_int(self):
        """Test conversion of integer."""
        assert int_or_none(5) == 5

    def test_int_or_none_with_string_int(self):
        """Test conversion of string integer."""
        assert int_or_none("5") == 5

    def test_int_or_none_with_none(self):
        """Test conversion of None."""
        assert int_or_none(None) is None

    def test_int_or_none_with_float(self):
        """Test conversion of float (truncates)."""
        assert int_or_none(5.7) == 5

    def test_int_or_none_with_invalid_string(self):
        """Test conversion of invalid string."""
        assert int_or_none("invalid") is None

    def test_int_or_none_with_empty_string(self):
        """Test conversion of empty string."""
        assert int_or_none("") is None


class TestParseDateField:
    """Test cases for _parse_date_field function."""

    def test_parse_date_field_valid_string(self):
        """Test parsing valid date string."""
        result = _parse_date_field("2024-01-15T10:30:00")
        assert isinstance(result, datetime.datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_parse_date_field_none(self):
        """Test parsing None."""
        assert _parse_date_field(None) is None

    def test_parse_date_field_empty_string(self):
        """Test parsing empty string."""
        assert _parse_date_field("") is None

    def test_parse_date_field_with_default(self):
        """Test parsing with default value."""
        default = datetime.datetime(2024, 1, 1)
        assert _parse_date_field(None, default=default) == default
        assert _parse_date_field("", default=default) == default

    def test_parse_date_field_invalid_string(self):
        """Test parsing invalid date string."""
        assert _parse_date_field("not a date") is None

    def test_parse_date_field_datetime_object(self):
        """Test parsing datetime object - should be parsed as string."""
        dt = datetime.datetime(2024, 1, 15, 10, 30, 0)
        # _parse_date_field uses dateutil.parser.parse which can handle
        # datetime objects
        result = _parse_date_field(dt)
        # The function will parse it successfully since dateutil can handle
        # datetime objects
        assert result is not None
        assert isinstance(result, datetime.datetime)


class TestDateValue:
    """Test cases for date_value function."""

    def test_date_value_with_field(self):
        """Test getting date value from field."""
        query_manager = Mock()
        query_manager.resolve_field_value.return_value = "2024-01-15T10:30:00"
        issue = Mock()

        result = date_value(query_manager, issue, "customfield_100")

        assert isinstance(result, datetime.datetime)
        query_manager.resolve_field_value.assert_called_once_with(
            issue, "customfield_100"
        )

    def test_date_value_no_field_name(self):
        """Test getting date with no field name."""
        query_manager = Mock()
        issue = Mock()

        result = date_value(query_manager, issue, None)

        assert result is None

    def test_date_value_empty_field_name(self):
        """Test getting date with empty field name."""
        query_manager = Mock()
        issue = Mock()

        result = date_value(query_manager, issue, "")

        assert result is None

    def test_date_value_with_default(self):
        """Test getting date with default value."""
        query_manager = Mock()
        query_manager.resolve_field_value.return_value = None
        issue = Mock()
        default = datetime.datetime(2024, 1, 1)

        result = date_value(query_manager, issue, "customfield_100", default=default)

        assert result == default


class TestUpdateStoryStatusCounts:
    """Test cases for _update_story_status_counts function."""

    def test_update_backlog_status(self):
        """Test updating backlog status count."""
        epic = Mock()
        epic.data = {
            "stories_in_backlog": 0,
            "stories_in_progress": 0,
            "stories_done": 0,
        }
        row = {"status": "Backlog", "started": None, "completed": None}

        _update_story_status_counts(epic, row, "Backlog", "Done")

        assert epic.data["stories_in_backlog"] == 1
        assert epic.data["stories_in_progress"] == 0
        assert epic.data["stories_done"] == 0

    def test_update_done_status(self):
        """Test updating done status count."""
        epic = Mock()
        epic.data = {
            "stories_in_backlog": 0,
            "stories_in_progress": 0,
            "stories_done": 0,
        }
        row = {"status": "Done", "started": None, "completed": None}

        _update_story_status_counts(epic, row, "Backlog", "Done")

        assert epic.data["stories_in_backlog"] == 0
        assert epic.data["stories_in_progress"] == 0
        assert epic.data["stories_done"] == 1

    def test_update_in_progress_status(self):
        """Test updating in-progress status count."""
        epic = Mock()
        epic.data = {
            "stories_in_backlog": 0,
            "stories_in_progress": 0,
            "stories_done": 0,
        }
        row = {"status": "In Progress", "started": None, "completed": None}

        _update_story_status_counts(epic, row, "Backlog", "Done")

        assert epic.data["stories_in_backlog"] == 0
        assert epic.data["stories_in_progress"] == 1
        assert epic.data["stories_done"] == 0


class TestUpdateStoryDates:
    """Test cases for _update_story_dates function."""

    def test_update_first_story_started(self):
        """Test updating first story started date."""
        epic = Mock()
        epic.data = {"first_story_started": None, "last_story_finished": None}
        row = {"started": pd.Timestamp("2024-01-15"), "completed": pd.NaT}

        _update_story_dates(epic, row, "started", "completed")

        assert epic.data["first_story_started"] == pd.Timestamp("2024-01-15")

    def test_update_first_story_started_earlier(self):
        """Test updating first story started with earlier date."""
        epic = Mock()
        epic.data = {
            "first_story_started": pd.Timestamp("2024-01-20"),
            "last_story_finished": None,
        }
        row = {"started": pd.Timestamp("2024-01-15"), "completed": pd.NaT}

        _update_story_dates(epic, row, "started", "completed")

        assert epic.data["first_story_started"] == pd.Timestamp("2024-01-15")

    def test_update_first_story_started_later(self):
        """Test not updating first story started with later date."""
        epic = Mock()
        epic.data = {
            "first_story_started": pd.Timestamp("2024-01-10"),
            "last_story_finished": None,
        }
        row = {"started": pd.Timestamp("2024-01-15"), "completed": pd.NaT}

        _update_story_dates(epic, row, "started", "completed")

        assert epic.data["first_story_started"] == pd.Timestamp("2024-01-10")

    def test_update_last_story_finished(self):
        """Test updating last story finished date."""
        epic = Mock()
        epic.data = {"first_story_started": None, "last_story_finished": None}
        row = {"started": pd.NaT, "completed": pd.Timestamp("2024-01-20")}

        _update_story_dates(epic, row, "started", "completed")

        assert epic.data["last_story_finished"] == pd.Timestamp("2024-01-20")

    def test_update_last_story_finished_later(self):
        """Test updating last story finished with later date."""
        epic = Mock()
        epic.data = {
            "first_story_started": None,
            "last_story_finished": pd.Timestamp("2024-01-15"),
        }
        row = {"started": pd.NaT, "completed": pd.Timestamp("2024-01-20")}

        _update_story_dates(epic, row, "started", "completed")

        assert epic.data["last_story_finished"] == pd.Timestamp("2024-01-20")

    def test_update_both_dates(self):
        """Test updating both dates in one row."""
        epic = Mock()
        epic.data = {"first_story_started": None, "last_story_finished": None}
        row = {
            "started": pd.Timestamp("2024-01-15"),
            "completed": pd.Timestamp("2024-01-20"),
        }

        _update_story_dates(epic, row, "started", "completed")

        assert epic.data["first_story_started"] == pd.Timestamp("2024-01-15")
        assert epic.data["last_story_finished"] == pd.Timestamp("2024-01-20")

    def test_update_ignores_nan(self):
        """Test that NaN values are ignored."""
        epic = Mock()
        epic.data = {
            "first_story_started": pd.Timestamp("2024-01-10"),
            "last_story_finished": pd.Timestamp("2024-01-15"),
        }
        row = {"started": pd.NaT, "completed": pd.NaT}

        _update_story_dates(epic, row, "started", "completed")

        assert epic.data["first_story_started"] == pd.Timestamp("2024-01-10")
        assert epic.data["last_story_finished"] == pd.Timestamp("2024-01-15")


class TestFindOutcomes:
    """Test cases for find_outcomes function."""

    def test_find_outcomes_basic(self):
        """Test finding outcomes with basic configuration."""
        query_manager = Mock()
        issue1 = Mock()
        issue1.key = "OUTCOME-1"
        issue1.fields.summary = "Outcome 1"
        query_manager.find_issues.return_value = [issue1]
        query_manager.resolve_field_value.return_value = None

        outcomes = list(
            find_outcomes(
                query_manager,
                "project=TEST",
                None,
                "epic_query_template_{outcome}",
            )
        )

        assert len(outcomes) == 1
        assert outcomes[0].key == "OUTCOME-1"
        assert outcomes[0].name == "Outcome 1"

    def test_find_outcomes_with_deadline(self):
        """Test finding outcomes with deadline field."""
        query_manager = Mock()
        issue1 = Mock()
        issue1.key = "OUTCOME-1"
        issue1.fields.summary = "Outcome 1"
        query_manager.find_issues.return_value = [issue1]
        query_manager.resolve_field_value.return_value = "2024-12-31"

        outcomes = list(
            find_outcomes(
                query_manager,
                "project=TEST",
                "customfield_100",
                "epic_query_template_{outcome}",
            )
        )

        assert len(outcomes) == 1
        assert outcomes[0].deadline is not None

    def test_find_outcomes_no_template(self):
        """Test finding outcomes without epic query template."""
        query_manager = Mock()
        issue1 = Mock()
        issue1.key = "OUTCOME-1"
        issue1.fields.summary = "Outcome 1"
        query_manager.find_issues.return_value = [issue1]
        query_manager.resolve_field_value.return_value = None

        outcomes = list(find_outcomes(query_manager, "project=TEST", None, None))

        assert len(outcomes) == 1
        assert outcomes[0].epic_query is None

    def test_find_outcomes_multiple(self):
        """Test finding multiple outcomes."""
        query_manager = Mock()
        issue1 = Mock()
        issue1.key = "OUTCOME-1"
        issue1.fields.summary = "Outcome 1"
        issue2 = Mock()
        issue2.key = "OUTCOME-2"
        issue2.fields.summary = "Outcome 2"
        query_manager.find_issues.return_value = [issue1, issue2]
        query_manager.resolve_field_value.return_value = None

        outcomes = list(
            find_outcomes(
                query_manager,
                "project=TEST",
                None,
                "epic_query_template_{outcome}",
            )
        )

        assert len(outcomes) == 2
        assert outcomes[0].key == "OUTCOME-1"
        assert outcomes[1].key == "OUTCOME-2"


class TestFindEpics:
    """Test cases for find_epics function."""

    def test_find_epics_basic(self):
        """Test finding epics with basic configuration."""
        query_manager = Mock()
        issue = Mock()
        issue.key = "EPIC-1"
        issue.fields.summary = "Epic 1"
        issue.fields.status.name = "In Progress"
        issue.fields.resolution = None
        issue.fields.resolutiondate = None
        query_manager.find_issues.return_value = [issue]
        query_manager.resolve_field_value.return_value = None

        epic_config = create_default_epic_config()

        outcome = Mock()
        outcome.epic_query = "project=TEST AND type=Epic"

        epics = list(find_epics(query_manager, epic_config, outcome))

        assert len(epics) == 1
        assert epics[0]["key"] == "EPIC-1"
        assert epics[0]["summary"] == "Epic 1"
        assert epics[0]["status"] == "In Progress"

    def test_find_epics_with_config(self):
        """Test finding epics with configuration fields."""
        query_manager = Mock()
        issue = Mock()
        issue.key = "EPIC-2"
        issue.fields.summary = "Epic 2"
        issue.fields.status.name = "Done"
        issue.fields.resolution = "Fixed"
        issue.fields.resolutiondate = None
        query_manager.find_issues.return_value = [issue]
        query_manager.resolve_field_value.side_effect = [5, 10, "Team A", "2024-12-31"]

        epic_config = {
            "min_stories_field": "customfield_001",
            "max_stories_field": "customfield_002",
            "team_field": "customfield_003",
            "deadline_field": "customfield_004",
        }

        outcome = Mock()
        outcome.epic_query = "project=TEST AND type=Epic"

        epics = list(find_epics(query_manager, epic_config, outcome))

        assert len(epics) == 1
        assert epics[0]["min_stories"] == 5
        assert epics[0]["max_stories"] == 10


class TestUpdateStoryCounts:
    """Test cases for update_story_counts function."""

    @pytest.fixture
    def mock_query_manager(self):
        """Create a mock query manager."""
        query_manager = Mock()
        story1 = Mock()
        story1.key = "STORY-1"
        story2 = Mock()
        story2.key = "STORY-2"
        query_manager.find_issues.return_value = [story1, story2]
        return query_manager

    @pytest.fixture
    def mock_epic(self):
        """Create a mock epic."""
        epic = Mock()
        story_counts = create_default_story_counts()
        story_counts["story_query"] = "project=TEST AND epicLink=EPIC-1"
        epic.data = story_counts
        return epic

    def test_update_story_counts_no_query(self, mock_epic, mock_query_manager):
        """Test updating story counts with no story query."""
        mock_epic.data["story_query"] = None

        cycle_config = [
            {"name": "Backlog", "statuses": ["Backlog"]},
            {"name": "Done", "statuses": ["Done"]},
        ]

        update_story_counts(
            mock_epic,
            mock_query_manager,
            cycle_config,
            "Backlog",
            "Done",
        )

        assert mock_epic.data["stories_raised"] == 0
        mock_query_manager.find_issues.assert_not_called()

    def test_update_story_counts_with_stories(self, mock_epic, mock_query_manager):
        """Test updating story counts with stories."""
        # Mock cycle time calculation
        # Note: cycle_data should have columns matching cycle config names
        cycle_config = [
            {"name": "Backlog", "statuses": ["Backlog"]},
            {"name": "Committed", "statuses": ["Next"]},
            {"name": "Done", "statuses": ["Done"]},
        ]
        cycle_data = pd.DataFrame(
            [
                {
                    "key": "STORY-1",
                    "status": "Backlog",
                    "Committed": pd.NaT,
                    "Done": pd.NaT,
                },
                {
                    "key": "STORY-2",
                    "status": "Done",
                    "Committed": pd.Timestamp("2024-01-10"),
                    "Done": pd.Timestamp("2024-01-15"),
                },
            ]
        )

        with patch(
            "jira_agile_metrics.calculators.progressreport_queries."
            "calculate_cycle_times"
        ) as mock_calc:
            mock_calc.return_value = cycle_data

            update_story_counts(
                mock_epic,
                mock_query_manager,
                cycle_config,
                "Backlog",
                "Done",
            )

        assert mock_epic.data["stories_raised"] == 2
        assert mock_epic.data["stories_in_backlog"] == 1
        assert mock_epic.data["stories_done"] == 1
        assert mock_epic.data["stories_in_progress"] == 0

    def test_update_story_counts_empty_stories(self, mock_epic, mock_query_manager):
        """Test updating story counts with no stories."""
        mock_query_manager.find_issues.return_value = []

        with patch(
            "jira_agile_metrics.calculators.progressreport_queries."
            "calculate_cycle_times"
        ) as mock_calc:
            mock_calc.return_value = pd.DataFrame()

            update_story_counts(
                mock_epic,
                mock_query_manager,
                {"status1": "Backlog", "status2": "Done"},
                "Backlog",
                "Done",
            )

        assert mock_epic.data["stories_raised"] == 0
        assert isinstance(mock_epic.data["story_cycle_times"], pd.DataFrame)
        assert len(mock_epic.data["story_cycle_times"]) == 0
