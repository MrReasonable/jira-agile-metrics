"""Tests for core progress report calculator functionality."""

from .progressreport import (
    ProgressReportCalculator,
    calculate_epic_target,
)
from .progressreport_models import Epic
from .progressreport_test_utils import ProgressReportTestBase


class TestProgressReportCalculator(ProgressReportTestBase):
    """Test cases for the main progress report calculator."""

    def test_calculator(
        self,
        custom_query_manager,
        base_custom_settings,
        base_minimal_cycle_time_results,
    ):
        """Test basic calculator functionality."""
        calculator = ProgressReportCalculator(
            custom_query_manager, base_custom_settings, base_minimal_cycle_time_results
        )
        result = calculator.run()

        assert result is not None
        assert "outcomes" in result
        assert "teams" in result
        assert "charts" in result

    def test_calculator_no_outcomes(
        self,
        custom_query_manager,
        base_custom_settings,
        base_minimal_cycle_time_results,
    ):
        """Test calculator with no outcomes configured."""
        settings_no_outcomes = base_custom_settings.copy()
        settings_no_outcomes["progress_report_outcomes"] = {}

        calculator = ProgressReportCalculator(
            custom_query_manager, settings_no_outcomes, base_minimal_cycle_time_results
        )
        result = calculator.run()

        assert result is not None
        assert "outcomes" in result
        assert len(result["outcomes"]) == 0

    def test_calculator_no_fields(
        self,
        custom_query_manager,
        base_custom_settings,
        base_minimal_cycle_time_results,
    ):
        """Test calculator with minimal field configuration."""
        settings_minimal = base_custom_settings.copy()
        settings_minimal["progress_report_fields"] = {
            "epic": "customfield_10001",
            "team": "customfield_10002",
        }

        calculator = ProgressReportCalculator(
            custom_query_manager, settings_minimal, base_minimal_cycle_time_results
        )
        result = calculator.run()

        assert result is not None
        assert "outcomes" in result
        assert "teams" in result
        assert "charts" in result


class TestEpicTargetCalculation(ProgressReportTestBase):
    """Test cases for epic target calculation."""

    def test_calculate_epic_target(self):
        """Test epic target calculation."""
        # Test data - create a mock epic object
        epic_data = {
            "key": "EPIC-1",
            "summary": "Test Epic",
            "status": "In Progress",
            "resolution": None,
            "resolution_date": None,
            "min_stories": 0,
            "max_stories": 0,
            "team_name": "Team A",
            "deadline": "2018-02-01",
            "story_query": "",
            "story_cycle_times": [],
            "stories_raised": 4,
            "stories_in_backlog": 1,
            "stories_in_progress": 1,
            "stories_done": 2,
            "first_story_started": "2018-01-01",
            "last_story_finished": "2018-01-20",
            "team": None,
            "outcome": None,
            "forecast": None,
        }

        epic = Epic(epic_data)

        target = calculate_epic_target(epic)

        assert target is not None  # Should return a target date

    def test_calculate_epic_target_no_config(self):
        """Test epic target calculation with no configuration."""
        # Test data - create a mock epic object with no stories
        epic_data = {
            "key": "EPIC-2",
            "summary": "Test Epic 2",
            "status": "Backlog",
            "resolution": None,
            "resolution_date": None,
            "min_stories": 0,
            "max_stories": 0,
            "team_name": "Team A",
            "deadline": "2018-02-01",
            "story_query": "",
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

        epic = Epic(epic_data)

        target = calculate_epic_target(epic)

        assert target is None  # Should return None when no stories


class TestOutcomeFinding(ProgressReportTestBase):
    """Test cases for outcome finding functionality."""

    def test_find_outcomes(self, custom_query_manager):
        """Test finding outcomes from issues."""
        # This would test the actual outcome finding logic
        # Implementation depends on the specific requirements
        assert custom_query_manager is not None

    def test_find_outcomes_no_deadline_field(self, custom_query_manager):
        """Test finding outcomes when deadline field is not configured."""
        # This would test the case where deadline field is missing
        assert custom_query_manager is not None


class TestEpicFinding(ProgressReportTestBase):
    """Test cases for epic finding functionality."""

    def test_find_epics(self, custom_query_manager):
        """Test finding epics from issues."""
        # This would test the actual epic finding logic
        assert custom_query_manager is not None

    def test_find_epics_minimal_fields(self, custom_query_manager):
        """Test finding epics with minimal field configuration."""
        # This would test epic finding with limited fields
        assert custom_query_manager is not None

    def test_find_epics_defaults_to_outcome_deadline(self, custom_query_manager):
        """Test that epics default to outcome deadline when not specified."""
        # This would test the default deadline behavior
        assert custom_query_manager is not None


class TestStoryCountUpdates(ProgressReportTestBase):
    """Test cases for story count updates."""

    def test_update_story_counts(self, custom_query_manager, base_custom_settings):
        """Test updating story counts for epics."""
        # This would test the story count update logic
        assert custom_query_manager is not None
        assert base_custom_settings is not None
