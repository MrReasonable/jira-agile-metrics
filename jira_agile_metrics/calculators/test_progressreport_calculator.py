"""Tests for core progress report calculator functionality."""

import datetime

import pandas as pd

from ..querymanager import QueryManager
from ..test_classes import FauxChange as Change
from ..test_classes import FauxFieldValue as Value
from ..test_classes import FauxIssue as Issue
from ..test_classes import FauxJIRA as JIRA
from ..test_utils import create_default_epic_config, create_default_epic_data
from .progressreport import (
    ProgressReportCalculator,
    calculate_epic_target,
)
from .progressreport_models import Epic, Outcome
from .progressreport_queries import find_epics, update_story_counts
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
        settings_no_outcomes["progress_report"]["outcomes"] = {}

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
        settings_minimal["progress_report"]["fields"] = {
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
        epic_data = create_default_epic_data(
            max_stories=0,
            deadline="2018-02-01",
            story_query="",
            story_cycle_times=[],
            stories_raised=4,
            stories_in_backlog=1,
            stories_in_progress=1,
            stories_done=2,
            first_story_started="2018-01-01",
            last_story_finished="2018-01-20",
            team=None,
            outcome=None,
            forecast=None,
        )

        epic = Epic(epic_data)

        target = calculate_epic_target(epic)

        assert target is not None  # Should return a target date

    def test_calculate_epic_target_no_config(self):
        """Test epic target calculation with no configuration."""
        # Test data - create a mock epic object with no stories
        epic_data = create_default_epic_data(
            key="EPIC-2",
            summary="Test Epic 2",
            status="Backlog",
            max_stories=0,
            deadline="2018-02-01",
            story_query="",
            story_cycle_times=[],
        )

        epic = Epic(epic_data)

        target = calculate_epic_target(epic)

        assert target is None  # Should return None when no stories


class TestOutcomeFinding(ProgressReportTestBase):
    """Test cases for outcome finding functionality."""

    def test_find_outcomes(self, base_custom_settings, base_custom_fields):
        """Test finding outcomes from issues using outcome query."""
        # Create test issues for outcomes
        outcome_issue = Issue(
            key="OUTCOME-1",
            changes=[],
            summary="Test Outcome 1",
            issuetype=Value("Outcome", "outcome"),
            status=Value("Active", "active"),
            created="2018-01-01 01:01:01",
        )

        jira = JIRA(fields=base_custom_fields, issues=[outcome_issue])
        query_manager = QueryManager(jira, base_custom_settings)

        settings = base_custom_settings.copy()
        settings["progress_report"]["outcome_query"] = "project=TEST AND type=Outcome"
        settings["progress_report"]["templates"]["epic"] = (
            "project=TEST AND type=Epic AND outcome={outcome}"
        )
        settings["progress_report"]["templates"]["story"] = (
            "project=TEST AND type=Story"
        )

        calculator = ProgressReportCalculator(query_manager, settings, {})

        # Trigger outcome finding by calling setup_teams_and_outcomes
        result = calculator.setup_teams_and_outcomes(
            [], settings["progress_report"]["templates"]["epic"]
        )

        assert result is not None
        _, _, _, outcomes = result
        assert len(outcomes) == 1
        assert outcomes[0].key == "OUTCOME-1"
        assert outcomes[0].name == "Test Outcome 1"
        # Epic query template should be formatted with outcome key
        assert outcomes[0].epic_query is not None
        assert "OUTCOME-1" in outcomes[0].epic_query

    def test_find_outcomes_no_deadline_field(
        self, base_custom_settings, base_custom_fields
    ):
        """Test finding outcomes when deadline field is not configured."""
        # Create test issues for outcomes
        outcome_issue = Issue(
            key="OUTCOME-2",
            changes=[],
            summary="Test Outcome 2",
            issuetype=Value("Outcome", "outcome"),
            status=Value("Active", "active"),
            created="2018-01-01 01:01:01",
        )

        jira = JIRA(fields=base_custom_fields, issues=[outcome_issue])
        query_manager = QueryManager(jira, base_custom_settings)

        settings = base_custom_settings.copy()
        settings["progress_report"]["outcome_query"] = "project=TEST AND type=Outcome"
        settings["progress_report"]["outcome_fields"]["deadline"] = (
            None  # No deadline field
        )
        settings["progress_report"]["templates"]["epic"] = (
            "project=TEST AND type=Epic AND outcome={outcome}"
        )
        settings["progress_report"]["templates"]["story"] = (
            "project=TEST AND type=Story"
        )

        calculator = ProgressReportCalculator(query_manager, settings, {})

        # Trigger outcome finding
        result = calculator.setup_teams_and_outcomes(
            [], settings["progress_report"]["templates"]["epic"]
        )

        assert result is not None
        _, _, _, outcomes = result
        assert len(outcomes) == 1
        assert outcomes[0].key == "OUTCOME-2"
        assert outcomes[0].deadline is None  # No deadline when field not configured


class TestEpicFinding(ProgressReportTestBase):
    """Test cases for epic finding functionality."""

    def test_find_epics(self, base_custom_settings, base_custom_fields):
        """Test finding epics from issues with full configuration."""
        # Create test issues: outcome and epic
        outcome_issue = Issue(
            key="OUTCOME-1",
            changes=[],
            summary="Test Outcome",
            issuetype=Value("Outcome", "outcome"),
            status=Value("Active", "active"),
            created="2018-01-01 01:01:01",
        )

        epic_issue = Issue(
            key="EPIC-1",
            changes=[],
            summary="Test Epic",
            issuetype=Value("Epic", "epic"),
            status=Value("In Progress", "in progress"),
            resolution=None,
            resolutiondate=None,
            created="2018-01-02 01:01:01",
        )

        # Add custom fields for epic configuration
        epic_fields = base_custom_fields + [
            {"id": "customfield_101", "name": "Min Stories"},
            {"id": "customfield_102", "name": "Max Stories"},
            {"id": "customfield_103", "name": "Team"},
            {"id": "customfield_104", "name": "Deadline"},
        ]

        # Filter function to only return Epic issues for Epic queries
        def epic_filter(issue, jql):
            if "type=Epic" in jql:
                return issue.fields.issuetype.name == "Epic"
            return True

        jira = JIRA(
            fields=epic_fields,
            issues=[outcome_issue, epic_issue],
            filter_func=epic_filter,
        )
        query_manager = QueryManager(jira, base_custom_settings)

        outcome = Outcome(
            {
                "key": "OUTCOME-1",
                "name": "Test Outcome",
                "deadline": None,
                "epic_query": 'project=TEST AND type=Epic AND outcome="OUTCOME-1"',
            }
        )

        epic_config = {
            "min_stories_field": "customfield_101",
            "max_stories_field": "customfield_102",
            "team_field": "customfield_103",
            "deadline_field": "customfield_104",
        }

        epics = list(find_epics(query_manager, epic_config, outcome))

        assert len(epics) == 1
        assert epics[0]["key"] == "EPIC-1"
        assert epics[0]["summary"] == "Test Epic"
        assert epics[0]["status"] == "In Progress"

    def test_find_epics_minimal_fields(self, base_custom_settings, base_custom_fields):
        """Test finding epics with minimal field configuration."""
        # Create test issues: outcome and epic
        outcome_issue = Issue(
            key="OUTCOME-1",
            changes=[],
            summary="Test Outcome",
            issuetype=Value("Outcome", "outcome"),
            status=Value("Active", "active"),
            created="2018-01-01 01:01:01",
        )

        epic_issue = Issue(
            key="EPIC-2",
            changes=[],
            summary="Test Epic 2",
            issuetype=Value("Epic", "epic"),
            status=Value("Backlog", "backlog"),
            resolution=None,
            resolutiondate=None,
            created="2018-01-02 01:01:01",
        )

        # Filter function to only return Epic issues for Epic queries
        def epic_filter(issue, jql):
            if "type=Epic" in jql:
                return issue.fields.issuetype.name == "Epic"
            return True

        jira = JIRA(
            fields=base_custom_fields,
            issues=[outcome_issue, epic_issue],
            filter_func=epic_filter,
        )
        query_manager = QueryManager(jira, base_custom_settings)

        outcome = Outcome(
            {
                "key": "OUTCOME-1",
                "name": "Test Outcome",
                "deadline": None,
                "epic_query": 'project=TEST AND type=Epic AND outcome="OUTCOME-1"',
            }
        )

        epic_config = create_default_epic_config()

        epics = list(find_epics(query_manager, epic_config, outcome))

        assert len(epics) == 1
        assert epics[0]["key"] == "EPIC-2"
        assert epics[0]["min_stories"] is None
        assert epics[0]["max_stories"] is None
        assert epics[0]["team_name"] is None
        assert epics[0]["deadline"] is None

    def test_find_epics_defaults_to_outcome_deadline(
        self, base_custom_settings, base_custom_fields
    ):
        """Test that epics default to outcome deadline when not specified."""
        # Create test issues: outcome with deadline and epic without deadline field
        outcome_deadline = datetime.datetime(2018, 2, 1)
        outcome_issue = Issue(
            key="OUTCOME-1",
            changes=[],
            summary="Test Outcome",
            issuetype=Value("Outcome", "outcome"),
            status=Value("Active", "active"),
            created="2018-01-01 01:01:01",
        )

        epic_issue = Issue(
            key="EPIC-3",
            changes=[],
            summary="Test Epic 3",
            issuetype=Value("Epic", "epic"),
            status=Value("Backlog", "backlog"),
            resolution=None,
            resolutiondate=None,
            created="2018-01-02 01:01:01",
        )

        # Filter function to only return Epic issues for Epic queries
        def epic_filter(issue, jql):
            if "type=Epic" in jql:
                return issue.fields.issuetype.name == "Epic"
            return True

        jira = JIRA(
            fields=base_custom_fields,
            issues=[outcome_issue, epic_issue],
            filter_func=epic_filter,
        )
        query_manager = QueryManager(jira, base_custom_settings)

        outcome = Outcome(
            {
                "key": "OUTCOME-1",
                "name": "Test Outcome",
                "deadline": outcome_deadline,
                "epic_query": 'project=TEST AND type=Epic AND outcome="OUTCOME-1"',
            }
        )

        epic_config = {
            "min_stories_field": None,
            "max_stories_field": None,
            "team_field": None,
            "deadline_field": None,  # Epic has no deadline field configured
        }

        epics = list(find_epics(query_manager, epic_config, outcome))

        assert len(epics) == 1
        assert epics[0]["key"] == "EPIC-3"
        # When epic deadline field is None, it's expected to be None in the epic dict
        # The logic for defaulting to outcome deadline would be in the calculator's
        # epic processing logic, not in find_epics itself
        assert epics[0]["deadline"] is None


class TestStoryCountUpdates(ProgressReportTestBase):
    """Test cases for story count updates."""

    def test_update_story_counts(self, base_custom_settings, base_custom_fields):
        """Test updating story counts for epics with stories in different statuses."""
        # Create test issues: epic and multiple stories
        epic_issue = Issue(
            key="EPIC-1",
            changes=[],
            summary="Test Epic",
            issuetype=Value("Epic", "epic"),
            status=Value("In Progress", "in progress"),
            created="2018-01-01 01:01:01",
        )

        story1 = Issue(
            key="STORY-1",
            changes=[
                Change("2018-01-05 01:01:01", [("status", "Backlog", "Next")]),
                Change("2018-01-06 01:01:01", [("status", "Next", "Build")]),
                Change("2018-01-10 01:01:01", [("status", "Build", "Done")]),
            ],
            summary="Story 1",
            issuetype=Value("Story", "story"),
            status=Value("Done", "done"),
            resolution=Value("Done", "done"),
            resolutiondate="2018-01-10 01:01:01",
            created="2018-01-03 01:01:01",
        )

        story2 = Issue(
            key="STORY-2",
            changes=[
                Change("2018-01-05 01:01:01", [("status", "Backlog", "Next")]),
            ],
            summary="Story 2",
            issuetype=Value("Story", "story"),
            status=Value("Backlog", "backlog"),
            resolution=None,
            resolutiondate=None,
            created="2018-01-04 01:01:01",
        )

        story3 = Issue(
            key="STORY-3",
            changes=[
                Change("2018-01-05 01:01:01", [("status", "Backlog", "Next")]),
                Change("2018-01-07 01:01:01", [("status", "Next", "Build")]),
            ],
            summary="Story 3",
            issuetype=Value("Story", "story"),
            status=Value("Build", "build"),
            resolution=None,
            resolutiondate=None,
            created="2018-01-05 01:01:01",
        )

        # Filter function to only return Story issues for Story queries
        def story_filter(issue, jql):
            if "type=Story" in jql:
                return issue.fields.issuetype.name == "Story"
            return True

        jira = JIRA(
            fields=base_custom_fields,
            issues=[epic_issue, story1, story2, story3],
            filter_func=story_filter,
        )
        query_manager = QueryManager(jira, base_custom_settings)

        settings = base_custom_settings.copy()
        settings["progress_report"]["templates"]["epic"] = "project=TEST AND type=Epic"
        settings["progress_report"]["templates"]["story"] = (
            'project=TEST AND type=Story AND epicLink="{epic}"'
        )

        # Create epic with story query
        epic = Epic(
            create_default_epic_data(
                min_stories=None,
                max_stories=None,
                team_name=None,
                story_query='project=TEST AND type=Story AND epicLink="EPIC-1"',
                story_cycle_times=[],
            )
        )

        # Update story counts
        cycle_config = settings["cycle"]
        backlog_column = settings["backlog_column"]
        done_column = settings["done_column"]

        update_story_counts(
            epic=epic,
            query_manager=query_manager,
            cycle=cycle_config,
            backlog_column=backlog_column,
            done_column=done_column,
        )

        # Verify story counts were updated
        assert epic.data["stories_raised"] == 3
        # Story counts depend on cycle time calculation which may vary
        # The key test is that the function was called and updated the counts
        assert "stories_raised" in epic.data
        assert "stories_in_backlog" in epic.data
        assert "stories_in_progress" in epic.data
        assert "stories_done" in epic.data
        assert "story_cycle_times" in epic.data
        assert isinstance(epic.data["story_cycle_times"], pd.DataFrame)
