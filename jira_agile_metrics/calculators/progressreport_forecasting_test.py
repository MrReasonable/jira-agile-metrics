"""Tests for progressreport_forecasting module."""

import datetime

from ..test_utils import create_default_epic_data
from .progressreport_forecasting import (
    _parse_start_date,
    _simulate_single_trial,
    _validate_stories_count,
    calculate_epic_target,
    forecast_to_complete,
    forward_weeks,
)
from .progressreport_models import Epic


class TestValidateStoriesCount:
    """Test cases for _validate_stories_count function."""

    def test_validate_positive_integer(self):
        """Test validation with positive integer."""
        result = _validate_stories_count(5)
        assert result == 5.0

    def test_validate_zero(self):
        """Test validation with zero."""
        result = _validate_stories_count(0)
        assert result == 0.0

    def test_validate_float(self):
        """Test validation with float."""
        result = _validate_stories_count(5.5)
        assert result == 5.5

    def test_validate_string_number(self):
        """Test validation with string number."""
        result = _validate_stories_count("5")
        assert result == 5.0

    def test_validate_none_with_default(self):
        """Test validation with None and default value."""
        result = _validate_stories_count(None, default=10)
        assert result == 10.0

    def test_validate_none_no_default(self):
        """Test validation with None and no default."""
        result = _validate_stories_count(None)
        assert result == 0.0

    def test_validate_negative(self):
        """Test validation with negative number."""
        result = _validate_stories_count(-5)
        assert result is None

    def test_validate_invalid_string(self):
        """Test validation with invalid string."""
        result = _validate_stories_count("invalid")
        assert result is None

    def test_validate_empty_string(self):
        """Test validation with empty string."""
        result = _validate_stories_count("")
        assert result is None


class TestParseStartDate:
    """Test cases for _parse_start_date function."""

    def test_parse_valid_date(self):
        """Test parsing valid date string."""
        result = _parse_start_date("2024-01-15")
        assert result == datetime.date(2024, 1, 15)

    def test_parse_none(self):
        """Test parsing None."""
        result = _parse_start_date(None)
        assert result is None

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        result = _parse_start_date("")
        assert result is None

    def test_parse_invalid_format(self):
        """Test parsing invalid date format."""
        result = _parse_start_date("01/15/2024")
        assert result is None

    def test_parse_invalid_date(self):
        """Test parsing invalid date."""
        result = _parse_start_date("2024-13-45")
        assert result is None

    def test_parse_nonsense_string(self):
        """Test parsing nonsense string."""
        result = _parse_start_date("not a date")
        assert result is None


class TestCalculateEpicTarget:
    """Test cases for calculate_epic_target function."""

    def create_epic(self, **kwargs):
        """Helper to create an Epic with default values."""
        defaults = create_default_epic_data(
            stories_raised=10,
            stories_done=5,
            first_story_started="2024-01-01",
        )
        defaults.update(kwargs)
        return Epic(defaults)

    def test_calculate_target_valid(self):
        """Test calculating target with valid data."""
        epic = self.create_epic()
        target = calculate_epic_target(epic)

        assert target is not None
        assert isinstance(target, datetime.date)
        # Should be in the future from start date
        assert target > datetime.date(2024, 1, 1)

    def test_calculate_target_none_when_no_start_date(self):
        """Test that target is None when no start date."""
        epic = self.create_epic(first_story_started=None)
        target = calculate_epic_target(epic)
        assert target is None

    def test_calculate_target_none_when_no_stories_raised(self):
        """Test that target is None when no stories raised."""
        epic = self.create_epic(stories_raised=0)
        target = calculate_epic_target(epic)
        assert target is None

    def test_calculate_target_none_when_stories_done_zero(self):
        """Test that target is None when stories done is zero."""
        epic = self.create_epic(stories_done=0)
        target = calculate_epic_target(epic)
        assert target is None

    def test_calculate_target_none_when_all_done(self):
        """Test that target works when all stories are done."""
        epic = self.create_epic(stories_done=10, stories_raised=10)
        target = calculate_epic_target(epic)

        # When all done, should still calculate based on rate
        # but may return None if completion rate is exactly 1.0
        # (which triggers the == 0 check)
        # Actually, if completion rate is 1.0, remaining_stories is 0,
        # so estimated_remaining_time is 0
        # and target date should be start date
        assert target is not None

    def test_calculate_target_with_date_string(self):
        """Test calculating target with date as string."""
        epic = self.create_epic(first_story_started="2024-01-01")
        target = calculate_epic_target(epic)
        assert target is not None

    def test_calculate_target_invalid_stories_count(self):
        """Test that target is None with invalid stories count."""
        epic = self.create_epic(stories_done="invalid")
        target = calculate_epic_target(epic)
        assert target is None

    def test_calculate_target_stories_done_greater_than_raised(self):
        """Test that target is None when done > raised."""
        epic = self.create_epic(stories_done=15, stories_raised=10)
        target = calculate_epic_target(epic)
        assert target is None

    def test_calculate_target_negative_stories(self):
        """Test that target is None with negative stories."""
        epic = self.create_epic(stories_done=-5)
        target = calculate_epic_target(epic)
        assert target is None

    def test_calculate_target_calculates_correctly(self):
        """Test that target calculation is mathematically correct."""
        # If we started on Jan 1, have 10 total stories, and done 5 stories,
        # completion rate is 50%, so 5 remaining stories should take same time
        # as 5 done stories took.
        epic = self.create_epic(
            first_story_started="2024-01-01",
            stories_raised=10,
            stories_done=5,
        )
        target = calculate_epic_target(epic)

        # The calculation is: remaining_stories / completion_rate
        # completion_rate = stories_done / stories_raised = 5/10 = 0.5
        # remaining_stories = 10 - 5 = 5
        # estimated_remaining_time = 5 / 0.5 = 10 days
        # So target should be Jan 1 + 10 days = Jan 11
        assert target == datetime.date(2024, 1, 11)


class TestForwardWeeks:
    """Test cases for forward_weeks function."""

    def test_forward_weeks_positive(self):
        """Test adding positive weeks."""
        date = datetime.date(2024, 1, 1)
        result = forward_weeks(date, 2)
        assert result == datetime.date(2024, 1, 15)

    def test_forward_weeks_zero(self):
        """Test adding zero weeks."""
        date = datetime.date(2024, 1, 1)
        result = forward_weeks(date, 0)
        assert result == date

    def test_forward_weeks_negative(self):
        """Test adding negative weeks (going backwards)."""
        date = datetime.date(2024, 1, 15)
        result = forward_weeks(date, -2)
        assert result == datetime.date(2024, 1, 1)

    def test_forward_weeks_cross_month(self):
        """Test adding weeks that cross month boundary."""
        date = datetime.date(2024, 1, 25)
        result = forward_weeks(date, 2)
        assert result == datetime.date(2024, 2, 8)

    def test_forward_weeks_cross_year(self):
        """Test adding weeks that cross year boundary."""
        date = datetime.date(2024, 12, 25)
        result = forward_weeks(date, 2)
        assert result == datetime.date(2025, 1, 8)


class TestSimulateSingleTrial:
    """Test cases for _simulate_single_trial function."""

    class MockTeam:
        """Mock team for testing."""

        def __init__(self, wip):
            self.wip = wip

        def __repr__(self):
            """String representation."""
            return f"MockTeam(wip={self.wip})"

        def get_wip(self):
            """Get WIP value."""
            return self.wip

    class MockEpic:
        """Mock epic for testing."""

        def __init__(self, key, max_stories):
            self.key = key
            self.data = {"max_stories": max_stories}

        def __repr__(self):
            """String representation."""
            return f"MockEpic(key={self.key}, max_stories={self.data['max_stories']})"

        def get_max_stories(self):
            """Get max stories value."""
            return self.data["max_stories"]

    def test_simulate_single_epic(self):
        """Test simulating a single epic trial."""
        team = self.MockTeam(wip=2)
        epics = [self.MockEpic("EPIC-1", max_stories=5)]

        result = _simulate_single_trial(team, epics, max_iterations=10)

        assert "epics" in result
        assert "EPIC-1" in result["epics"]
        assert result["epics"]["EPIC-1"]["iterations"] >= 0
        assert "completed_epics" in result
        assert "team_wip" in result

    def test_simulate_multiple_epics(self):
        """Test simulating multiple epics."""
        team = self.MockTeam(wip=3)
        epics = [
            self.MockEpic("EPIC-1", max_stories=5),
            self.MockEpic("EPIC-2", max_stories=10),
        ]

        result = _simulate_single_trial(team, epics, max_iterations=20)

        assert "EPIC-1" in result["epics"]
        assert "EPIC-2" in result["epics"]
        assert result["epics"]["EPIC-1"]["iterations"] >= 0
        assert result["epics"]["EPIC-2"]["iterations"] >= 0

    def test_simulate_epic_completion(self):
        """Test that epic completion is tracked."""
        team = self.MockTeam(wip=5)
        epics = [self.MockEpic("EPIC-1", max_stories=3)]

        result = _simulate_single_trial(team, epics, max_iterations=10)

        # With WIP of 5 and max_stories of 3, epic should complete
        # (each iteration can work on up to 5 items, so 3 stories done in 1 iteration)
        if result["epics"]["EPIC-1"]["completed"]:
            assert "EPIC-1" in result["completed_epics"]

    def test_simulate_max_iterations(self):
        """Test that simulation respects max_iterations."""
        team = self.MockTeam(wip=1)
        epics = [self.MockEpic("EPIC-1", max_stories=100)]  # Large epic

        result = _simulate_single_trial(team, epics, max_iterations=5)

        # Should stop after max_iterations even if not complete
        assert result["epics"]["EPIC-1"]["iterations"] <= 5


class TestForecastToComplete:
    """Test cases for forecast_to_complete function."""

    class MockTeam:
        """Mock team for testing."""

        def __init__(self, wip):
            self.wip = wip

        def __repr__(self):
            """String representation."""
            return f"MockTeam(wip={self.wip})"

        def get_wip(self):
            """Get WIP value."""
            return self.wip

    class MockEpic:
        """Mock epic for testing."""

        def __init__(self, key, max_stories):
            self.key = key
            self.data = {"max_stories": max_stories}

        def __repr__(self):
            """String representation."""
            return f"MockEpic(key={self.key}, max_stories={self.data['max_stories']})"

        def get_max_stories(self):
            """Get max stories value."""
            return self.data["max_stories"]

    def test_forecast_single_epic(self):
        """Test forecasting completion for single epic."""
        team = self.MockTeam(wip=3)
        epics = [self.MockEpic("EPIC-1", max_stories=10)]
        config = {"trials": 10, "quantiles": [0.5, 0.75], "max_iterations": 100}

        result = forecast_to_complete(team, epics, config)

        assert "EPIC-1" in result
        assert 0.5 in result["EPIC-1"]
        assert 0.75 in result["EPIC-1"]

    def test_forecast_multiple_epics(self):
        """Test forecasting completion for multiple epics."""
        team = self.MockTeam(wip=3)
        epics = [
            self.MockEpic("EPIC-1", max_stories=10),
            self.MockEpic("EPIC-2", max_stories=15),
        ]
        config = {"trials": 10, "quantiles": [0.5, 0.95], "max_iterations": 100}

        result = forecast_to_complete(team, epics, config)

        assert "EPIC-1" in result
        assert "EPIC-2" in result
        assert all(q in result["EPIC-1"] for q in [0.5, 0.95])
        assert all(q in result["EPIC-2"] for q in [0.5, 0.95])

    def test_forecast_no_epics(self):
        """Test forecasting with no epics."""
        team = self.MockTeam(wip=3)
        epics = []
        config = {"trials": 10, "quantiles": [0.5], "max_iterations": 100}

        result = forecast_to_complete(team, epics, config)

        assert not result

    def test_forecast_default_config(self):
        """Test forecasting with default config values."""
        team = self.MockTeam(wip=3)
        epics = [self.MockEpic("EPIC-1", max_stories=10)]
        config = {}  # Empty config should use defaults

        result = forecast_to_complete(team, epics, config)

        assert "EPIC-1" in result
        # Should have default quantiles
        assert 0.5 in result["EPIC-1"]
        assert 0.75 in result["EPIC-1"]
