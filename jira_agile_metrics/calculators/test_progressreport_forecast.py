"""Tests for forecast functionality in progress reports."""

from .progressreport_test_utils import ProgressReportTestBase
from .progressreport_utils import (
    forecast_to_complete,
    throughput_range_sampler,
)


class MockTeam:
    """Mock team class for testing."""

    def __init__(self):
        self.wip = 3
        self.throughput_config = {
            "min_throughput": 1,
            "max_throughput": 5,
            "sampler": throughput_range_sampler(1, 5),
        }

    def get_wip(self):
        """Get work in progress."""
        return self.wip

    def get_throughput_config(self):
        """Get throughput configuration."""
        return self.throughput_config


class MockEpic:
    """Mock epic class for testing."""

    def __init__(self, key, max_stories=10, team_ref=None):
        self.key = key
        self.team = team_ref
        self.data = {
            "max_stories": max_stories,
            "min_stories": max_stories // 2,
            "deadline": None,
            "first_story_started": None,
            "last_story_finished": None,
            "stories_raised": max_stories,
            "stories_in_backlog": max_stories // 2,
            "stories_in_progress": max_stories // 4,
            "stories_done": max_stories // 4,
        }

    def get_key(self):
        """Get epic key."""
        return self.key

    def get_data(self):
        """Get epic data."""
        return self.data


class TestForecastToComplete(ProgressReportTestBase):
    """Test cases for forecast to complete functionality."""

    def _create_mock_team(self):
        """Create a mock team object for testing."""
        return MockTeam()

    def _create_mock_epics(self):
        """Create mock epic objects for testing."""
        team = self._create_mock_team()
        return [MockEpic("EPIC-1", 10, team), MockEpic("EPIC-2", 15, team)]

    def test_mock_team_creation(self):
        """Test mock team creation functionality."""
        team = self._create_mock_team()
        assert hasattr(team, "wip")
        assert hasattr(team, "throughput_config")

    def test_forecast_to_complete_basic(self):
        """Test forecast to complete with basic parameters."""
        # Create mock team and epics for testing
        team = self._create_mock_team()
        epics = self._create_mock_epics()
        forecast_config = {"trials": 100, "quantiles": [0.5, 0.75, 0.85, 0.95]}

        forecast = forecast_to_complete(team, epics, forecast_config)

        assert forecast is not None
        assert isinstance(forecast, dict)

    def test_forecast_to_complete_no_epics(self):
        """Test forecast to complete with no epics."""
        team = self._create_mock_team()
        epics = []
        forecast_config = {"trials": 100, "quantiles": [0.5, 0.75, 0.85, 0.95]}

        forecast = forecast_to_complete(team, epics, forecast_config)

        assert forecast is not None
        assert isinstance(forecast, dict)
        assert len(forecast) == 0  # Should be empty when no epics

    def test_forecast_to_complete_with_randomness(self):
        """Test forecast to complete with randomness."""
        team = self._create_mock_team()
        epics = self._create_mock_epics()
        forecast_config = {"trials": 10, "quantiles": [0.5, 0.75, 0.85, 0.95]}

        # Run multiple forecasts to test randomness
        forecasts = []
        for _ in range(3):
            forecast = forecast_to_complete(team, epics, forecast_config)
            forecasts.append(forecast)

        # All forecasts should be dictionaries
        assert all(isinstance(f, dict) for f in forecasts)

    def test_forecast_team_validation(self):
        """Test forecast with team validation."""
        team = self._create_mock_team()
        epics = self._create_mock_epics()
        forecast_config = {"trials": 50, "quantiles": [0.5, 0.75, 0.85, 0.95]}

        # Test that team has required attributes
        assert hasattr(team, "wip")
        assert hasattr(team, "throughput_config")
        assert team.wip > 0
        assert "sampler" in team.throughput_config

        forecast = forecast_to_complete(team, epics, forecast_config)
        assert forecast is not None


class TestThroughputRangeSampler(ProgressReportTestBase):
    """Test cases for throughput range sampler functionality."""

    def test_sampler_validation(self):
        """Test sampler validation functionality."""
        sampler = throughput_range_sampler(1, 3)
        assert sampler is not None
        assert callable(sampler)

    def test_throughput_range_sampler(self):
        """Test throughput range sampler creation."""
        min_val = 1
        max_val = 3

        sampler = throughput_range_sampler(min_val, max_val)

        assert sampler is not None
        assert callable(sampler)

    def test_throughput_range_sampler_empty_data(self):
        """Test throughput range sampler with empty data."""
        sampler = throughput_range_sampler(None, None)

        assert sampler is None

    def test_throughput_range_sampler_single_value(self):
        """Test throughput range sampler with single value."""
        sampler = throughput_range_sampler(5, 5)

        assert sampler is not None
        assert callable(sampler)

    def test_throughput_range_sampler_consistency(self):
        """Test that throughput range sampler produces consistent results."""
        sampler = throughput_range_sampler(1, 3)

        # Sample multiple values
        samples = [sampler() for _ in range(100)]

        # All samples should be within the range
        assert all(1 <= sample <= 3 for sample in samples)

        # Should have some variation
        assert len(set(samples)) > 1

    def test_throughput_range_sampler_edge_cases(self):
        """Test throughput range sampler with edge cases."""
        # Test with zero range
        sampler_zero = throughput_range_sampler(5, 5)
        assert sampler_zero is not None
        assert callable(sampler_zero)

        # Test with negative values
        sampler_neg = throughput_range_sampler(-1, 1)
        assert sampler_neg is not None
        assert callable(sampler_neg)

        # Test with large range
        sampler_large = throughput_range_sampler(1, 1000)
        assert sampler_large is not None
        assert callable(sampler_large)


class TestForecastEdgeCases(ProgressReportTestBase):
    """Test cases for forecast edge cases."""

    def test_edge_case_validation(self):
        """Test edge case validation functionality."""
        team = self._create_mock_team()
        assert hasattr(team, "wip")
        assert team.wip > 0

    def _create_mock_team(self):
        """Create a mock team object for testing."""
        return MockTeam()

    def _create_mock_epics(self):
        """Create mock epic objects for testing."""
        team = self._create_mock_team()
        return [MockEpic("EPIC-1", 10, team), MockEpic("EPIC-2", 15, team)]

    def test_mock_epic_creation(self):
        """Test mock epic creation functionality."""
        epics = self._create_mock_epics()
        assert len(epics) == 2
        assert epics[0].key == "EPIC-1"
        assert epics[1].key == "EPIC-2"

    def test_forecast_empty_config(self):
        """Test forecast with empty configuration."""
        team = self._create_mock_team()
        epics = self._create_mock_epics()
        forecast_config = {}

        forecast = forecast_to_complete(team, epics, forecast_config)

        # Should handle empty config gracefully
        assert forecast is not None
        assert isinstance(forecast, dict)

    def test_forecast_minimal_config(self):
        """Test forecast with minimal configuration."""
        team = self._create_mock_team()
        epics = self._create_mock_epics()
        forecast_config = {"trials": 1}

        forecast = forecast_to_complete(team, epics, forecast_config)

        assert forecast is not None
        assert isinstance(forecast, dict)

    def test_forecast_large_trials(self):
        """Test forecast with large number of trials."""
        team = self._create_mock_team()
        epics = self._create_mock_epics()
        forecast_config = {"trials": 10000, "quantiles": [0.5, 0.75, 0.85, 0.95]}

        forecast = forecast_to_complete(team, epics, forecast_config)

        assert forecast is not None
        assert isinstance(forecast, dict)

    def test_forecast_custom_quantiles(self):
        """Test forecast with custom quantiles."""
        team = self._create_mock_team()
        epics = self._create_mock_epics()
        forecast_config = {"trials": 100, "quantiles": [0.25, 0.5, 0.75, 0.9, 0.95]}

        forecast = forecast_to_complete(team, epics, forecast_config)

        assert forecast is not None
        assert isinstance(forecast, dict)

    def test_forecast_performance(self):
        """Test forecast performance with different trial counts."""
        team = self._create_mock_team()
        epics = self._create_mock_epics()

        # Test with small number of trials
        forecast_config_small = {"trials": 10, "quantiles": [0.5, 0.75, 0.85, 0.95]}
        forecast_small = forecast_to_complete(team, epics, forecast_config_small)
        assert forecast_small is not None

        # Test with medium number of trials
        forecast_config_medium = {"trials": 100, "quantiles": [0.5, 0.75, 0.85, 0.95]}
        forecast_medium = forecast_to_complete(team, epics, forecast_config_medium)
        assert forecast_medium is not None

        # Both should be valid forecasts
        assert isinstance(forecast_small, dict)
        assert isinstance(forecast_medium, dict)
