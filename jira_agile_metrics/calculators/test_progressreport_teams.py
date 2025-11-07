"""Tests for team-related functionality in progress reports."""

from unittest.mock import Mock

import pytest

from ..config.exceptions import ConfigError
from ..config.progress_report_utils import to_progress_report_teams_list
from ..config.type_utils import force_int
from ..querymanager import QueryManager
from ..test_classes import FauxChange as Change
from ..test_classes import FauxFieldValue as Value
from ..test_classes import FauxIssue, FauxJIRA
from .progressreport import ProgressReportCalculator
from .progressreport_test_utils import ProgressReportTestBase
from .progressreport_utils import (
    calculate_team_throughput,
    update_team_sampler,
)


class TestTeamThroughput(ProgressReportTestBase):
    """Test cases for team throughput calculations."""

    @pytest.fixture
    def query_manager_with_data(self, base_custom_fields, base_custom_settings):
        """Query manager with test issues for data availability tests."""
        # Create test issues with proper status transitions within the throughput window
        # These issues transition from Backlog -> Next -> Build -> QA -> Done
        test_issues = [
            FauxIssue(
                key="TEST-1",
                changes=[
                    Change(
                        "2018-01-03 01:01:01",
                        [("status", "Backlog", "Next")],
                    ),
                    Change(
                        "2018-01-04 01:01:01",
                        [("status", "Next", "Build")],
                    ),
                    Change(
                        "2018-01-05 01:01:01",
                        [("status", "Build", "QA")],
                    ),
                    Change(
                        "2018-01-08 01:01:01",
                        [("status", "QA", "Done")],
                    ),
                ],
                status=Value("Done", "Done"),
                resolution=Value("Done", "Done"),
                resolutiondate="2018-01-08 01:01:01",
                summary="Completed Story 1",
                issuetype=Value("Story", "story"),
                created="2018-01-02 01:01:01",
            ),
            FauxIssue(
                key="TEST-2",
                changes=[
                    Change(
                        "2018-01-04 01:01:01",
                        [("status", "Backlog", "Next")],
                    ),
                    Change(
                        "2018-01-05 01:01:01",
                        [("status", "Next", "Build")],
                    ),
                    Change(
                        "2018-01-06 01:01:01",
                        [("status", "Build", "QA")],
                    ),
                    Change(
                        "2018-01-09 01:01:01",
                        [("status", "QA", "Done")],
                    ),
                ],
                status=Value("Done", "Done"),
                resolution=Value("Done", "Done"),
                resolutiondate="2018-01-09 01:01:01",
                summary="Completed Story 2",
                issuetype=Value("Story", "story"),
                created="2018-01-03 01:01:01",
            ),
        ]
        jira = FauxJIRA(fields=base_custom_fields, issues=test_issues)
        return QueryManager(jira, base_custom_settings)

    def test_calculate_team_throughput(
        self, query_manager_with_data, base_custom_settings
    ):
        """Test team throughput calculation."""
        # Test data setup
        # When using min/max throughput (without throughput_samples),
        # the function returns the midpoint of min and max as a static estimate.
        # Default min_throughput=0, max_throughput=10, so midpoint = 5.0
        team_config = {
            "name": "Test Team",
            "throughput_window_start": "2018-01-01",
            "throughput_window_end": "2018-01-15",
        }

        throughput = calculate_team_throughput(
            team_config, query_manager_with_data, base_custom_settings
        )

        # Expected throughput is midpoint of default min/max range (0+10)/2 = 5.0
        # Note: To get actual story count, configure throughput_samples instead
        assert throughput == 5.0

    def test_calculate_team_throughput_no_data(
        self, custom_query_manager, base_custom_settings
    ):
        """Test team throughput calculation with no completed stories."""
        team_config = {
            "throughput_window_start": "2018-01-01",
            "throughput_window_end": "2018-01-15",
        }

        # No completed stories data

        throughput = calculate_team_throughput(
            team_config, custom_query_manager, base_custom_settings
        )

        assert throughput == 0

    def test_calculate_team_throughput_outside_window(
        self, base_custom_fields, base_custom_settings
    ):
        """Test team throughput calculation with stories outside the window."""
        # Create issues with dates OUTSIDE the window
        # (before 2018-01-01 or after 2018-01-15)
        test_issues = [
            FauxIssue(
                key="TEST-OLD",
                changes=[
                    Change(
                        "2017-12-15 01:01:01",
                        [("status", "Backlog", "Next")],
                    ),
                    Change(
                        "2017-12-20 01:01:01",
                        [("status", "Next", "Build")],
                    ),
                    Change(
                        "2017-12-25 01:01:01",
                        [("status", "Build", "QA")],
                    ),
                    Change(
                        "2017-12-28 01:01:01",
                        [("status", "QA", "Done")],
                    ),
                ],
                status=Value("Done", "Done"),
                resolution=Value("Done", "Done"),
                resolutiondate="2017-12-28 01:01:01",  # Before window (2018-01-01)
                summary="Old Story",
                issuetype=Value("Story", "story"),
                created="2017-12-10 01:01:01",
            ),
            FauxIssue(
                key="TEST-FUTURE",
                changes=[
                    Change(
                        "2018-01-20 01:01:01",
                        [("status", "Backlog", "Next")],
                    ),
                    Change(
                        "2018-01-22 01:01:01",
                        [("status", "Next", "Build")],
                    ),
                    Change(
                        "2018-01-25 01:01:01",
                        [("status", "Build", "QA")],
                    ),
                    Change(
                        "2018-01-28 01:01:01",
                        [("status", "QA", "Done")],
                    ),
                ],
                status=Value("Done", "Done"),
                resolution=Value("Done", "Done"),
                resolutiondate="2018-01-28 01:01:01",  # After window (2018-01-15)
                summary="Future Story",
                issuetype=Value("Story", "story"),
                created="2018-01-18 01:01:01",
            ),
        ]
        jira = FauxJIRA(fields=base_custom_fields, issues=test_issues)
        query_manager = QueryManager(jira, base_custom_settings)

        team_config = {
            "throughput_window_start": "2018-01-01",
            "throughput_window_end": "2018-01-15",
        }

        throughput = calculate_team_throughput(
            team_config,
            query_manager,
            base_custom_settings,
        )

        assert throughput == 0


class TestTeamSampler(ProgressReportTestBase):
    """Test cases for team sampler functionality."""

    def _assert_sampler_created(self, team_config, initial_sampler):
        """Helper method to validate that a sampler was created correctly.

        Args:
            team_config: The team configuration object
            initial_sampler: The initial sampler value before update
        """
        assert "sampler" in team_config.throughput_config, (
            "Sampler should be created in throughput_config"
        )
        assert team_config.throughput_config["sampler"] is not None, (
            "Sampler should not be None"
        )
        assert team_config.throughput_config["sampler"] != initial_sampler, (
            "Sampler should be different from initial state"
        )
        assert callable(team_config.throughput_config["sampler"]), (
            "Sampler should be callable"
        )

    @pytest.fixture
    def query_manager_for_sampler(self, base_custom_fields, base_custom_settings):
        """Query manager with test issues for sampler tests."""
        test_issues = [
            FauxIssue(
                key="STORY-1",
                changes=[
                    Change(
                        "2018-01-04 01:01:01",
                        [("status", "Backlog", "Next")],
                    ),
                    Change(
                        "2018-01-05 01:01:01",
                        [("status", "Next", "Build")],
                    ),
                    Change(
                        "2018-01-06 01:01:01",
                        [("status", "Build", "QA")],
                    ),
                    Change(
                        "2018-01-09 01:01:01",
                        [("status", "QA", "Done")],
                    ),
                ],
                status=Value("Done", "Done"),
                resolution=Value("Done", "Done"),
                resolutiondate="2018-01-09 01:01:01",
                summary="Sampler Test Story 1",
                issuetype=Value("Story", "story"),
                created="2018-01-03 01:01:01",
            ),
            FauxIssue(
                key="STORY-2",
                changes=[
                    Change(
                        "2018-01-05 01:01:01",
                        [("status", "Backlog", "Next")],
                    ),
                    Change(
                        "2018-01-06 01:01:01",
                        [("status", "Next", "Build")],
                    ),
                    Change(
                        "2018-01-07 01:01:01",
                        [("status", "Build", "QA")],
                    ),
                    Change(
                        "2018-01-10 01:01:01",
                        [("status", "QA", "Done")],
                    ),
                ],
                status=Value("Done", "Done"),
                resolution=Value("Done", "Done"),
                resolutiondate="2018-01-10 01:01:01",
                summary="Sampler Test Story 2",
                issuetype=Value("Story", "story"),
                created="2018-01-04 01:01:01",
            ),
            FauxIssue(
                key="STORY-3",
                changes=[
                    Change(
                        "2018-01-06 01:01:01",
                        [("status", "Backlog", "Next")],
                    ),
                    Change(
                        "2018-01-07 01:01:01",
                        [("status", "Next", "Build")],
                    ),
                    Change(
                        "2018-01-08 01:01:01",
                        [("status", "Build", "QA")],
                    ),
                    Change(
                        "2018-01-11 01:01:01",
                        [("status", "QA", "Done")],
                    ),
                ],
                status=Value("Done", "Done"),
                resolution=Value("Done", "Done"),
                resolutiondate="2018-01-11 01:01:01",
                summary="Sampler Test Story 3",
                issuetype=Value("Story", "story"),
                created="2018-01-05 01:01:01",
            ),
        ]
        jira = FauxJIRA(fields=base_custom_fields, issues=test_issues)
        return QueryManager(jira, base_custom_settings)

    def test_update_team_sampler(self, query_manager_for_sampler, base_custom_settings):
        """Test updating team sampler with new throughput data."""

        # (1) Set clear initial state - create mock object with throughput_config
        mock_team = Mock()
        mock_team.throughput_config = {
            "throughput_samples": [],  # Empty list so it falls back to min/max
            "min_throughput": 1,
            "max_throughput": 5,
        }

        team_config = mock_team
        initial_sampler = team_config.throughput_config.get("sampler")

        # (2) Arrange - query_manager_for_sampler already has controlled throughput data
        # (non-empty with multiple completed stories)
        update_team_sampler(
            team_config, query_manager_for_sampler, base_custom_settings
        )

        # (4) Assert - verify sampler was created/mutated as expected
        self._assert_sampler_created(team_config, initial_sampler)

        # Verify it's a range sampler (since throughput_samples is empty)
        # Range sampler should return values between min and max
        sampler_func = team_config.throughput_config["sampler"]
        test_values = [sampler_func() for _ in range(10)]
        assert all(1 <= val <= 5 for val in test_values), (
            "Sampler should return values in range [1, 5]"
        )

    def test_update_team_sampler_empty_data(
        self, custom_query_manager, base_custom_settings
    ):
        """Test updating team sampler with empty throughput data."""

        # (1) Set clear initial state - create mock object with throughput_config
        mock_team = Mock()
        mock_team.throughput_config = {
            "throughput_samples": [],  # Empty list
            "min_throughput": 0,
            "max_throughput": 10,
        }

        team_config = mock_team
        initial_sampler = team_config.throughput_config.get("sampler")

        # (2) Arrange - custom_query_manager returns no data (empty scenario)
        update_team_sampler(team_config, custom_query_manager, base_custom_settings)

        # (4) Assert - verify sampler exists but handles empty data
        # (range sampler created)
        self._assert_sampler_created(team_config, initial_sampler)

        # With empty data, should fall back to range sampler with min/max
        sampler_func = team_config.throughput_config["sampler"]
        test_values = [sampler_func() for _ in range(10)]
        assert all(0 <= val <= 10 for val in test_values), (
            "Sampler should return values in range [0, 10]"
        )

    def test_update_team_sampler_single_value(
        self, query_manager_for_sampler, base_custom_settings
    ):
        """Test updating team sampler with single throughput value."""

        # (1) Set clear initial state - create mock object with single value range
        mock_team = Mock()
        mock_team.throughput_config = {
            "throughput_samples": [],  # Empty list
            "min_throughput": 3,
            "max_throughput": 3,  # Single value
        }

        team_config = mock_team
        initial_sampler = team_config.throughput_config.get("sampler")

        # (2) Arrange - query_manager_for_sampler has data but we test
        # single-value range
        update_team_sampler(
            team_config, query_manager_for_sampler, base_custom_settings
        )

        # (4) Assert - verify sampler works with single-value range (min == max)
        self._assert_sampler_created(team_config, initial_sampler)

        # With single value (min == max), sampler should always return that value
        sampler_func = team_config.throughput_config["sampler"]
        test_values = [sampler_func() for _ in range(10)]
        assert all(val == 3 for val in test_values), (
            "Sampler should return value 3 for single-value range"
        )


class TestTeamConfiguration(ProgressReportTestBase):
    """Test cases for team configuration handling."""

    def test_team_config_validation_valid_config(
        self, base_custom_fields, base_custom_settings
    ):
        """Test validation passes for valid team configuration."""
        # Create calculator instance
        jira = FauxJIRA(fields=base_custom_fields, issues=[])
        query_manager = QueryManager(jira, base_custom_settings)

        # Mock settings required for calculator
        settings = {
            "progress_report": {
                "enabled": True,
                "templates": {
                    "epic": "query",
                    "story": "query",
                },
                "teams": to_progress_report_teams_list(
                    [
                        {
                            "name": "Team A",
                            "wip": 2,
                            "min_throughput": 1,
                            "max_throughput": 5,
                        }
                    ]
                ),
            },
        }

        calculator = ProgressReportCalculator(query_manager, settings, results={})

        # Call validation - should not return an error message
        team_config = settings["progress_report"]["teams"][0]
        error = calculator.validate_single_team(team_config)
        assert error is None, "Valid team config should not produce validation errors"

    def test_team_config_validation_missing_name(
        self, base_custom_fields, base_custom_settings
    ):
        """Test validation fails for team without name."""
        jira = FauxJIRA(fields=base_custom_fields, issues=[])
        query_manager = QueryManager(jira, base_custom_settings)

        settings = {
            "progress_report": {
                "enabled": True,
                "templates": {
                    "epic": "query",
                    "story": "query",
                },
                "teams": to_progress_report_teams_list(
                    [
                        {
                            "wip": 2,
                            "min_throughput": 1,
                            "max_throughput": 5,
                        }
                    ]
                ),
            },
        }

        calculator = ProgressReportCalculator(query_manager, settings, results={})

        team_config = settings["progress_report"]["teams"][0]
        error = calculator.validate_single_team(team_config)
        assert error is not None, "Team without name should produce validation error"
        assert "name" in error.lower()

    def test_team_config_validation_invalid_wip(
        self, base_custom_fields, base_custom_settings
    ):
        """Test validation fails for team with invalid WIP."""
        jira = FauxJIRA(fields=base_custom_fields, issues=[])
        query_manager = QueryManager(jira, base_custom_settings)

        settings = {
            "progress_report": {
                "enabled": True,
                "templates": {
                    "epic": "query",
                    "story": "query",
                },
                "teams": to_progress_report_teams_list(
                    [
                        {
                            "name": "Team A",
                            "wip": 0,  # Invalid WIP
                        }
                    ]
                ),
            },
        }

        calculator = ProgressReportCalculator(query_manager, settings, results={})

        team_config = settings["progress_report"]["teams"][0]
        error = calculator.validate_single_team(team_config)
        assert error is not None, (
            "Team with invalid WIP should produce validation error"
        )
        assert "wip" in error.lower() and ">= 1" in error

    def test_team_config_validation_missing_wip(
        self, base_custom_fields, base_custom_settings
    ):
        """Test validation fails for team with missing WIP."""

        jira = FauxJIRA(fields=base_custom_fields, issues=[])
        query_manager = QueryManager(jira, base_custom_settings)

        settings = {
            "progress_report": {
                "enabled": True,
                "templates": {
                    "epic": "query",
                    "story": "query",
                },
                "teams": [
                    {
                        "name": "Team A",
                        # No wip specified
                    }
                ],
            },
        }

        calculator = ProgressReportCalculator(query_manager, settings, results={})

        # Test with raw dict without processing through to_progress_report_teams_list
        raw_team_config = {"name": "Team A"}
        try:
            _ = calculator.validate_single_team(raw_team_config)
            # Should raise ConfigError when 'wip' key is missing
            assert False, "Expected ConfigError when 'wip' key is missing"
        except ConfigError as e:
            # This is expected - validation should raise ConfigError
            # for missing required keys
            assert "missing required field" in str(e).lower()
            assert "wip" in str(e).lower()
            assert "team a" in str(e).lower() or "team: Team A" in str(e)

    def test_team_config_validation_incomplete_throughput_range(
        self, base_custom_fields, base_custom_settings
    ):
        """Test validation fails when only one throughput value is set."""
        jira = FauxJIRA(fields=base_custom_fields, issues=[])
        query_manager = QueryManager(jira, base_custom_settings)

        # Test with only min_throughput set
        settings = {
            "progress_report": {
                "enabled": True,
                "templates": {
                    "epic": "query",
                    "story": "query",
                },
                "teams": to_progress_report_teams_list(
                    [
                        {
                            "name": "Team A",
                            "wip": 2,
                            "min_throughput": 1,
                            # Missing max_throughput
                        }
                    ]
                ),
            },
        }

        calculator = ProgressReportCalculator(query_manager, settings, results={})

        # Test with raw dict to avoid defaults
        raw_team_config = {
            "name": "Team A",
            "wip": 2,
            "min_throughput": 1,
            "max_throughput": None,  # Missing
            "throughput_samples": None,
            "throughput_samples_window": None,
        }
        error = calculator.validate_single_team(raw_team_config)
        assert error is not None, (
            "Incomplete throughput range should produce validation error"
        )
        assert "min throughput" in error.lower() and "max throughput" in error.lower()

    def test_team_config_validation_inverted_throughput_range(
        self, base_custom_fields, base_custom_settings
    ):
        """Test validation fails when min > max throughput."""
        jira = FauxJIRA(fields=base_custom_fields, issues=[])
        query_manager = QueryManager(jira, base_custom_settings)

        settings = {
            "progress_report": {
                "enabled": True,
                "templates": {
                    "epic": "query",
                    "story": "query",
                },
                "teams": to_progress_report_teams_list(
                    [
                        {
                            "name": "Team A",
                            "wip": 2,
                            "min_throughput": 5,
                            "max_throughput": 1,  # min > max
                        }
                    ]
                ),
            },
        }

        calculator = ProgressReportCalculator(query_manager, settings, results={})

        # Test with raw dict to ensure proper values
        raw_team_config = {
            "name": "Team A",
            "wip": 2,
            "min_throughput": 5,
            "max_throughput": 1,  # min > max
            "throughput_samples": None,
            "throughput_samples_window": None,
        }
        error = calculator.validate_single_team(raw_team_config)
        assert error is not None, (
            "Inverted throughput range should produce validation error"
        )
        assert "min throughput" in error.lower() and "max throughput" in error.lower()

    def test_team_config_validation_conflicting_sampler_config(
        self, base_custom_fields, base_custom_settings
    ):
        """Test validation fails when both throughput range and samples are set."""
        jira = FauxJIRA(fields=base_custom_fields, issues=[])
        query_manager = QueryManager(jira, base_custom_settings)

        settings = {
            "progress_report": {
                "enabled": True,
                "templates": {
                    "epic": "query",
                    "story": "query",
                },
                "teams": to_progress_report_teams_list(
                    [
                        {
                            "name": "Team A",
                            "wip": 2,
                            "min_throughput": 1,
                            "max_throughput": 5,
                            "throughput_samples": "query",  # Cannot set both
                        }
                    ]
                ),
            },
        }

        calculator = ProgressReportCalculator(query_manager, settings, results={})

        # Test with raw dict to avoid defaults
        raw_team_config = {
            "name": "Team A",
            "wip": 2,
            "min_throughput": 1,
            "max_throughput": 5,
            "throughput_samples": "query",  # Cannot set both
            "throughput_samples_window": None,
        }
        error = calculator.validate_single_team(raw_team_config)
        assert error is not None, (
            "Conflicting sampler config should produce validation error"
        )
        assert "throughput samples" in error.lower()

    def test_team_config_validation_missing_samples_window(
        self, base_custom_fields, base_custom_settings
    ):
        """Test validation fails when throughput_samples is set without window."""
        jira = FauxJIRA(fields=base_custom_fields, issues=[])
        query_manager = QueryManager(jira, base_custom_settings)

        settings = {
            "progress_report": {
                "enabled": True,
                "templates": {
                    "epic": "query",
                    "story": "query",
                },
                "teams": to_progress_report_teams_list(
                    [
                        {
                            "name": "Team A",
                            "wip": 2,
                            "throughput_samples": "query",
                            # Missing throughput_samples_window
                        }
                    ]
                ),
            },
        }

        calculator = ProgressReportCalculator(query_manager, settings, results={})

        # Test with raw dict to avoid defaults
        raw_team_config = {
            "name": "Team A",
            "wip": 2,
            "min_throughput": None,
            "max_throughput": None,
            "throughput_samples": "query",
            "throughput_samples_window": None,  # Missing
        }
        error = calculator.validate_single_team(raw_team_config)
        assert error is not None, (
            "Missing samples window should produce validation error"
        )
        assert "throughput samples window" in error.lower()

    def test_team_config_invalid_integer_values(self):
        """Test validation of invalid integer values in team configuration."""

        # Test force_int raises exception for non-integer string
        with pytest.raises(ConfigError) as exc_info:
            force_int("wip", "not-an-integer")

        assert "wip" in str(exc_info.value).lower()

        # Test force_int raises exception for invalid int conversion
        with pytest.raises(ConfigError) as exc_info:
            force_int("throughput_samples_window", "12.5")  # Float string

        assert "throughput samples window" in str(exc_info.value).lower()

    def test_team_config_edge_case_wip_one(
        self, base_custom_fields, base_custom_settings
    ):
        """Test validation passes for WIP value of 1 (minimum valid)."""
        jira = FauxJIRA(fields=base_custom_fields, issues=[])
        query_manager = QueryManager(jira, base_custom_settings)

        settings = {
            "progress_report": {
                "enabled": True,
                "templates": {
                    "epic": "query",
                    "story": "query",
                },
                "teams": to_progress_report_teams_list(
                    [
                        {
                            "name": "Team A",
                            "wip": 1,  # Minimum valid WIP
                        }
                    ]
                ),
            },
        }

        calculator = ProgressReportCalculator(query_manager, settings, results={})

        team_config = settings["progress_report"]["teams"][0]
        error = calculator.validate_single_team(team_config)
        assert error is None, "WIP of 1 should be valid"

    def test_team_config_edge_case_large_values(
        self, base_custom_fields, base_custom_settings
    ):
        """Test validation passes for large but valid throughput values."""
        jira = FauxJIRA(fields=base_custom_fields, issues=[])
        query_manager = QueryManager(jira, base_custom_settings)

        settings = {
            "progress_report": {
                "enabled": True,
                "templates": {
                    "epic": "query",
                    "story": "query",
                },
                "teams": to_progress_report_teams_list(
                    [
                        {
                            "name": "Team A",
                            "wip": 2,
                            "min_throughput": 100,
                            "max_throughput": 1000,
                        }
                    ]
                ),
            },
        }

        calculator = ProgressReportCalculator(query_manager, settings, results={})

        team_config = settings["progress_report"]["teams"][0]
        error = calculator.validate_single_team(team_config)
        assert error is None, "Large throughput values should be valid"

    def test_team_config_edge_case_single_throughput_value(
        self, base_custom_fields, base_custom_settings
    ):
        """Test validation passes when min and max are the same."""
        jira = FauxJIRA(fields=base_custom_fields, issues=[])
        query_manager = QueryManager(jira, base_custom_settings)

        settings = {
            "progress_report": {
                "enabled": True,
                "templates": {
                    "epic": "query",
                    "story": "query",
                },
                "teams": to_progress_report_teams_list(
                    [
                        {
                            "name": "Team A",
                            "wip": 2,
                            "min_throughput": 5,
                            "max_throughput": 5,  # Same as min
                        }
                    ]
                ),
            },
        }

        calculator = ProgressReportCalculator(query_manager, settings, results={})

        team_config = settings["progress_report"]["teams"][0]
        error = calculator.validate_single_team(team_config)
        assert error is None, "Equal min and max throughput should be valid"
