"""Pytest fixtures for functional tests."""

from pathlib import Path

import pandas as pd
import pytest

from jira_agile_metrics.calculator import run_calculators
from jira_agile_metrics.calculators.burnup import BurnupCalculator
from jira_agile_metrics.calculators.cfd import CFDCalculator
from jira_agile_metrics.calculators.cycletime import CycleTimeCalculator
from jira_agile_metrics.calculators.forecast import BurnupForecastCalculator
from jira_agile_metrics.querymanager import QueryManager
from jira_agile_metrics.test_file_jira_client import FileJiraClient
from jira_agile_metrics.tests.e2e.e2e_config import _get_standard_cycle_config


def pytest_collection_modifyitems(_config, items):
    """Automatically mark all tests in the functional directory.

    This hook runs during test collection and adds the 'functional' marker to all
    test items found in this directory. This allows us to consolidate marker
    application in one place rather than requiring pytestmark in each test file.

    Args:
        _config: Pytest config object (unused, required by hook signature)
        items: List of test items collected by pytest
    """
    functional_dir = Path(__file__).parent.resolve()
    for item in items:
        # Get the test file path from the item's location
        # item.location is a tuple: (filepath, lineno, testname)
        test_file_path = Path(item.location[0]).resolve()
        # Check if the test file is in the functional directory
        if (
            functional_dir in test_file_path.parents
            or test_file_path.parent == functional_dir
        ):
            item.add_marker(pytest.mark.functional)


def fixtures_path(*parts):
    """Return path to test fixtures directory."""
    return str(Path(__file__).resolve().parent.parent.joinpath("fixtures", *parts))


@pytest.fixture()
def jira_client():
    """Create a FileJiraClient fixture using test data."""
    return FileJiraClient(fixtures_path("jira"))


def _create_query_manager_settings():
    """Create minimal settings for QueryManager.

    Returns:
        Dictionary with minimal QueryManager settings.
    """
    return {
        "attributes": {},
        "known_values": {},
        "max_results": False,
    }


@pytest.fixture()
def query_manager(request):
    """Create a QueryManager fixture with minimal settings.

    Args:
        request: Pytest request object to access fixtures.
    """
    client = request.getfixturevalue("jira_client")
    return QueryManager(client, settings=_create_query_manager_settings())


def _create_base_settings(cycle_time_data=None):
    """Create base settings dictionary for functional tests.

    Args:
        cycle_time_data: Optional list of cycle time data output paths.
            If None, defaults to empty list.

    Returns:
        Dictionary with base settings.
    """
    if cycle_time_data is None:
        cycle_time_data = []
    return {
        "cycle": _get_standard_cycle_config(),
        "committed_column": "Committed",
        "done_column": "Done",
        "attributes": {},
        "queries": [{"jql": "project=TEST"}],
        "query_attribute": None,
        "cycle_time_data": cycle_time_data,
    }


@pytest.fixture()
def simple_cycle_settings(tmp_path):
    """Create simple cycle time settings for functional tests."""
    output_csv = tmp_path / "cycletime.csv"
    settings = _create_base_settings(cycle_time_data=[str(output_csv)])
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


def get_default_forecast_settings():
    """Get default forecast settings for tests.

    Returns a dictionary with common forecast test settings.
    """
    return {
        "burnup_forecast_chart_throughput_window": 30,
        "burnup_forecast_chart_throughput_window_end": None,
        "burnup_forecast_chart_target": None,
        "burnup_forecast_chart_deadline": None,
        "burnup_forecast_chart_deadline_confidence": 0.85,
    }


def run_forecast_calculators(query_mgr, settings):
    """Run forecast-related calculators and return results.

    Args:
        query_mgr: QueryManager instance
        settings: Settings dictionary

    Returns:
        Dictionary of calculator results
    """
    return run_calculators(
        [
            CycleTimeCalculator,
            CFDCalculator,
            BurnupCalculator,
            BurnupForecastCalculator,
        ],
        query_mgr,
        settings,
    )


def validate_forecast_result_structure(forecast_result):
    """Validate basic structure of forecast result DataFrame.

    Args:
        forecast_result: Forecast result DataFrame

    Returns:
        None (raises AssertionError if validation fails)
    """
    assert isinstance(forecast_result, pd.DataFrame), (
        "Forecast result should be a DataFrame"
    )
    assert len(forecast_result.columns) > 0, "Forecast should have trial columns"
    assert isinstance(forecast_result.index, pd.DatetimeIndex), (
        "Forecast index should be datetime"
    )


def validate_forecast_trial_values(forecast_result, num_trials=3):
    """Validate forecast trial values are reasonable.

    Args:
        forecast_result: Forecast result DataFrame
        num_trials: Number of trial columns to check

    Returns:
        None (raises AssertionError if validation fails)
    """
    sample_trials = list(forecast_result.columns)[:num_trials]
    for trial_col in sample_trials:
        trial_values = forecast_result[trial_col].dropna()
        if len(trial_values) > 1:
            assert (trial_values >= 0).all(), (
                f"Trial {trial_col} should have non-negative values"
            )


# end of file
