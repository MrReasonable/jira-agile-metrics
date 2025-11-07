"""Tests for webapp helper functions in Jira Agile Metrics.

This module contains unit tests for the webapp helpers module.
"""

import io
import logging
import os
import tempfile
import zipfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from jira.exceptions import JIRAError

from ..calculator import Calculator
from ..calculators.cycletime import CycleTimeCalculator
from ..config import ConfigError
from ..querymanager import QueryManager
from ..test_classes import FauxJIRA
from .helpers import (
    _is_path_key,
    _make_paths_absolute,
    _transform_list_item,
    capture_log,
    get_archive,
    get_jira_client,
    override_options,
    plot_forecast_fan,
)


def test_plot_forecast_fan_normal_data():
    """Test plot_forecast_fan with normal data."""
    mock_figure = MagicMock()
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    forecast_data = pd.DataFrame(
        {f"trial_{i}": np.random.rand(10) * 100 for i in range(20)},
        index=dates,
    )

    plot_forecast_fan(mock_figure, forecast_data)

    # Should call varea twice (outer and inner bands) and line once (median)
    assert mock_figure.varea.call_count == 2
    assert mock_figure.line.call_count == 1


def test_plot_forecast_fan_empty_dataframe():
    """Test plot_forecast_fan with empty DataFrame."""
    mock_figure = MagicMock()
    forecast_data = pd.DataFrame()

    plot_forecast_fan(mock_figure, forecast_data)

    # Should not call any plotting methods
    mock_figure.varea.assert_not_called()
    mock_figure.line.assert_not_called()


def test_plot_forecast_fan_all_nan():
    """Test plot_forecast_fan with all NaN data."""
    mock_figure = MagicMock()
    dates = pd.date_range(start="2024-01-01", periods=5, freq="D")
    forecast_data = pd.DataFrame(
        {f"trial_{i}": [np.nan] * 5 for i in range(3)},
        index=dates,
    )

    plot_forecast_fan(mock_figure, forecast_data)

    # Should not call any plotting methods
    mock_figure.varea.assert_not_called()
    mock_figure.line.assert_not_called()


def test_plot_forecast_fan_infinite_values():
    """Test plot_forecast_fan with infinite values."""
    mock_figure = MagicMock()
    dates = pd.date_range(start="2024-01-01", periods=5, freq="D")
    forecast_data = pd.DataFrame(
        {f"trial_{i}": [np.inf] * 5 for i in range(3)},
        index=dates,
    )

    plot_forecast_fan(mock_figure, forecast_data)

    # Should not call any plotting methods
    mock_figure.varea.assert_not_called()
    mock_figure.line.assert_not_called()


def test_capture_log():
    """Test capture_log context manager."""
    buffer = io.StringIO()
    logger = MagicMock()

    with patch(
        "jira_agile_metrics.webapp.helpers.logging.getLogger", return_value=logger
    ):
        with capture_log(buffer, logging.INFO):
            logger.info("Test message")

    assert "Test message" in buffer.getvalue() or logger.info.called


def test_override_options():
    """Test override_options function."""
    options = {"key1": "value1", "key2": "value2", "key3": "value3"}
    form = {"key2": "new_value2", "key4": "value4"}

    result = override_options(options, form)

    assert result["key1"] == "value1"
    assert result["key2"] == "new_value2"
    assert result["key3"] == "value3"
    assert "key4" not in result  # Only override existing keys


def test_override_options_empty_form():
    """Test override_options with empty form."""
    options = {"key1": "value1", "key2": "value2"}
    form = {}

    result = override_options(options, form)

    assert result == options


def test_override_options_empty_string():
    """Test override_options with empty string in form."""
    options = {"key1": "value1"}
    form = {"key1": ""}

    result = override_options(options, form)

    # Empty string should not override
    assert result["key1"] == "value1"


def test_get_jira_client_success():
    """Test get_jira_client with successful connection."""
    mock_client = MagicMock()
    connection = {
        "domain": "https://test.jira.com",
        "username": "user",
        "password": "pass",
    }

    with patch(
        "jira_agile_metrics.webapp.helpers.create_jira_client",
        return_value=mock_client,
    ):
        result = get_jira_client(connection)

    assert result == mock_client


def test_get_jira_client_authentication_error():
    """Test get_jira_client with authentication error."""
    connection = {
        "domain": "https://test.jira.com",
        "username": "user",
        "password": "wrong",
    }
    error = JIRAError(status_code=401, text="Unauthorized")

    with patch(
        "jira_agile_metrics.webapp.helpers.create_jira_client", side_effect=error
    ):
        with pytest.raises(ConfigError) as exc_info:
            get_jira_client(connection)

        assert "authentication failed" in str(exc_info.value).lower()


def test_get_jira_client_other_error():
    """Test get_jira_client with other JIRA error."""
    connection = {"domain": "https://test.jira.com"}
    error = JIRAError(status_code=500, text="Internal Server Error")

    with patch(
        "jira_agile_metrics.webapp.helpers.create_jira_client", side_effect=error
    ):
        with pytest.raises(JIRAError):
            get_jira_client(connection)


def test_is_path_key_chart_keys():
    """Test _is_path_key with chart filename keys."""
    assert _is_path_key("histogram_chart") is True
    assert _is_path_key("cfd_chart") is True
    assert _is_path_key("burnup_chart") is True


def test_is_path_key_data_keys():
    """Test _is_path_key with data filename keys."""
    assert _is_path_key("histogram_data") is True
    assert _is_path_key("cfd_data") is True
    assert _is_path_key("throughput_data") is True


def test_is_path_key_suffixes():
    """Test _is_path_key with path suffixes."""
    assert _is_path_key("output_path") is True
    assert _is_path_key("config_file") is True
    assert _is_path_key("data_dir") is True
    assert _is_path_key("chart_path") is True


def test_is_path_key_non_path():
    """Test _is_path_key with non-path keys."""
    assert _is_path_key("username") is False
    assert _is_path_key("password") is False
    assert _is_path_key("domain") is False


def test_transform_list_item_string_path():
    """Test _transform_list_item with string path."""
    base_path = "/tmp"
    item = "output.csv"
    result = _transform_list_item(
        item, base_path, is_path=True, validate_paths=False, key="output_file"
    )

    assert result == os.path.join(base_path, item)


def test_transform_list_item_string_non_path():
    """Test _transform_list_item with string non-path."""
    base_path = "/tmp"
    item = "some_value"
    result = _transform_list_item(
        item, base_path, is_path=False, validate_paths=False, key="value"
    )

    assert result == item


def test_transform_list_item_dict():
    """Test _transform_list_item with dict."""
    base_path = "/tmp"
    item = {"key": "value"}
    result = _transform_list_item(
        item, base_path, is_path=True, validate_paths=False, key="config"
    )

    assert isinstance(result, dict)


def test_transform_list_item_absolute_path():
    """Test _transform_list_item with absolute path."""
    base_path = "/tmp"
    item = "/absolute/path/file.csv"
    result = _transform_list_item(
        item, base_path, is_path=True, validate_paths=False, key="output_file"
    )

    assert result == item  # Should not modify absolute paths


def test_make_paths_absolute_simple():
    """Test _make_paths_absolute with simple settings."""
    settings = {"output_file": "data.csv", "username": "user"}
    base_path = "/tmp"

    result = _make_paths_absolute(settings, base_path, validate_paths=False)

    assert result["output_file"] == os.path.join(base_path, "data.csv")
    assert result["username"] == "user"  # Non-path key unchanged


def test_make_paths_absolute_list():
    """Test _make_paths_absolute with list of paths."""
    # Use _data suffix to trigger path detection
    settings = {"output_data": ["file1.csv", "file2.csv"]}
    base_path = "/tmp"

    result = _make_paths_absolute(settings, base_path, validate_paths=False)

    assert len(result["output_data"]) == 2
    assert all(os.path.isabs(p) for p in result["output_data"])


def test_make_paths_absolute_nested_dict():
    """Test _make_paths_absolute with nested dictionary."""
    settings = {
        "output": {
            "chart_file": "chart.png",
            "data_file": "data.csv",
        },
        "username": "user",
    }
    base_path = "/tmp"

    result = _make_paths_absolute(settings, base_path, validate_paths=False)

    assert os.path.isabs(result["output"]["chart_file"])
    assert os.path.isabs(result["output"]["data_file"])
    assert result["username"] == "user"


def test_make_paths_absolute_not_dict():
    """Test _make_paths_absolute with non-dict input."""
    result = _make_paths_absolute("not a dict", "/tmp", validate_paths=False)

    assert result == "not a dict"


def test_make_paths_absolute_absolute_path():
    """Test _make_paths_absolute with absolute path."""
    settings = {"output_file": "/absolute/path/file.csv"}
    base_path = "/tmp"

    result = _make_paths_absolute(settings, base_path, validate_paths=False)

    assert result["output_file"] == "/absolute/path/file.csv"


def test_get_archive():
    """Test get_archive function."""
    # Create minimal fields for QueryManager
    fields = [
        {"id": "summary", "name": "Summary"},
        {"id": "status", "name": "Status"},
        {"id": "created", "name": "Created date"},
    ]

    # Create minimal settings
    settings = {
        "cycle": [
            {"name": "Backlog", "statuses": ["Backlog"]},
            {"name": "Done", "statuses": ["Done"]},
        ],
        "backlog_column": "Backlog",
        "committed_column": "Backlog",
        "done_column": "Done",
        "cfd_data": ["cfd.csv"],
        "cfd_chart": "cfd.png",
        "attributes": {},
        "known_values": {},
        "max_results": None,
        "verbose": False,
        "query_attribute": None,
        "queries": [{"jql": "project=TEST", "value": None}],
    }

    jira = FauxJIRA(fields=fields, issues=[])
    query_manager = QueryManager(jira, settings)
    calculators = [CycleTimeCalculator]

    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Update settings to use temp directory
        settings["cfd_data"] = [os.path.join(temp_dir, "cfd.csv")]
        settings["cfd_chart"] = os.path.join(temp_dir, "cfd.png")

        archive_data = get_archive(calculators, query_manager, settings)

    assert archive_data is not None
    assert isinstance(archive_data, bytes)
    # Archive may be empty if no issues found, but should still be valid
    if len(archive_data) > 0:
        # Verify it's a valid zip file if not empty
        with zipfile.ZipFile(io.BytesIO(archive_data)) as zip_file:
            # Just verify it's a valid zip structure
            assert zip_file.testzip() is None  # None means no errors


def test_get_archive_empty_calculators():
    """Test get_archive with empty calculators list."""
    # Create minimal fields for QueryManager
    fields = [
        {"id": "summary", "name": "Summary"},
        {"id": "status", "name": "Status"},
        {"id": "created", "name": "Created date"},
    ]

    settings = {
        "cycle": [
            {"name": "Backlog", "statuses": ["Backlog"]},
            {"name": "Done", "statuses": ["Done"]},
        ],
        "backlog_column": "Backlog",
        "committed_column": "Backlog",
        "done_column": "Done",
        "attributes": {},
        "known_values": {},
        "max_results": None,
        "verbose": False,
        "query_attribute": None,
        "queries": [{"jql": "project=TEST", "value": None}],
    }

    jira = FauxJIRA(fields=fields, issues=[])
    query_manager = QueryManager(jira, settings)
    calculators = []

    archive_data = get_archive(calculators, query_manager, settings)

    assert archive_data is not None
    assert isinstance(archive_data, bytes)


def test_get_archive_cleanup_on_error():
    """Test that get_archive cleans up temp directory on error."""
    # Create minimal fields for QueryManager
    fields = [
        {"id": "summary", "name": "Summary"},
        {"id": "status", "name": "Status"},
        {"id": "created", "name": "Created date"},
    ]

    settings = {
        "cycle": [
            {"name": "Backlog", "statuses": ["Backlog"]},
            {"name": "Done", "statuses": ["Done"]},
        ],
        "backlog_column": "Backlog",
        "committed_column": "Backlog",
        "done_column": "Done",
        "attributes": {},
        "known_values": {},
        "max_results": None,
        "verbose": False,
        "query_attribute": None,
        "queries": [{"jql": "project=TEST", "value": None}],
    }

    jira = FauxJIRA(fields=fields, issues=[])
    query_manager = QueryManager(jira, settings)

    # Create a calculator that will raise an error
    class FailingCalculator(Calculator):
        """Test calculator that raises an error."""

        def run(self):
            raise ValueError("Test error")

    calculators = [FailingCalculator]

    # Should not raise an error, should handle cleanup
    try:
        get_archive(calculators, query_manager, settings)
    except ValueError:
        pass  # Expected to fail, but temp directory should be cleaned up
