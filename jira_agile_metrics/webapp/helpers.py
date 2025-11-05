"""Helper functions for the web application."""

import contextlib
import logging
import os
import os.path
import shutil
import tempfile
import zipfile

import numpy as np
from jira.exceptions import JIRAError

from ..calculator import run_calculators
from ..common_constants import (
    BOTTLENECK_CHART_SETTINGS,
    CHART_FILENAME_KEYS,
    DATA_FILENAME_KEYS,
)
from ..config import ConfigError
from ..jira_client import create_jira_client


def plot_forecast_fan(p, forecast_data):
    """Plot forecast fan bands on a Bokeh figure.

    Args:
        p: Bokeh figure to plot on
        forecast_data: DataFrame with forecast trial data (columns are trials)
    """
    # Early return if forecast_data is empty or has no rows
    if forecast_data.empty or len(forecast_data.index) == 0:
        return

    # Drop rows and columns that are all-NaN
    forecast_data = forecast_data.dropna(how="all", axis=0)
    forecast_data = forecast_data.dropna(how="all", axis=1)

    # Check again after dropping all-NaN rows/columns
    if (
        forecast_data.empty
        or len(forecast_data.index) == 0
        or len(forecast_data.columns) == 0
    ):
        return

    trials_array = forecast_data.values.T  # Transpose to get trials as rows

    # Ensure trials_array has non-zero size and contains at least one
    # finite numeric value
    if trials_array.size == 0 or np.all(~np.isfinite(trials_array)):
        return

    # Convert forecast_dates to list after validation
    forecast_dates = forecast_data.index.tolist()
    percentiles = [10, 25, 50, 75, 90]
    fan_data = np.percentile(trials_array, percentiles, axis=0)
    p_indices = {pct: percentiles.index(pct) for pct in percentiles}

    # Outer band (10-90 percentile)
    p.varea(
        x=forecast_dates,
        y1=fan_data[p_indices[10]],
        y2=fan_data[p_indices[90]],
        alpha=0.2,
        color="green",
        legend_label="Done forecast (10-90%)",
    )
    # Inner band (25-75 percentile)
    p.varea(
        x=forecast_dates,
        y1=fan_data[p_indices[25]],
        y2=fan_data[p_indices[75]],
        alpha=0.3,
        color="green",
        legend_label="Done forecast (25-75%)",
    )
    # Median line
    p.line(
        forecast_dates,
        fan_data[p_indices[50]],
        legend_label="Done forecast (median)",
        line_width=2,
        line_dash="dashed",
        color="darkgreen",
        alpha=0.7,
    )


@contextlib.contextmanager
def capture_log(buffer, level, formatter=None):
    """Temporarily write log output to the StringIO `buffer` with log level
    threshold `level`, before returning logging to normal.
    """
    root_logger = logging.getLogger()

    old_level = root_logger.getEffectiveLevel()
    root_logger.setLevel(level)

    handler = logging.StreamHandler(buffer)

    if formatter:
        formatter = logging.Formatter(formatter)
        handler.setFormatter(formatter)

    root_logger.addHandler(handler)

    yield

    root_logger.removeHandler(handler)
    root_logger.setLevel(old_level)

    handler.flush()
    buffer.flush()


def override_options(options, form):
    """Override options from the configuration files with form data where
    applicable.
    """
    for key in options.keys():
        value = form.get(key)
        if value is not None and value != "":
            options[key] = value


def get_jira_client(connection):
    """Create a JIRA client with the given connection options"""
    try:
        return create_jira_client(connection)
    except JIRAError as e:
        if e.status_code == 401:
            raise ConfigError(
                (
                    "JIRA authentication failed. "
                    "Check URL and credentials, "
                    "and ensure the account is not locked."
                )
            ) from e
        raise


def _is_path_key(key):
    """Check if a key should be treated as a file path.

    Args:
        key: Setting key to check

    Returns:
        True if the key represents a file path, False otherwise
    """
    # Explicit allowlist of keys that should be treated as file paths
    # These match the keys used in config/loader.py for filename parsing
    explicit_path_keys = (
        set(CHART_FILENAME_KEYS)
        | set(DATA_FILENAME_KEYS)
        | set(BOTTLENECK_CHART_SETTINGS)
    )

    # Check explicit allowlist
    if key in explicit_path_keys:
        return True

    # Check for suffix-based detection
    path_suffixes = ("_path", "_file", "_dir", "_chart", "_data")
    if any(key.endswith(suffix) for suffix in path_suffixes):
        return True

    return False


def _transform_list_item(item, base_path, is_path, validate_paths, key):
    """Transform a single list item for path conversion.

    Args:
        item: List item to transform (string, dict, or other type)
        base_path: Base directory to resolve relative paths against
        is_path: Whether the item should be treated as a file path
        validate_paths: If True, validate that converted paths exist
        key: Parent key for nested structures (used for propagation)

    Returns:
        Transformed item: os.path.join(base_path, item) for relative strings when
        is_path is True, _make_paths_absolute(...) for dicts, or item otherwise
    """
    if isinstance(item, str) and not os.path.isabs(item) and is_path:
        return os.path.join(base_path, item)
    if isinstance(item, dict):
        return _make_paths_absolute(item, base_path, validate_paths, key)
    return item


def _make_paths_absolute(settings, base_path, validate_paths=False, parent_key=None):
    """Recursively convert relative file paths in settings to absolute paths.

    Uses key-based detection instead of fragile filename extension checks.
    Only converts paths for keys in the allowlist or keys with recognized
    suffixes (_path, _file, _dir, _chart, _data).

    Args:
        settings: Dictionary of settings that may contain file paths
        base_path: Base directory to resolve relative paths against
        validate_paths: If True, validate that converted paths exist
        parent_key: Parent key for nested structures (used for propagation)

    Returns:
        New dictionary with all relative file paths converted to absolute paths
    """
    if not isinstance(settings, dict):
        return settings

    result = {}
    for key, value in settings.items():
        # Determine if this key should be treated as a file path
        # Check current key first, then parent key (for nested structures)
        is_path = _is_path_key(key) or (
            parent_key is not None and _is_path_key(parent_key)
        )

        if isinstance(value, str):
            # Only convert if it's a relative path and the key indicates it's a path
            if is_path and not os.path.isabs(value):
                converted_path = os.path.join(base_path, value)
                # Optionally validate the path exists
                if validate_paths and not os.path.exists(converted_path):
                    # For new files that will be created, we don't require existence
                    # Only validate if it's a directory or file that should exist
                    pass  # Could add logging here if needed
                result[key] = converted_path
            else:
                result[key] = value
        elif isinstance(value, list):
            # Handle lists that may contain file paths
            # For list elements, use the list's key to determine if elements are paths
            result[key] = [
                _transform_list_item(item, base_path, is_path, validate_paths, key)
                for item in value
            ]
        elif isinstance(value, dict):
            # Recursively process nested dictionaries
            # Check current key first, but also propagate it for nested structures
            result[key] = _make_paths_absolute(
                value, base_path, validate_paths, parent_key=key
            )
        else:
            result[key] = value

    return result


def get_archive(calculators, query_manager, settings):
    """Run all calculators and write outputs to a temporary directory.
    Create a zip archive of all the files written, and return it as a bytes
    array. Remove the temporary directory on completion.

    This function uses absolute paths instead of os.chdir() to avoid
    thread-safety issues and enable concurrent archive requests.
    """
    temp_path = tempfile.mkdtemp()

    try:
        # Convert all relative paths in settings to absolute paths
        # within the temp directory
        abs_settings = _make_paths_absolute(settings, temp_path)

        # Run calculators with absolute paths (no os.chdir() needed)
        run_calculators(calculators, query_manager, abs_settings)

        # Create compressed zip archive using absolute paths
        zip_path = os.path.join(temp_path, "metrics.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _dirs, files in os.walk(temp_path):
                for file_name in files:
                    if file_name != "metrics.zip":
                        file_path = os.path.join(root, file_name)
                        # Use relative path within temp_path for ZIP archive structure
                        arcname = os.path.join(
                            "metrics", os.path.relpath(file_path, temp_path)
                        )
                        z.write(file_path, arcname)

        # Read the ZIP file into memory
        with open(zip_path, "rb") as metrics_zip:
            zip_data = metrics_zip.read()

        return zip_data

    finally:
        # Clean up temporary directory - always remove, even if an exception occurred
        shutil.rmtree(temp_path)
