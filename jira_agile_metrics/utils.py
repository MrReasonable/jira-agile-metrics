"""Utility functions for Jira Agile Metrics.

This module provides common utility functions used across the metrics calculations
including data processing, chart styling, and date handling.
"""

import datetime
import logging
import os.path
import random
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .chart_styling_utils import apply_common_chart_styling

T = TypeVar("T")
logger = logging.getLogger(__name__)


# Status type constants for issue tracking
STATUS_BACKLOG = "backlog"
STATUS_ACCEPTED = "accepted"
STATUS_COMPLETE = "complete"


def retry_with_backoff(
    max_attempts: int = 5,
    base_delay: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    return_on_failure: Optional[Callable] = None,
    should_retry: Optional[Callable[[Exception], bool]] = None,
):
    """Decorator for API calls with retry logic and exponential backoff.

    Provides exponential backoff with jitter and configurable retry behavior.

    Args:
        max_attempts: Maximum number of retry attempts (default: 5)
        base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        exceptions: Tuple of exception types to catch and retry (default: Exception)
        return_on_failure: Callable that returns value on final failure. If None,
                          re-raises the exception (default: None)
        should_retry: Optional function to determine if an exception should be retried.
                     Takes the exception as argument, returns bool.

    Returns:
        Decorated function with retry logic

    Example:
        @retry_with_backoff(max_attempts=3, base_delay=2.0, exceptions=(HTTPError,))
        def api_call(self, url):
            return requests.get(url)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exception:
                    # Check if we should retry this exception
                    if should_retry and not should_retry(exception):
                        raise exception

                    # Log retry attempt
                    if attempt < max_attempts:
                        # Calculate exponential backoff with jitter
                        delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                        logger.warning(
                            "%s failed (attempt %d/%d): %s. "
                            "Retrying in %.2f seconds...",
                            func.__name__,
                            attempt,
                            max_attempts,
                            exception,
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        # Final failure after max attempts
                        logger.error(
                            "%s failed after %d attempts: %s",
                            func.__name__,
                            max_attempts,
                            exception,
                        )
                        if return_on_failure:
                            return return_on_failure()
                        # Explicitly re-raise when no return_on_failure provided
                        raise exception

            # This should never be reached, but satisfies the type checker
            raise RuntimeError("Retry loop completed without returning or raising")

        return wrapper

    return decorator


def extend_dict(d, e):
    """Extend dictionary d with entries from e, returning a new dictionary."""
    r = d.copy()
    r.update(e)
    return r


def find_backlog_and_done_columns(burnup_data: pd.DataFrame) -> tuple:
    """Find backlog and done columns in burnup data."""
    backlog_column = None
    done_column = None

    for col in burnup_data.columns:
        if col.lower() in ["backlog", "backlog items"]:
            backlog_column = col
        elif col.lower() in ["done", "done items"]:
            done_column = col

    return backlog_column, done_column


def _create_generic_sampler(
    data: pd.DataFrame, column_name: str, sample_buffer_size: int = 100
) -> callable:
    """Create a generic sampler function to eliminate code duplication."""
    try:
        if data is None or data.empty:
            return lambda: 0

        # Create sample buffer
        sample_buffer: Dict[str, Any] = {
            "buffer": None,
            "idx": 0,
            # Convert column to numpy array for consistent numpy array interface
            # (needed for efficient/random sampling via random.choices); note that
            # to_numpy() may create a copy of the data; use explicit len()
            # Missing columns will raise KeyError which is caught by the error handler
            "data": data[column_name].to_numpy(),
        }

        def get_sample():
            buffer_list: Optional[List[Any]] = sample_buffer["buffer"]
            # If data is empty, return 0 (explicit length check supports numpy/list)
            if len(sample_buffer["data"]) == 0:
                return 0
            if buffer_list is None or sample_buffer["idx"] >= len(buffer_list):
                # Refill buffer
                sample_buffer["buffer"] = random.choices(
                    sample_buffer["data"], k=sample_buffer_size
                )
                sample_buffer["idx"] = 0
                buffer_list = sample_buffer["buffer"]

            if buffer_list is not None:
                sample = buffer_list[sample_buffer["idx"]]
                sample_buffer["idx"] += 1
                return sample
            return 0

        return get_sample

    except (ValueError, TypeError, KeyError, AttributeError) as e:
        logger.error("Error creating sampler: %s", e)
        return lambda: 0


def create_throughput_sampler(
    throughput_data: pd.DataFrame, sample_buffer_size: int = 100
) -> callable:
    """Create a common throughput sampler function to eliminate code duplication."""
    # Check for None or empty data first to avoid calling with irrelevant column name
    if throughput_data is None or throughput_data.empty:
        return _create_generic_sampler(None, "count", sample_buffer_size)
    # Throughput DataFrame has column "count", not "throughput"
    # Try "throughput" first for compatibility, fall back to "count"
    column_name = "throughput" if "throughput" in throughput_data.columns else "count"
    return _create_generic_sampler(throughput_data, column_name, sample_buffer_size)


def to_json_string(value):
    """Convert value to JSON-serializable string format."""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if value in (None, np.nan, pd.NaT):
        return ""

    try:
        return str(value)
    except TypeError:
        return value


def get_extension(filename):
    """Get file extension from filename."""
    return os.path.splitext(filename)[1].lower()


def to_days_since_epoch(d):
    """Convert date to days since epoch."""
    return (d - datetime.date(1970, 1, 1)).days


def set_chart_context(context):
    """Set seaborn chart context."""
    sns.set_context(context)


def breakdown_by_month(df, breakdown_config):
    """Break down DataFrame by month based on configuration.

    Args:
        df: DataFrame to break down
        breakdown_config: Dictionary containing:
            - start_column: Column name for start timestamp
            - end_column: Column name for end timestamp
            - key_column: Column name for unique identifier
            - value_column: Column name for categorical value
            - output_columns: Optional list of output columns
            - aggfunc: Aggregation function (default: "count")
    """
    start_column = breakdown_config["start_column"]
    end_column = breakdown_config["end_column"]
    key_column = breakdown_config["key_column"]
    value_column = breakdown_config["value_column"]
    output_columns = breakdown_config.get("output_columns")
    aggfunc = breakdown_config.get("aggfunc", "count")

    # If `df` is a DataFrame of items that are valid/active between the
    # timestamps stored in `start_column` and `end_column`, and where each item
    # is uniquely identified by `key_column` and has a categorical value in
    # `value_column`, return a new DataFrame counting the number of items in
    # each month broken down by each unique value in `value_column`. To restrict
    # (and order) the value columns, pass a list of valid values as
    # `output_columns`.

    def build_df(t):
        start_date = getattr(t, start_column)
        end_date = getattr(t, end_column)
        key = getattr(t, key_column)
        value = getattr(t, value_column)

        if end_date is pd.NaT:
            end_date = pd.Timestamp.today()

        first_month = start_date.normalize().to_period("M").to_timestamp("D", "S")
        last_month = end_date.normalize().to_period("M").to_timestamp("D", "S")

        index = pd.date_range(first_month, last_month, freq="MS")

        return pd.DataFrame(index=index, data=[[key]], columns=[value])

    breakdown = (
        pd.concat([build_df(t) for t in df.itertuples()], sort=True)
        .resample("MS")
        .agg(aggfunc)
    )

    if output_columns:
        breakdown = breakdown[[s for s in output_columns if s in breakdown.columns]]

    return breakdown


def breakdown_by_month_sum_days(df, breakdown_config):
    """Break down DataFrame by month with sum of days based on configuration.

    Args:
        df: DataFrame to break down
        breakdown_config: Dictionary containing:
            - start_column: Column name for start timestamp
            - end_column: Column name for end timestamp
            - value_column: Column name for categorical value
            - output_columns: Optional list of output columns
            - aggfunc: Aggregation function (default: "sum")
    """
    start_column = breakdown_config["start_column"]
    end_column = breakdown_config["end_column"]
    value_column = breakdown_config["value_column"]
    output_columns = breakdown_config.get("output_columns")
    aggfunc = breakdown_config.get("aggfunc", "sum")

    # If `df` is a DataFrame of items that are valid/active between the
    # timestamps stored in `start_column` and `end_column`, and where each has a
    # categorical value in `value_column`, return a new DataFrame summing the
    # overlapping days of items in each month broken down by each unique value in
    # `value_column`. To restrict (and order) the value columns, pass a list of
    # valid values as `output_columns`.

    def build_df(t):
        start_date = getattr(t, start_column)
        end_date = getattr(t, end_column)
        value = getattr(t, value_column)

        if end_date is pd.NaT:
            end_date = pd.Timestamp.today()

        days_range = pd.date_range(start_date, end_date, freq="D")
        first_month = start_date.normalize().to_period("M").to_timestamp("D", "S")
        last_month = end_date.normalize().to_period("M").to_timestamp("D", "S")

        index = pd.date_range(first_month, last_month, freq="MS")

        return pd.DataFrame(
            index=index,
            data=[
                [
                    len(
                        pd.date_range(
                            month_start,
                            month_start + pd.tseries.offsets.MonthEnd(1),
                            freq="D",
                        ).intersection(days_range)
                    )
                ]
                for month_start in index
            ],
            columns=[value],
        )

    breakdown = (
        pd.concat([build_df(t) for t in df.itertuples()], sort=True)
        .resample("MS")
        .agg(aggfunc)
    )

    if output_columns:
        breakdown = breakdown[[s for s in output_columns if s in breakdown.columns]]

    return breakdown


def to_bin(value, edges):
    """Pass a list of numbers in `edges` and return which of them `value` falls
    between. If < the first item, return (0, <first>). If > last item, return
    (<last>, None).
    """

    previous = 0
    for v in edges:
        if previous <= value <= v:
            return (previous, v)
        previous = v
    return (previous, None)


def create_monthly_bar_chart(breakdown, title, output_file, chart_title_key=None):
    """Create a common monthly bar chart with standard styling.

    Args:
        breakdown: DataFrame with monthly breakdown data
        title: Chart title
        output_file: Path to save the chart
        chart_title_key: Optional settings key for custom chart title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    breakdown.plot.bar(ax=ax, stacked=True)

    # Apply common chart styling directly
    if chart_title_key:
        ax.set_title(chart_title_key)

    apply_common_chart_styling(ax, breakdown)

    # Write file
    logger.info("Writing %s chart to %s", title, output_file)
    fig.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close(fig)


def create_common_defects_columns():
    """Create common defects columns used for defect tracking.

    Returns a list of column names for defect data including issue key,
    priority, type, environment, created date, and resolved date.

    Returns:
        List[str]: List of column names for defects data
    """
    return [
        "key",
        "priority",
        "type",
        "environment",
        "created",
        "resolved",
    ]
