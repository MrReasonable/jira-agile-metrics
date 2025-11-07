"""Utility functions for burnup chart generation."""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config.exceptions import ChartGenerationError
from .forecast_utils import extrapolate_date

logger = logging.getLogger(__name__)


def format_date_for_legend(date_value: Any) -> str:
    """Format a date value for display in the legend."""

    try:
        if date_value is None:
            return "N/A"
        return pd.Timestamp(date_value).strftime("%Y-%m-%d")
    except (ValueError, TypeError) as e:
        logger.debug("Error formatting date for legend: %s", e)
        return "N/A"


def validate_figure_size(
    figure_size: Optional[Tuple[float, float]],
    default: Tuple[float, float] = (12, 8),
) -> Tuple[float, float]:
    """Validate and normalize a matplotlib figure size."""
    if figure_size is None:
        return default
    try:
        if len(figure_size) != 2 or not all(
            isinstance(x, (int, float)) and x > 0 for x in figure_size
        ):
            raise ValueError("figure_size must be a tuple of 2 positive numbers")
        return tuple(float(x) for x in figure_size)
    except (ValueError, TypeError) as e:
        logger.error("Invalid figure_size: %s. Using default %s", e, default)
        return default


def save_chart(fig: plt.Figure, output_file: str) -> None:
    """Save the chart to file."""
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Save figure
        fig.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close(fig)

    except OSError as e:
        # Catch file I/O errors from matplotlib savefig
        logger.error("Error saving chart file: %s", e)
        raise ChartGenerationError(f"Failed to save chart file: {e}") from e
    except (ValueError, TypeError) as e:
        # Catch ValueError/TypeError from matplotlib operations
        logger.error("Error saving chart: %s", e)
        raise ChartGenerationError(f"Failed to save chart: {e}") from e


def normalize_trial_length(
    trial: list, expected_length: int, trial_idx: int
) -> Optional[list]:
    """Normalize a trial to expected_length+1 (preserving initial_state).

    Args:
        trial: Trial data list
        expected_length: Expected length of forecast values
        trial_idx: Index of trial for logging purposes

    Returns:
        Normalized trial list, or None if invalid/empty. The returned list has:
        - Length expected_length when only forecast values are present
          (no initial_state)
        - Length expected_length+1 when an initial_state is included
        - None if the trial is empty or invalid
    """
    trial_len = len(trial)

    # Handle empty trials first
    if trial_len == 0:
        logger.warning("Trial %d is empty, skipping", trial_idx)
        return None

    # Handle exact matches for expected_length or expected_length+1
    if trial_len == expected_length:
        # Trial has forecast values only, no initial_state
        # Return as-is (treated as expected_length forecast values)
        return trial
    if trial_len == expected_length + 1:
        # Trial has initial_state + forecast values, correct length
        return trial

    # Handle too-long trials by truncating to expected_length+1
    if trial_len > expected_length + 1:
        normalized = trial[: expected_length + 1]
        logger.debug(
            "Trial %d truncated from %d to %d elements (includes initial state)",
            trial_idx,
            trial_len,
            len(normalized),
        )
        return normalized

    # Handle too-short trials by padding with the last value to expected_length
    # (trial_len < expected_length is the only remaining case)
    last_value = trial[-1]
    normalized = trial + [last_value] * (expected_length - trial_len)
    logger.warning(
        "Trial %d padded from %d to %d elements",
        trial_idx,
        trial_len,
        expected_length,
    )
    return normalized


def extract_forecast_trials(trials: list, expected_length: int) -> list:
    """Extract and validate forecast trials from trial data.

    Trials typically have format [initial_state] + [forecast_values...]
    where initial_state is at the last historical date. Since forecast_dates
    includes last_date (from _get_forecast_dates), we need to extract exactly
    expected_length values to match the number of forecast_dates.

    If trial length equals expected_length, use as-is.
    If trial length is expected_length + 1, extract first expected_length
    values (initial_state + first forecast values map to forecast_dates).
    If trial length > expected_length + 1, truncate to expected_length.
    """
    forecast_trials = []
    for idx, trial in enumerate(trials):
        # Validate type
        if not isinstance(trial, list):
            logger.warning(
                "Trial %d skipped: not a list (type=%s, value=%s)",
                idx,
                type(trial).__name__,
                trial,
            )
            continue

        # Normalize trial length using helper method
        normalized_trial = normalize_trial_length(trial, expected_length, idx)
        if normalized_trial is not None:
            # Extract exactly expected_length values to match forecast_dates
            if len(normalized_trial) > expected_length:
                # Take first expected_length values (initial + forecast)
                extracted_trial = normalized_trial[:expected_length]
                forecast_trials.append(extracted_trial)
            elif len(normalized_trial) == expected_length:
                # Perfect match
                forecast_trials.append(normalized_trial)
            else:
                # Shorter than expected - should have been padded already
                logger.warning(
                    "Trial %d has %d values, expected %d. Skipping.",
                    idx,
                    len(normalized_trial),
                    expected_length,
                )

    return forecast_trials


def validate_initial_state(value: Any, trial_idx: int) -> bool:
    """Validate that a value matches expected initial_state shape/marker.

    Args:
        value: The value to validate (typically trial[0])
        trial_idx: Index of trial for logging purposes

    Returns:
        True if value appears to be a valid initial_state, False otherwise
    """
    # Check if value is numeric
    if not isinstance(value, (int, float)):
        logger.warning(
            "Trial %d: First element is not numeric (type=%s). "
            "Not treating as initial_state.",
            trial_idx,
            type(value).__name__,
        )
        return False

    # Check if value is NaN or infinite
    try:
        if np.isnan(value) or np.isinf(value):
            logger.warning(
                "Trial %d: First element is NaN or infinite. "
                "Not treating as initial_state.",
                trial_idx,
            )
            return False
    except (TypeError, ValueError):
        logger.warning(
            "Trial %d: First element failed numeric validation. "
            "Not treating as initial_state.",
            trial_idx,
        )
        return False

    # Negative values are suspicious but not impossible, so log debug only
    if value < 0:
        logger.debug(
            "Trial %d: First element is negative. "
            "Treating as initial_state but this may be unexpected.",
            trial_idx,
        )

    return True


def get_frequency_string(forecast_dates: list) -> str:
    """Get frequency string from forecast dates.

    Args:
        forecast_dates: List of forecast dates

    Returns:
        Frequency string ("D", "W", "ME", or "D" as fallback)
    """
    if len(forecast_dates) <= 1:
        return "D"

    # Infer frequency from the date range
    # Note: pd.infer_freq requires at least 3 dates, so we catch ValueError
    try:
        freq_str = pd.infer_freq(forecast_dates)
        if freq_str is not None:
            return freq_str
    except ValueError:
        # Need at least 3 dates to infer frequency, fall through to manual calculation
        pass

    # Calculate frequency from the interval between dates
    date_diff = pd.Timestamp(forecast_dates[1]) - pd.Timestamp(forecast_dates[0])
    if date_diff.days == 1:
        return "D"
    if date_diff.days == 7:
        return "W"
    if 28 <= date_diff.days <= 31:
        return "ME"  # Month End

    # Fallback: use daily frequency
    return "D"


def calculate_next_date(last_date: pd.Timestamp, freq_str: str) -> pd.Timestamp:
    """Calculate next date based on frequency.

    Args:
        last_date: Last date in the sequence
        freq_str: Frequency string ("D", "W", "ME")

    Returns:
        Next date after last_date based on frequency
    """
    if freq_str == "D":
        return last_date + pd.Timedelta(days=1)
    if freq_str == "W":
        return last_date + pd.Timedelta(weeks=1)
    if freq_str == "ME":
        return last_date + pd.DateOffset(months=1)

    # Default: add one day
    return last_date + pd.Timedelta(days=1)


def find_latest_completion_date(
    quantile_data: Dict[str, Any],
) -> Optional[pd.Timestamp]:
    """Find the latest completion date from quantile data.

    Args:
        quantile_data: Dictionary with quantile completion dates

    Returns:
        Latest completion date as Timestamp, or None if not found
    """
    latest_completion_date = None
    for quantile in ["50%", "75%", "85%", "90%", "99%"]:
        if quantile not in quantile_data:
            continue
        date = quantile_data[quantile]
        if not date:
            continue
        try:
            date_ts = pd.Timestamp(date)
            if latest_completion_date is None or date_ts > latest_completion_date:
                latest_completion_date = date_ts
        except (ValueError, TypeError):
            continue
    return latest_completion_date


def _find_trial_completion_index(done_trial: List[float], target: int) -> Optional[int]:
    """Find the index when a trial reaches the target.

    Args:
        done_trial: Trial data (initial_state + forecast values)
        target: Target value to find

    Returns:
        Completion index (0-based for forecast_dates), or None if not reached
    """
    for idx in range(1, len(done_trial)):
        if done_trial[idx] >= target:
            # idx-1 corresponds to forecast_dates index (0 = first period)
            return idx - 1
    return None


def _calculate_extrapolation_rate(done_trial: List[float]) -> Optional[float]:
    """Calculate the rate of progress for extrapolation.

    Args:
        done_trial: Trial data

    Returns:
        Rate of progress per period, or None if can't calculate
    """
    if len(done_trial) < 2:
        return None

    last_value = done_trial[-1]
    if last_value <= 0:
        return None

    # Calculate average rate of progress (per forecast period)
    # Use last few values to estimate rate
    num_periods = min(3, len(done_trial) - 1)
    if num_periods > 0:
        recent_values = done_trial[-num_periods:]
        rate = (recent_values[-1] - recent_values[0]) / num_periods
    else:
        rate = done_trial[-1] - done_trial[0]

    return rate if rate > 0 else None


def _get_completion_date_from_index(
    completion_idx: int, forecast_dates: list
) -> datetime:
    """Get completion date from completion index.

    Args:
        completion_idx: Index in forecast_dates
        forecast_dates: List of forecast dates

    Returns:
        Completion date as datetime
    """
    if completion_idx < len(forecast_dates):
        date = forecast_dates[completion_idx]
        if isinstance(date, pd.Timestamp):
            return date.to_pydatetime()
        return pd.Timestamp(date).to_pydatetime()
    # Extrapolate beyond forecast horizon
    return extrapolate_date(completion_idx, forecast_dates)


def _extrapolate_completion_date(
    done_trial: List[float],
    target: int,
    last_forecast_idx: int,
    forecast_dates: list,
) -> Optional[datetime]:
    """Extrapolate completion date for a trial that hasn't reached target.

    Args:
        done_trial: Trial data
        target: Target value
        last_forecast_idx: Last index in forecast_dates
        forecast_dates: List of forecast dates

    Returns:
        Extrapolated completion date, or None if can't extrapolate
    """
    rate = _calculate_extrapolation_rate(done_trial)
    if rate is None:
        return None

    last_value = done_trial[-1]
    remaining = target - last_value
    periods_needed = max(1, int(np.ceil(remaining / rate)))
    extrapolated_idx = last_forecast_idx + periods_needed
    return extrapolate_date(extrapolated_idx, forecast_dates)


def find_max_completion_date_from_trials(
    done_trials: List[List[float]], target: int, forecast_dates: list
) -> Optional[pd.Timestamp]:
    """Find the maximum completion date across ALL trials.

    This finds when the slowest trial reaches the target, ensuring all trials
    have reached the target by this date. For trials that haven't reached
    the target within the forecast horizon, extrapolates based on their
    current rate of progress.

    Args:
        done_trials: List of done_trial lists from simulation
        target: Target value to find when each trial exceeds
        forecast_dates: List of forecast dates to map indices to

    Returns:
        Maximum completion date as Timestamp, or None if not found
    """
    if not done_trials or not target or not forecast_dates:
        return None

    completion_dates = []
    last_forecast_idx = len(forecast_dates) - 1

    for done_trial in done_trials:
        if not isinstance(done_trial, list) or len(done_trial) < 2:
            continue

        completion_idx = _find_trial_completion_index(done_trial, target)
        if completion_idx is not None:
            date = _get_completion_date_from_index(completion_idx, forecast_dates)
            completion_dates.append(date)
        else:
            # Trial hasn't reached target - extrapolate
            date = _extrapolate_completion_date(
                done_trial, target, last_forecast_idx, forecast_dates
            )
            if date is not None:
                completion_dates.append(date)

    if not completion_dates:
        return None

    # Find the maximum (latest) completion date
    max_date = max(completion_dates)
    return pd.Timestamp(max_date)


def extend_forecast_dates_to_completion(
    forecast_dates: list,
    quantile_data: Dict[str, Any],
    done_trials: Optional[List[List[float]]] = None,
    target: int = 0,
) -> list:
    """Extend forecast dates to include latest completion date if needed.

    If done_trials and target are provided, extends to when ALL trials reach
    the target. Otherwise, uses quantile_data to find the latest percentile date.

    Args:
        forecast_dates: Current forecast dates
        quantile_data: Quantile completion dates
        done_trials: Optional list of done_trial lists from simulation
        target: Optional target value to find when all trials exceed

    Returns:
        Extended forecast dates (or original if no extension needed)
    """
    # Prefer finding max completion date from all trials if available
    if done_trials and target > 0:
        latest_completion_date = find_max_completion_date_from_trials(
            done_trials, target, forecast_dates
        )
    else:
        # Fall back to quantile data
        latest_completion_date = find_latest_completion_date(quantile_data)

    if not latest_completion_date:
        return forecast_dates

    last_forecast_date = pd.Timestamp(forecast_dates[-1])
    if latest_completion_date <= last_forecast_date:
        return forecast_dates

    # Extend forecast_dates to include latest completion date
    freq_str = get_frequency_string(forecast_dates)
    next_date = calculate_next_date(last_forecast_date, freq_str)

    extended_dates = pd.date_range(
        start=next_date,
        end=latest_completion_date,
        freq=freq_str,
    ).tolist()

    extended_forecast_dates = forecast_dates + extended_dates
    logger.debug(
        "Extended forecast_dates from %s to %s to include latest completion date %s",
        last_forecast_date,
        extended_forecast_dates[-1],
        latest_completion_date,
    )
    return extended_forecast_dates
