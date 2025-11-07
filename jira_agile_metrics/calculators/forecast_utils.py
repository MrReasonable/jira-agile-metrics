"""Utility functions for forecast calculations."""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List

import pandas as pd

from ..utils import _create_generic_sampler, create_throughput_sampler

logger = logging.getLogger(__name__)


def calculate_daily_throughput(
    cycle_data: pd.DataFrame, done_column: str
) -> pd.DataFrame:
    """Calculate daily throughput from cycle data."""
    try:
        if (
            cycle_data is None
            or cycle_data.empty
            or done_column not in cycle_data.columns
        ):
            return pd.DataFrame()

        # Group by completion date and count items
        daily_throughput = cycle_data.groupby(done_column).size().reset_index()
        daily_throughput.columns = ["date", "throughput"]
        daily_throughput = daily_throughput.set_index("date")

        return daily_throughput

    except (ValueError, TypeError, KeyError, AttributeError) as e:
        logger.error("Error calculating daily throughput: %s", e)
        return pd.DataFrame()


def throughput_sampler(
    throughput_data: pd.DataFrame, sample_buffer_size: int = 100
) -> Callable[..., Any]:
    """Create a throughput sampler function."""
    return create_throughput_sampler(throughput_data, sample_buffer_size)


def calculate_daily_backlog_growth(
    burnup_data: pd.DataFrame, backlog_column: str
) -> pd.DataFrame:
    """Calculate daily backlog growth from burnup data."""
    try:
        if (
            burnup_data is None
            or burnup_data.empty
            or backlog_column not in burnup_data.columns
        ):
            return pd.DataFrame()

        # Calculate daily changes
        backlog_changes = burnup_data[backlog_column].diff().dropna()

        # Filter out negative changes (items being completed)
        growth_only = backlog_changes[backlog_changes > 0]

        return growth_only.to_frame("backlog_growth")

    except (ValueError, TypeError, KeyError, AttributeError) as e:
        logger.error("Error calculating daily backlog growth: %s", e)
        return pd.DataFrame()


def backlog_growth_sampler(
    backlog_growth_data: pd.DataFrame, sample_buffer_size: int = 100
) -> Callable[..., Any]:
    """Create a backlog growth sampler function."""
    return _create_generic_sampler(
        backlog_growth_data, "backlog_growth", sample_buffer_size
    )


def burnup_monte_carlo_horizon(simulation_params: Dict[str, Any]) -> Dict[str, Any]:
    """Run Monte Carlo simulation for burnup forecast horizon."""
    try:
        # Extract parameters
        trials = simulation_params.get("trials", 1000)
        throughput_sampler_func = simulation_params.get("throughput_sampler")
        backlog_growth_sampler_func = simulation_params.get("backlog_growth_sampler")

        if not throughput_sampler_func or not backlog_growth_sampler_func:
            missing_samplers = []
            if not throughput_sampler_func:
                missing_samplers.append("throughput_sampler")
            if not backlog_growth_sampler_func:
                missing_samplers.append("backlog_growth_sampler")
            logger.warning(
                "Required samplers are missing: %s. Cannot run Monte Carlo simulation.",
                ", ".join(missing_samplers),
            )
            return {}

        # Run trials
        trial_results = []
        for trial_num in range(trials):
            trial_result = run_single_trial(simulation_params, trial_num)
            trial_results.append(trial_result)

        return {"trials": trial_results, "num_trials": trials}

    except (ValueError, TypeError, KeyError, AttributeError) as e:
        logger.error("Error running Monte Carlo horizon: %s", e)
        return {}


def run_single_trial(
    simulation_params: Dict[str, Any], trial_num: int
) -> Dict[str, Any]:
    """Run a single simulation trial."""
    try:
        # Extract parameters
        forecast_dates = simulation_params.get("forecast_dates", [])
        throughput_sampler_func = simulation_params.get("throughput_sampler")
        backlog_growth_sampler_func = simulation_params.get("backlog_growth_sampler")
        initial_backlog = simulation_params.get("initial_backlog", 0)
        initial_done = simulation_params.get("initial_done", 0)

        # Initialize trial state
        current_backlog = initial_backlog
        current_done = initial_done

        backlog_trial = [current_backlog]
        done_trial = [current_done]

        # Simulate each forecast period
        for _ in forecast_dates:
            # Sample throughput and backlog growth
            throughput = throughput_sampler_func() if throughput_sampler_func else 0
            backlog_growth = (
                backlog_growth_sampler_func() if backlog_growth_sampler_func else 0
            )

            # Update state
            # Backlog and Done are independent cumulative metrics
            # Backlog grows by backlog_growth only (not affected by throughput)
            current_backlog += backlog_growth
            # Done grows by throughput only (not affected by backlog)
            current_done += throughput

            # Ensure backlog doesn't go negative
            current_backlog = max(0, current_backlog)

            backlog_trial.append(current_backlog)
            done_trial.append(current_done)

        return {
            "trial_num": trial_num,
            "backlog_trial": backlog_trial,
            "done_trial": done_trial,
            "final_backlog": current_backlog,
            "final_done": current_done,
        }

    except (ValueError, TypeError, KeyError, AttributeError) as e:
        logger.error("Error running single trial: %s", e)
        return {
            "trial_num": trial_num,
            "backlog_trial": [],
            "done_trial": [],
            "final_backlog": 0,
            "final_done": 0,
        }


def find_completion_indices(done_trials: List[List[float]], target: int) -> List[int]:
    """Find completion indices when done_trial exceeds target.

    Args:
        done_trials: List of done_trial lists from simulation trials.
        target: Target value to find when each trial exceeds.

    Returns:
        List of completion indices. Each index represents the forecast period
        (0 = first period) where the target was reached. Trials that don't
        reach the target are skipped.
    """
    completion_indices = []
    for done_trial in done_trials:
        if not isinstance(done_trial, list) or len(done_trial) < 2:
            continue
        # Skip index 0 (initial state) and find where we reach the target
        for idx in range(1, len(done_trial)):
            if done_trial[idx] >= target:
                # idx-1 corresponds to forecast_dates index (0 = first period)
                completion_indices.append(idx - 1)
                break
    return completion_indices


def extrapolate_date(idx: int, forecast_dates: List[Any]) -> datetime:
    """Extrapolate a date beyond the forecast_dates range.

    Args:
        idx: Index position that may be beyond the forecast_dates range.
        forecast_dates: List of forecast dates to extrapolate from.

    Returns:
        Extrapolated datetime based on the interval between forecast dates.
        If idx is within the forecast_dates range, returns the corresponding
        forecast date converted to datetime.

    Raises:
        ValueError: If forecast_dates is empty or idx is negative.
    """
    if not forecast_dates:
        raise ValueError(
            "Cannot extrapolate date: forecast_dates is empty. "
            "At least one forecast date is required."
        )

    if not isinstance(idx, int) or idx < 0:
        raise ValueError(
            f"Invalid index value: {idx}. Index must be a non-negative integer."
        )

    # If idx is within the forecast_dates range, return the corresponding date
    if idx < len(forecast_dates):
        date = pd.Timestamp(forecast_dates[idx])
        if isinstance(date, pd.Timestamp):
            return date.to_pydatetime()
        return date

    # Only extrapolate when idx >= len(forecast_dates)
    # Convert dates to pandas Timestamps for consistent handling
    dates = [pd.Timestamp(d) for d in forecast_dates]
    last_date = dates[-1]

    # Calculate interval between dates
    if len(dates) > 1:
        interval = dates[1] - dates[0]
    else:
        interval = pd.Timedelta(days=1)

    # Calculate steps beyond the last date
    steps_beyond = idx - len(dates) + 1
    extrapolated = last_date + steps_beyond * interval

    # Convert to datetime if needed
    if isinstance(extrapolated, pd.Timestamp):
        return extrapolated.to_pydatetime()
    return extrapolated


def convert_indices_to_dates(
    completion_indices: List[int], forecast_dates: List[Any]
) -> List[datetime]:
    """Convert completion indices to actual dates.

    Args:
        completion_indices: List of indices where completion occurred.
        forecast_dates: List of forecast dates to map indices to.

    Returns:
        List of completion dates. Returns an empty list if forecast_dates
        is empty or falsy. Handles extrapolation for indices beyond
        the forecast_dates range.
    """
    if not forecast_dates:
        return []

    # Convert forecast_dates to list if needed
    if hasattr(forecast_dates, "tolist"):
        dates_list = forecast_dates.tolist()
    else:
        dates_list = list(forecast_dates)

    completion_dates = []
    for idx in completion_indices:
        if idx < len(dates_list):
            # Use the date directly from the list
            date = dates_list[idx]
            # Convert to datetime if needed
            if isinstance(date, pd.Timestamp):
                completion_dates.append(date.to_pydatetime())
            elif isinstance(date, datetime):
                completion_dates.append(date)
            else:
                # Try to convert to datetime
                try:
                    completion_dates.append(pd.Timestamp(date).to_pydatetime())
                except Exception as e:
                    error_msg = (
                        f"Failed to convert date to datetime at index {idx}: "
                        f"value={repr(date)}, type={type(date).__name__}, "
                        f"error={str(e)}"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg) from e
        else:
            # Extrapolate beyond the forecast range
            completion_dates.append(extrapolate_date(idx, dates_list))

    return completion_dates
