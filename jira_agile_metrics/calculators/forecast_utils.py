"""Utility functions for forecast calculations."""

import logging
from typing import Any, Callable, Dict

import pandas as pd

from ..utils import _create_generic_sampler, create_throughput_sampler

logger = logging.getLogger(__name__)


def calculate_daily_throughput(
    cycle_data: pd.DataFrame, done_column: str
) -> pd.DataFrame:
    """Calculate daily throughput from cycle data."""
    try:
        if cycle_data is None or done_column not in cycle_data.columns:
            return pd.DataFrame()

        # Group by completion date and count items
        daily_throughput = cycle_data.groupby(done_column).size().reset_index()
        daily_throughput.columns = ["date", "throughput"]
        daily_throughput.set_index("date", inplace=True)

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
        if burnup_data is None or backlog_column not in burnup_data.columns:
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
            current_done += throughput
            current_backlog += backlog_growth - throughput

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
