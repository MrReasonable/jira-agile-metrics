"""Monte Carlo simulation service for forecast calculations."""

import logging
import random
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .forecast_utils import run_single_trial

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """Handles Monte Carlo simulation for burnup forecasting."""

    def __init__(self):
        self.trials = 1000  # Default number of trials
        self.random_seed = None

    def run_simulation(self, simulation_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run Monte Carlo simulation for burnup forecast."""
        try:
            # Set random seed for reproducibility
            if self.random_seed is not None:
                random.seed(self.random_seed)
                np.random.seed(self.random_seed)

            # Run simulation trials
            trials_result = self._run_trials(simulation_params)

            # Calculate trust metrics
            trust_metrics = self._calculate_trust_metrics(
                trials_result, simulation_params
            )

            return {
                "trials": trials_result,
                "trust_metrics": trust_metrics,
                "num_trials": self.trials,
            }

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error running Monte Carlo simulation: %s", e)
            return {}

    def _run_trials(self, simulation_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run individual simulation trials."""
        trials = []

        for trial_num in range(self.trials):
            trial_result = run_single_trial(simulation_params, trial_num)
            trials.append(trial_result)

        return trials

    def _calculate_trust_metrics(
        self,
        trials: List[Dict[str, Any]],
        _simulation_params: Dict[str, Any],
    ) -> Dict[str, float]:
        """Calculate trustworthiness metrics for the forecast."""
        try:
            if not trials:
                return {}

            # Get target from simulation params to calculate completion dates
            target = _simulation_params.get("target", 0)
            if target == 0:
                return {}

            # Calculate completion dates based on when done_trial exceeds target
            completion_dates = []
            for trial in trials:
                done_trial = trial.get("done_trial", [])
                if done_trial and len(done_trial) > 0:
                    # Find the index where we reach the target
                    for idx, done_value in enumerate(done_trial):
                        if done_value >= target:
                            # Calculate approximate date based on index
                            completion_dates.append(idx)
                            break

            if not completion_dates:
                return {}

            # Calculate quantiles
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
            quantile_values = np.percentile(
                completion_dates, [q * 100 for q in quantiles]
            )

            return {
                "p10": quantile_values[0],
                "p25": quantile_values[1],
                "p50": quantile_values[2],
                "p75": quantile_values[3],
                "p90": quantile_values[4],
                "mean": np.mean(completion_dates),
                "std": np.std(completion_dates),
                "confidence": 0.8,  # Default confidence level
            }

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error calculating trust metrics: %s", e)
            return {}

    def setup_backlog_growth_sampler(
        self,
        burnup_data: pd.DataFrame,
        backlog_column: str,
        _window_params: Dict[str, Any],
    ) -> callable:
        """Setup backlog growth sampler function."""
        try:
            # Calculate daily backlog growth
            backlog_growth_data = self._calculate_daily_backlog_growth(
                burnup_data, backlog_column
            )

            # Convert to list once to avoid pandas deprecation warning
            # and to avoid repeated conversions in the sampler function
            backlog_growth_list = backlog_growth_data.tolist()

            # Create sampler function
            def get_backlog_growth_sample():
                if not backlog_growth_list:
                    return 0.0
                return random.choice(backlog_growth_list)

            return get_backlog_growth_sample

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error setting up backlog growth sampler: %s", e)
            return lambda: 0.0

    def _calculate_daily_backlog_growth(
        self, burnup_data: pd.DataFrame, backlog_column: str
    ) -> pd.Series:
        """Calculate daily backlog growth from burnup data."""
        try:
            if backlog_column not in burnup_data.columns:
                return pd.Series()

            # Calculate daily changes
            backlog_changes = burnup_data[backlog_column].diff().dropna()

            # Filter out negative changes (items being completed)
            growth_only = backlog_changes[backlog_changes > 0]

            return growth_only

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error calculating daily backlog growth: %s", e)
            return pd.Series()
