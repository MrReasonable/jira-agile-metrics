"""Monte Carlo simulation service for forecast calculations."""

import logging
import random
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .forecast_utils import run_single_trial

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """Handles Monte Carlo simulation for burnup forecasting.

    The trust metrics returned by `_calculate_trust_metrics` now return percentiles
    as datetime objects when forecast_dates are available in simulation_params,
    falling back to numeric index values when date information is missing. This provides
    more meaningful dates for stakeholders instead of abstract period indices.
    """

    def __init__(
        self,
        trials: int = 1000,
        random_seed: int | None = None,
        confidence: float = 0.8,
    ):
        """Initialize Monte Carlo simulator with configurable parameters.

        Args:
            trials: Number of simulation trials to run. Must be a positive integer.
            random_seed: Random seed for reproducibility. Must be None or an integer.
            confidence: Confidence level for trust metrics, must be between 0 and 1
                (exclusive of 0, inclusive of 1).
                Default is 0.8 (80% confidence interval). This represents a balanced
                confidence level suitable for most forecasting scenarios - high enough
                to provide meaningful predictions while allowing for reasonable
                uncertainty bounds. Common alternatives: 0.9 (90%) for more
                conservative, 0.95 (95%) for very conservative, or 0.68 (68%, one
                sigma) for tighter bounds.

        Raises:
            ValueError: If trials is not a positive integer,
                random_seed is not None or an integer, or
                confidence is not between 0 and 1 (exclusive of 0, inclusive of 1).
        """
        # Validate trials
        if not isinstance(trials, int) or trials <= 0:
            raise ValueError(
                f"trials must be a positive integer, got {trials} "
                f"(type: {type(trials).__name__})"
            )

        # Validate random_seed
        if random_seed is not None and not isinstance(random_seed, int):
            raise ValueError(
                f"random_seed must be None or an integer, got {random_seed} "
                f"(type: {type(random_seed).__name__})"
            )

        # Validate confidence (explicitly reject booleans which are subclasses of int)
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 < confidence <= 1
        ):
            raise ValueError(
                f"confidence must be a number between 0 and 1 (exclusive of 0), "
                f"and boolean values are not allowed; got {confidence} "
                f"(type: {type(confidence).__name__})"
            )

        self.trials = trials
        self.random_seed = random_seed
        self.confidence = float(confidence)

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
        simulation_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate trustworthiness metrics for the forecast.

        Returns percentiles and statistics as dates when forecast_dates are available,
        falling back to index-based values when date information is missing.

        Returns:
            Dictionary containing:
            - p10, p25, p50, p75, p90: Percentile values (dates or indices)
            - mean, std: Mean and standard deviation (dates or indices)
            - confidence: Confidence level
            - use_dates: Boolean indicating if dates were used (True) or indices (False)
        """
        try:
            if not trials:
                return {}

            # Get target from simulation params to calculate completion dates
            target = simulation_params.get("target", 0)
            if target == 0:
                return {}

            # Calculate completion indices based on when done_trial exceeds target
            # Note: done_trial[0] is initial state, so we skip it and look from index 1
            completion_indices = []
            for trial in trials:
                done_trial = trial.get("done_trial", [])
                if done_trial and len(done_trial) > 1:
                    # Skip index 0 (initial state) and find where we reach the target
                    for idx in range(1, len(done_trial)):
                        if done_trial[idx] >= target:
                            # idx-1 is forecast period index (0 = first period)
                            completion_indices.append(idx - 1)
                            break

            if not completion_indices:
                logger.warning(
                    "No trials reached target %d. Trust metrics cannot be calculated.",
                    target,
                )
                return {}

            # Try to use dates if available, fall back to indices
            forecast_dates = simulation_params.get("forecast_dates", [])
            use_dates = self._can_use_dates(forecast_dates)

            if use_dates:
                return self._calculate_date_based_metrics(
                    completion_indices, forecast_dates
                )

            # Fall back to index-based calculation
            return self._calculate_index_based_metrics(completion_indices)

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error calculating trust metrics: %s", e)
            return {}

    def _calculate_date_based_metrics(
        self, completion_indices: List[int], forecast_dates: List[Any]
    ) -> Dict[str, Any]:
        """Calculate trust metrics using date-based percentiles.

        Args:
            completion_indices: List of completion indices
            forecast_dates: List of forecast dates to map indices to

        Returns:
            Dictionary with date-based percentile metrics
        """
        # Convert completion indices to dates
        completion_dates = self._convert_indices_to_dates(
            completion_indices, forecast_dates
        )

        # Convert dates to numeric values (timestamps) for percentile
        # Use pandas Timestamp for conversion to ensure consistent handling
        numeric_dates = []
        for date in completion_dates:
            if isinstance(date, (int, float)):
                numeric_dates.append(float(date))
            else:
                # Convert to pandas Timestamp and then to numeric timestamp
                ts = pd.Timestamp(date)
                numeric_dates.append(ts.value)  # Use nanosecond timestamp

        # Calculate quantiles on numeric values
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        quantile_numeric = np.percentile(numeric_dates, [q * 100 for q in quantiles])

        # Convert numeric timestamp values back to dates
        quantile_dates = [
            pd.Timestamp(int(q)).to_pydatetime() for q in quantile_numeric
        ]

        # Calculate mean and std on numeric values
        mean_numeric = np.mean(numeric_dates)
        std_numeric = np.std(numeric_dates)

        # Convert mean back to date, but std remains as days (numeric)
        # Convert std from nanoseconds to days
        std_days = std_numeric / (1e9 * 60 * 60 * 24)  # nanoseconds to days

        return {
            "p10": quantile_dates[0],
            "p25": quantile_dates[1],
            "p50": quantile_dates[2],
            "p75": quantile_dates[3],
            "p90": quantile_dates[4],
            "mean": pd.Timestamp(int(mean_numeric)).to_pydatetime(),
            "std": float(std_days),  # Standard deviation in days
            "confidence": self.confidence,
            "use_dates": True,
        }

    def _calculate_index_based_metrics(
        self, completion_indices: List[int]
    ) -> Dict[str, Any]:
        """Calculate trust metrics using index-based percentiles.

        Args:
            completion_indices: List of completion indices

        Returns:
            Dictionary with index-based percentile metrics
        """
        logger.debug("Forecast dates not available, using index-based percentiles")

        # Calculate quantiles on indices
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        quantile_values = np.percentile(
            completion_indices, [q * 100 for q in quantiles]
        )

        return {
            "p10": float(quantile_values[0]),
            "p25": float(quantile_values[1]),
            "p50": float(quantile_values[2]),
            "p75": float(quantile_values[3]),
            "p90": float(quantile_values[4]),
            "mean": float(np.mean(completion_indices)),
            "std": float(np.std(completion_indices)),
            "confidence": self.confidence,
            "use_dates": False,
        }

    def _can_use_dates(self, forecast_dates: Any) -> bool:
        """Check if dates can be used for percentile calculation.

        Args:
            forecast_dates: Forecast dates from simulation params
                (can be list, array, etc.)

        Returns:
            True if dates are available and valid, False otherwise
        """
        if not forecast_dates:
            return False

        try:
            # Convert to list if it's a pandas DatetimeIndex or similar
            if hasattr(forecast_dates, "tolist"):
                dates_list = forecast_dates.tolist()
            elif isinstance(forecast_dates, (list, tuple)):
                dates_list = list(forecast_dates)
            else:
                return False

            if not dates_list:
                return False

            # We can use dates if we have at least one date
            # (extrapolation handles beyond range)
            return True

        except (ValueError, TypeError, AttributeError):
            return False

    def _extrapolate_date(self, idx: int, forecast_dates: List[Any]) -> datetime:
        """Extrapolate a date beyond the forecast_dates range.

        Args:
            idx: Index position that may be beyond the forecast_dates range.
            forecast_dates: List of forecast dates to extrapolate from.

        Returns:
            Extrapolated datetime based on the interval between forecast dates.

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
                f"Invalid index value: {idx}. " "Index must be a non-negative integer."
            )

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

    def _convert_indices_to_dates(
        self, completion_indices: List[int], forecast_dates: List[Any]
    ) -> List[datetime]:
        """Convert completion indices to actual dates.

        Args:
            completion_indices: List of indices where completion occurred.
            forecast_dates: List of forecast dates to map indices to.

        Returns:
            List of completion dates. Handles extrapolation for indices beyond
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
                    completion_dates.append(pd.Timestamp(date).to_pydatetime())
            else:
                # Extrapolate beyond the forecast range
                completion_dates.append(self._extrapolate_date(idx, dates_list))

        return completion_dates

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
