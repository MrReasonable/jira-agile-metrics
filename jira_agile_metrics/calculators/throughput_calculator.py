"""Throughput calculation service for forecast calculations."""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from ..utils import create_throughput_sampler
from .throughput import calculate_throughput

logger = logging.getLogger(__name__)

# Adaptive window configuration constants
ADAPTIVE_MIN_WINDOW_DAYS = 14  # Minimum window size for statistical validity
ADAPTIVE_MAX_WINDOW_DAYS = 90  # Maximum window size to avoid outdated patterns
ADAPTIVE_TARGET_COMPLETIONS = 30  # Target number of completed items in adaptive window


class ThroughputCalculator:
    """Handles throughput calculations for forecast analysis."""

    def calculate_throughput(
        self,
        cycle_data: pd.DataFrame,
        done_column: str,
        forecast_params: Dict[str, Any],
    ) -> Optional[pd.DataFrame]:
        """Calculate throughput using smart or fixed window logic."""
        try:
            # Determine window parameters
            window_params = self._calculate_window_parameters(forecast_params)
            if window_params is None:
                return None

            # Calculate throughput based on window type
            if window_params.get("smart_window", False):
                throughput_data = self._calculate_smart_window_throughput(
                    cycle_data, done_column, window_params
                )
            else:
                throughput_data = self._calculate_fixed_window_throughput(
                    cycle_data, done_column, window_params
                )

            return throughput_data

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error calculating throughput: %s", e)
            return None

    def _calculate_window_parameters(
        self, forecast_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Calculate window-related parameters."""
        try:
            freq = forecast_params.get("freq", "D")
            freq_label = forecast_params.get("freq_label", "day")

            # Default window size based on frequency
            if freq == "D":
                window_size = 30  # 30 days
            elif freq == "W":
                window_size = 4  # 4 weeks
            else:
                window_size = 1  # 1 month

            # Read smart_window from forecast_params with validation
            smart_window_raw = forecast_params.get("smart_window", True)
            if isinstance(smart_window_raw, bool):
                smart_window = smart_window_raw
            elif isinstance(smart_window_raw, str):
                # Handle string representations of boolean values
                smart_window = smart_window_raw.lower() in ("true", "1", "yes", "on")
                logger.warning(
                    "smart_window parameter received string value '%s', "
                    "converted to boolean: %s",
                    smart_window_raw,
                    smart_window,
                )
            else:
                # Fall back to default for any other invalid type
                smart_window = True
                logger.warning(
                    "smart_window parameter has invalid type (%s), "
                    "expected boolean, falling back to default: True",
                    type(smart_window_raw).__name__,
                )

            return {
                "freq": freq,
                "freq_label": freq_label,
                "window_size": window_size,
                "smart_window": smart_window,
            }

        except (ValueError, TypeError, KeyError) as e:
            logger.error("Error calculating window parameters: %s", e)
            return None

    def _calculate_smart_window_throughput(
        self, cycle_data: pd.DataFrame, done_column: str, window_params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Calculate throughput using smart window logic.

        Determines the optimal window size based on historical completion patterns
        and calculates throughput using that adaptive window.
        """
        try:
            frequency = window_params.get("freq", "D")

            # Calculate optimal window size for smart window
            window_size = self._determine_optimal_window_from_data(
                cycle_data, done_column, frequency
            )

            logger.info(
                "Smart window: using adaptive window size of %d %s",
                window_size,
                window_params.get("freq_label", "periods"),
            )

            return calculate_throughput(cycle_data, frequency, window_size)

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error in smart window throughput calculation: %s", e)
            # Fall back to fixed window
            frequency = window_params.get("freq", "D")
            window = window_params.get("window_size")
            return calculate_throughput(cycle_data, frequency, window)

    def _calculate_fixed_window_throughput(
        self, cycle_data: pd.DataFrame, _done_column: str, window_params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Calculate throughput using fixed window logic.

        Uses the configured window size without adaptive analysis.
        """
        frequency = window_params.get("freq", "D")
        window = window_params.get("window_size")
        freq_label = window_params.get("freq_label", "periods")

        if window:
            logger.debug(
                "Fixed window: using configured window size of %d %s",
                window,
                freq_label,
            )
        else:
            logger.debug(
                "Fixed window: using all available data (%s)",
                freq_label,
            )

        return calculate_throughput(cycle_data, frequency, window)

    def _determine_optimal_window_from_data(
        self, cycle_data: pd.DataFrame, done_column: str, freq: str
    ) -> int:
        """Determine optimal window size for smart window based on available data.

        Args:
            cycle_data: DataFrame with cycle time data
            done_column: Name of the column containing done timestamps
            freq: Frequency string ('D', 'W', 'M')

        Returns:
            Optimal window size in periods
        """
        try:
            if done_column not in cycle_data.columns:
                logger.warning(
                    "done_column '%s' not found in cycle_data. Using default window.",
                    done_column,
                )
                return 30 if freq == "D" else (4 if freq == "W" else 1)

            completion_data = cycle_data[[done_column]].dropna()
            if len(completion_data) == 0:
                logger.warning("No completion data available. Using default window.")
                return 30 if freq == "D" else (4 if freq == "W" else 1)

            # Use the current date as end_date for throughput analysis
            end_date = datetime.now()

            # Calculate optimal window size based on historical patterns
            window_size = self._calculate_adaptive_window_size(
                completion_data[done_column], end_date, freq
            )

            return window_size

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error(
                "Error determining optimal window from data: %s. Using default.", e
            )
            return 30 if freq == "D" else (4 if freq == "W" else 1)

    def calculate_window_start(self, params: Dict[str, Any]) -> Optional[datetime]:
        """Calculate the throughput window start date."""
        try:
            cycle_data = params.get("cycle_data")
            done_column = params.get("done_column")
            sampling_window_end = params.get("sampling_window_end")
            smart_window = params.get("smart_window", True)
            freq = params.get("freq", "D")

            if cycle_data is None or done_column is None or sampling_window_end is None:
                return None

            if smart_window:
                # Smart window logic - find optimal start date
                return self._find_optimal_window_start(
                    cycle_data, done_column, sampling_window_end, freq
                )

            # Fixed window logic
            window_size = params.get("window_size", 30)
            return self._calculate_periods_back(sampling_window_end, window_size, freq)

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error calculating window start: %s", e)
            return None

    def _find_optimal_window_start(
        self,
        cycle_data: pd.DataFrame,
        done_column: str,
        end_date: datetime,
        freq: str,
    ) -> datetime:
        """Find optimal window start date using smart logic.

        Analyzes historical throughput patterns to determine an adaptive window size
        that balances data quality, statistical validity, and recency of data.
        """
        try:
            # Get completion dates from the data
            if done_column not in cycle_data.columns:
                logger.warning(
                    "done_column '%s' not found in cycle_data. Using default window.",
                    done_column,
                )
                return self._calculate_periods_back(end_date, 30, freq)

            completion_data = cycle_data[[done_column]].dropna()
            if len(completion_data) == 0:
                logger.warning("No completion data available. Using default window.")
                return self._calculate_periods_back(end_date, 30, freq)

            # Filter to only completed items up to end_date
            completion_data = completion_data[completion_data[done_column] <= end_date]
            if len(completion_data) == 0:
                logger.warning(
                    "No historical data up to end_date. Using default window."
                )
                return self._calculate_periods_back(end_date, 30, freq)

            # Calculate throughput statistics
            window_size = self._calculate_adaptive_window_size(
                completion_data[done_column], end_date, freq
            )

            return self._calculate_periods_back(end_date, window_size, freq)

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error(
                "Error in smart window calculation: %s. Using default window.", e
            )
            return self._calculate_periods_back(end_date, 30, freq)

    def _calculate_adaptive_window_size(
        self, completion_dates: pd.Series, end_date: datetime, freq: str
    ) -> int:
        """Calculate optimal adaptive window size based on completion data.

        Args:
            completion_dates: Series of completion timestamps
            end_date: End date for the sampling window
            freq: Frequency string ('D', 'W', 'M')

        Returns:
            Optimal window size in periods
        """
        try:
            # Define constraints
            min_window = ADAPTIVE_MIN_WINDOW_DAYS
            max_window = ADAPTIVE_MAX_WINDOW_DAYS
            target_completions = ADAPTIVE_TARGET_COMPLETIONS

            # Calculate minimum window needed to get target completions
            if len(completion_dates) >= target_completions:
                # Find how far back we need to go for target_completions items
                sorted_dates = completion_dates.sort_values(ascending=False)
                if len(sorted_dates) >= target_completions:
                    nth_oldest_completion = sorted_dates.iloc[target_completions - 1]
                    days_diff = (end_date - nth_oldest_completion).days
                    data_driven_window = max(min_window, min(days_diff, max_window))
                else:
                    # Not enough completions, use max window
                    data_driven_window = max_window
            else:
                # Very few completions, use minimum window
                data_driven_window = min_window

            # Convert to period-based window based on frequency
            if freq == "D":
                return max(min_window, min(data_driven_window, max_window))
            if freq == "W":
                # Convert days to weeks
                weeks = max(
                    min_window // 7, min(data_driven_window // 7, max_window // 7)
                )
                return max(2, weeks)  # At least 2 weeks
            if freq == "M":
                # Convert days to months
                months = max(1, min(data_driven_window // 30, 3))
                return months
            return max(min_window, min(data_driven_window, max_window))

        except (ValueError, TypeError, AttributeError) as e:
            logger.error(
                "Error calculating adaptive window size: %s. Using default.", e
            )
            return 30

    def _calculate_periods_back(
        self, end_date: datetime, periods: int, freq: str
    ) -> datetime:
        """Calculate date that is N periods back from end_date."""
        try:
            if freq == "D":
                return end_date - pd.Timedelta(days=periods)
            if freq == "W":
                return end_date - pd.Timedelta(weeks=periods)
            if freq == "M":
                return end_date - pd.DateOffset(months=periods)
            return end_date - pd.Timedelta(days=periods)

        except (ValueError, TypeError, AttributeError) as e:
            logger.error("Error calculating periods back: %s", e)
            return end_date

    def create_throughput_sampler(
        self, throughput_data: pd.DataFrame, sample_buffer_size: int = 100
    ) -> callable:
        """Return a throughput sampler callable.

        This is a thin re-export of `jira_agile_metrics.utils.create_throughput_sampler`
        to keep throughput-related helpers accessible from this calculator for API
        cohesion. Prefer importing from `jira_agile_metrics.utils` in new code if you
        do not depend on the calculator interface.

        Args:
            throughput_data: DataFrame containing a `throughput` column used for
                sampling.
            sample_buffer_size: Optional buffer size used by the sampler for efficient
                random sampling.

        Returns:
            Callable with zero arguments that returns a sampled throughput value
                per call.
        """
        return create_throughput_sampler(throughput_data, sample_buffer_size)
