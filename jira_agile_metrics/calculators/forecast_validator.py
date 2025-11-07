"""Data validation service for forecast calculations."""

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from ..utils import find_backlog_and_done_columns

logger = logging.getLogger(__name__)


@dataclass
class HorizonCalculationParams:
    """Parameters for calculating forecast horizon."""

    burnup_data: pd.DataFrame
    last_date: pd.Timestamp
    target_info: Dict[str, int]
    horizon_months: int
    horizon_multiplier: float = 1.5


@dataclass
class ForecastParameters:
    """Parameters for setting up forecast."""

    burnup_data: pd.DataFrame
    horizon_months: int = 6
    freq: str = "D"
    target_info: Optional[Dict[str, int]] = None
    horizon_multiplier: float = 1.5


class ForecastDataValidator:
    """Handles validation of forecast input data and parameters."""

    def __init__(
        self,
        fallback_items_per_month: Optional[float] = None,
        fallback_min_items_per_month: float = 0.01,
        fallback_max_items_per_month: float = 5.0,
    ):
        """Initialize the forecast data validator.

        Args:
            fallback_items_per_month: Optional fallback throughput (items/month) to use
                when no historical completions are available. If None, will be computed
                dynamically as max(1.0 / months_of_history, 0.01), then clamped to
                [fallback_min_items_per_month, fallback_max_items_per_month].
                Defaults to None.
            fallback_min_items_per_month: Minimum bound for dynamic fallback throughput
                calculation. Defaults to 0.01.
            fallback_max_items_per_month: Maximum bound for dynamic fallback throughput
                calculation. Defaults to 5.0.
        """
        self._fallback_items_per_month = fallback_items_per_month
        self._fallback_min_items_per_month = fallback_min_items_per_month
        self._fallback_max_items_per_month = fallback_max_items_per_month

    def validate_run_prerequisites(self, cycle_data: pd.DataFrame) -> bool:
        """Validate prerequisites for running the forecast."""
        if cycle_data is None or cycle_data.empty:
            logger.warning("No cycle data available for forecast")
            return False

        if len(cycle_data) < 2:
            logger.warning(
                "Insufficient cycle data for forecast (need at least 2 items)"
            )
            return False

        return True

    def validate_data(
        self, burnup_data: pd.DataFrame, _cycle_data: pd.DataFrame
    ) -> Optional[Tuple[str, str]]:
        """Validate input data and required columns."""
        if burnup_data is None or burnup_data.empty:
            logger.warning("No burnup data available for forecast")
            return None

        # Check for required columns
        backlog_column, done_column = find_backlog_and_done_columns(burnup_data)

        if backlog_column is None:
            logger.warning("No backlog column found in burnup data")
            return None

        if done_column is None:
            logger.warning("No done column found in burnup data")
            return None

        return backlog_column, done_column

    def _calculate_target_based_horizon(
        self, params: HorizonCalculationParams
    ) -> pd.Timestamp:
        """Calculate forecast horizon based on target and throughput.

        Args:
            params: HorizonCalculationParams containing:
                burnup_data: Historical burnup data
                last_date: Last date in burnup data
                target_info: Dict with 'target' and 'initial_done' keys
                horizon_months: Default horizon in months (should be <= 36, enforced
                    by caller)
                horizon_multiplier: Multiplier for conservative horizon estimation
                    (default: 1.5). Must be a positive float. Used to account for
                    variability and ensure most trials can reach the target, especially
                    for percentile calculations (p90, p99).

        Returns:
            Forecast horizon end date (maximum 36 months from last_date)
        """
        # Validate horizon_multiplier is a positive float
        if (
            not isinstance(params.horizon_multiplier, (int, float))
            or params.horizon_multiplier <= 0
        ):
            raise ValueError(
                f"horizon_multiplier must be a positive float, "
                f"got {params.horizon_multiplier}"
            )
        remaining_items = (
            params.target_info["target"] - params.target_info["initial_done"]
        )
        if remaining_items <= 0:
            # Target already reached, use default horizon
            return params.last_date + pd.DateOffset(months=params.horizon_months)

        # Throughput-based estimate: calculate from historical completion rate
        # Compute months of history from burnup_data index span (at least 1)
        date_span = params.last_date - params.burnup_data.index[0]
        months_of_history = max(date_span.days / 30.44, 1.0)  # Average days per month

        # Compute completed_items from historical completed/count delta
        _backlog_column, done_column = find_backlog_and_done_columns(params.burnup_data)
        if done_column and done_column in params.burnup_data.columns:
            # Use the delta in the Done column
            completed_items = (
                params.burnup_data[done_column].iloc[-1]
                - params.burnup_data[done_column].iloc[0]
            )
        else:
            # Fallback: use length of data if only counts available
            completed_items = len(params.burnup_data)

        # Validate completed_items before computing throughput
        if completed_items <= 0:
            # Edge case: No historical completions available
            # This can occur with very new projects or when data is incomplete.
            # Using a fallback throughput to avoid division by zero, but this
            # can produce very long forecast horizons. Tune fallback_items_per_month
            # in settings to adjust for your team's expected velocity.
            if self._fallback_items_per_month is not None:
                avg_items_per_month = self._fallback_items_per_month
                fallback_source = "explicit setting"
                clamp_info = (
                    f"(bounds: [{self._fallback_min_items_per_month:.3f}, "
                    f"{self._fallback_max_items_per_month:.3f}])"
                )
            else:
                # Compute reasonable default: use 1.0 / months_of_history
                # This provides a throughput estimate that scales inversely with
                # historical period length. The value is then clamped to
                # [fallback_min_items_per_month, fallback_max_items_per_month] to
                # prevent extreme values for very short or very long histories.
                # For example:
                # - 0.1 months (3 days): ~10 items/month -> clamped to max (5.0)
                # - 1 month: 1 item/month -> unchanged
                # - 12 months: ~0.083 items/month -> clamped to min (0.01)
                dynamic_value = 1.0 / months_of_history
                avg_items_per_month = min(
                    max(dynamic_value, self._fallback_min_items_per_month),
                    self._fallback_max_items_per_month,
                )
                fallback_source = "dynamic calculation"
                clamp_info = (
                    f"clamped to [{self._fallback_min_items_per_month:.3f}, "
                    f"{self._fallback_max_items_per_month:.3f}]"
                )

            logger.warning(
                "No historical completions found (completed_items=%d, "
                "months_of_history=%.2f). Using fallback throughput of %.3f "
                "items/month (%s, %s). This may produce very long forecast "
                "horizons. Consider setting "
                "'burnup_forecast_chart_fallback_items_per_month' in your "
                "configuration to match your team's expected velocity.",
                completed_items,
                months_of_history,
                avg_items_per_month,
                fallback_source,
                clamp_info,
            )
        else:
            # Compute throughput as completed_items / months_of_history
            avg_items_per_month = completed_items / months_of_history

        # Calculate estimated_months based on throughput
        # Use a conservative multiplier (configurable via horizon_multiplier,
        # default 1.5x) to account for variability and ensure most trials can
        # reach the target, especially for percentile calculations (p90, p99)
        estimated_months = math.ceil(
            remaining_items / avg_items_per_month * params.horizon_multiplier
        )

        # Apply existing flooring/ceiling logic
        # Allow up to 36 months (3 years) instead of 24 to better handle
        # longer-term forecasts and ensure target completion
        max_months = 36
        estimated_months = max(min(estimated_months, max_months), params.horizon_months)

        logger.debug(
            "Forecast horizon calculation: remaining_items=%d, "
            "avg_items_per_month=%.2f, estimated_months=%.1f (capped at %d)",
            remaining_items,
            avg_items_per_month,
            estimated_months,
            max_months,
        )

        return params.last_date + pd.DateOffset(months=estimated_months)

    def setup_forecast_parameters(self, params: ForecastParameters) -> Optional[dict]:
        """Setup forecast parameters including dates and frequency settings.

        Args:
            params: ForecastParameters containing:
                burnup_data: Historical burnup data
                horizon_months: Default horizon in months (default: 6). Values > 36
                    are automatically normalized to 36 months to enforce the maximum
                    forecast horizon.
                freq: Frequency string ('D', 'W', 'ME')
                target_info: Optional dict with 'target' and 'initial_done' keys.
                    If provided, calculates a horizon sufficient to reach the target
                    (defaulting to 3 years max with configurable horizon multiplier).
                horizon_multiplier: Multiplier for conservative horizon estimation
                    (default: 1.5). Must be a positive float. Configurable via
                    burnup_forecast_chart_horizon_multiplier setting.

        Returns:
            Dictionary with forecast parameters or None if invalid data
        """
        # Enforce 36-month maximum by normalizing horizon_months
        # (increased from 24 to allow longer forecasts for target completion)
        horizon_months = min(params.horizon_months, 36)

        if params.burnup_data is None or params.burnup_data.empty:
            return None

        # Get the last date from burnup data
        last_date = params.burnup_data.index[-1]

        # Calculate forecast horizon
        if (
            params.target_info
            and "target" in params.target_info
            and "initial_done" in params.target_info
        ):
            horizon_params = HorizonCalculationParams(
                burnup_data=params.burnup_data,
                last_date=last_date,
                target_info=params.target_info,
                horizon_months=horizon_months,
                horizon_multiplier=params.horizon_multiplier,
            )
            forecast_horizon_end = self._calculate_target_based_horizon(horizon_params)
        else:
            # Use default horizon
            forecast_horizon_end = last_date + pd.DateOffset(months=horizon_months)

        # Determine frequency based on data
        freq_label = "day" if params.freq == "D" else params.freq

        return {
            "forecast_horizon_end": forecast_horizon_end,
            "freq": params.freq,
            "freq_label": freq_label,
            "last_date": last_date,
        }
