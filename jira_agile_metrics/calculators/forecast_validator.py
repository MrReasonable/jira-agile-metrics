"""Data validation service for forecast calculations."""

import logging
from typing import Optional, Tuple

import pandas as pd

from ..utils import find_backlog_and_done_columns

logger = logging.getLogger(__name__)


class ForecastDataValidator:
    """Handles validation of forecast input data and parameters."""

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

    def setup_forecast_parameters(self, burnup_data: pd.DataFrame) -> Optional[dict]:
        """Setup forecast parameters including dates and frequency settings."""
        if burnup_data is None or burnup_data.empty:
            return None

        # Get the last date from burnup data
        last_date = burnup_data.index[-1]

        # Calculate forecast horizon (default to 6 months)
        forecast_horizon_end = last_date + pd.DateOffset(months=6)

        # Determine frequency based on data
        freq = "D"  # Daily frequency
        freq_label = "day"

        return {
            "forecast_horizon_end": forecast_horizon_end,
            "freq": freq,
            "freq_label": freq_label,
            "last_date": last_date,
        }
