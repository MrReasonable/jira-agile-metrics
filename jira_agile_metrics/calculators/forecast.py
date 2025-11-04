"""Forecast calculation and burnup chart generation for Jira Agile Metrics."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd

from ..calculator import Calculator
from .burnup import BurnupCalculator
from .burnup_chart_generator import BurnupChartGenerator
from .cycletime import CycleTimeCalculator
from .forecast_validator import ForecastDataValidator
from .monte_carlo_simulator import MonteCarloSimulator
from .throughput_calculator import ThroughputCalculator

logger = logging.getLogger(__name__)


@dataclass
class WindowStartParams:
    """Parameters for calculating window start date."""

    cycle_data: pd.DataFrame
    done_column: str
    sampling_window_end: datetime
    smart_window: bool
    freq: str
    freq_label: str
    window_size: Optional[int] = None

    def __str__(self):
        """Return string representation."""
        return f"WindowStartParams(freq={self.freq}, freq_label={self.freq_label})"


class ForecastResults(TypedDict, total=False):
    """Typed dictionary for forecast results.

    Contains the results of a Monte Carlo forecast simulation including
    trust metrics, trial data, and completion information.
    """

    trust_metrics: Dict[str, Any]
    backlog_trials: List[List[float]]
    done_trials: List[List[float]]
    forecast_horizon_end: Optional[datetime]
    target: int


class BurnupForecastCalculator(Calculator):
    """Draw a burn-up chart with a forecast run to completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store forecast results in a single dictionary to reduce instance attributes
        self._forecast_results: ForecastResults = {
            "trust_metrics": None,
            "backlog_trials": None,
            "done_trials": None,
            "forecast_horizon_end": None,
            "target": None,
        }

        # Initialize service dependencies
        self._validator = ForecastDataValidator()
        self._throughput_calculator = ThroughputCalculator()
        self._monte_carlo_simulator = MonteCarloSimulator()
        # Set number of trials from settings (default 1000 for production)
        self._monte_carlo_simulator.trials = self.settings.get(
            "burnup_forecast_chart_trials", 1000
        )

    def get_forecast_data(self):
        """Get forecast data for external use."""
        return {
            "trust_metrics": self._forecast_results["trust_metrics"],
            "backlog_trials": self._forecast_results["backlog_trials"],
            "done_trials": self._forecast_results["done_trials"],
        }

    def get_forecast_parameters(self):
        """Get forecast parameters for external use."""
        return {
            "forecast_horizon_end": self._forecast_results["forecast_horizon_end"],
            "target": self._forecast_results["target"],
        }

    def run(self):
        """Run the burnup forecast calculation."""
        try:
            # Get input data
            burnup_data = self.get_result(BurnupCalculator)
            cycle_data = self.get_result(CycleTimeCalculator)

            # Validate prerequisites and data
            validation_result = self._validate_and_prepare_data(burnup_data, cycle_data)
            if validation_result is None:
                return None

            (
                _backlog_column,
                _done_column,
                forecast_params,
                _throughput_data,  # Used indirectly via sim_params
                sim_params,
            ) = validation_result

            # Store target and forecast_horizon_end before running simulation
            # (in case simulation fails)
            # Ensure we always have a target value (default to 0 if missing)
            self._forecast_results["target"] = (
                sim_params.get("target", 0) if sim_params else 0
            )
            # forecast_params should always have forecast_horizon_end from validator,
            # but handle case where it might be missing (e.g., in tests)
            if forecast_params:
                self._forecast_results["forecast_horizon_end"] = forecast_params.get(
                    "forecast_horizon_end"
                )
            else:
                self._forecast_results["forecast_horizon_end"] = None

            # Run Monte Carlo simulation
            simulation_result = self._monte_carlo_simulator.run_simulation(sim_params)

            # Store results
            self._forecast_results["trust_metrics"] = simulation_result.get(
                "trust_metrics", {}
            )
            # Extract backlog_trial and done_trial arrays from each trial dictionary
            trials = simulation_result.get("trials", [])
            self._forecast_results["backlog_trials"] = [
                trial.get("backlog_trial", []) for trial in trials
            ]
            self._forecast_results["done_trials"] = [
                trial.get("done_trial", []) for trial in trials
            ]

            # Convert trials to DataFrame for return
            trials = simulation_result.get("trials", [])
            if not trials:
                return None

            # Convert trials list to DataFrame with trial columns
            trials_df = self._convert_trials_to_dataframe(trials)
            # Return None if conversion failed (empty DataFrame indicates failure)
            if trials_df.empty:
                return None
            return trials_df

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error running burnup forecast: %s", e)
            # Ensure target is set even if there's an error
            # (it may have been set before the error occurred)
            if self._forecast_results["target"] is None:
                self._forecast_results["target"] = 0
            return None

    def _get_last_burnup_date(self):
        """Get the last date from burnup data."""
        try:
            burnup_data = self.get_result(BurnupCalculator)
            if burnup_data is None or burnup_data.empty:
                logger.warning("No burnup data available for trials conversion")
                return None
            return burnup_data.index[-1]
        except (ValueError, IndexError, AttributeError) as e:
            logger.error("Error getting last date from burnup data: %s", e)
            return None

    def _process_single_trial(self, trial, trial_idx, last_date):
        """Process a single trial and return a Series."""
        try:
            done_values = trial.get("done_trial", [])
            if not done_values or not isinstance(done_values, list):
                logger.warning(
                    "Trial %s has no valid done_trial data, skipping", trial_idx
                )
                return None

            numeric_values = pd.to_numeric(done_values, errors="coerce")
            numeric_series = pd.Series(numeric_values)
            if numeric_series.isna().all():
                logger.warning(
                    "Trial %s values could not be converted to numeric", trial_idx
                )
                return None

            num_points = len(done_values)
            forecast_dates = pd.date_range(
                start=last_date, periods=num_points, freq="D"
            )
            return pd.Series(
                numeric_values, index=forecast_dates, name=f"Trial {trial_idx}"
            )

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.warning("Error processing trial %s: %s", trial_idx, e)
            return None

    def _convert_series_list_to_dataframe(self, series_list):
        """Convert list of series to DataFrame."""
        if not series_list:
            logger.warning(
                "No valid trial series could be created from trials data. "
                "Trials must contain 'done_trial' field with time series values."
            )
            return pd.DataFrame()

        try:
            data_dict = {f"Trial {i}": series for i, series in enumerate(series_list)}
            df = pd.DataFrame(data_dict)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="first")]
            df = df.ffill().bfill()
            return df

        except (ValueError, TypeError, AttributeError) as e:
            logger.error("Error concatenating trial series: %s", e)
            return pd.DataFrame()

    def _convert_trials_to_dataframe(
        self, trials: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Convert trials list to DataFrame with trial columns."""
        if not trials:
            return pd.DataFrame()

        forecast_horizon_end = self._forecast_results.get("forecast_horizon_end")
        if forecast_horizon_end is None:
            logger.warning("No forecast horizon available for trials conversion")
            return pd.DataFrame()

        last_date = self._get_last_burnup_date()
        if last_date is None:
            logger.warning("No last date available for trials conversion")
            return pd.DataFrame()

        series_list = [
            series
            for i, trial in enumerate(trials)
            if (series := self._process_single_trial(trial, i, last_date)) is not None
        ]

        return self._convert_series_list_to_dataframe(series_list)

    def _validate_and_prepare_data(self, burnup_data, cycle_data):
        """Validate prerequisites and prepare all data needed for forecast."""
        try:
            # Validate prerequisites
            if not self._validate_prerequisites(cycle_data):
                return None

            # Validate data and get column names
            validation_result = self._validate_data_and_get_columns(
                burnup_data, cycle_data
            )
            if validation_result is None:
                return None

            backlog_column, done_column = validation_result

            # Setup forecast parameters and calculate throughput
            forecast_throughput_result = self._setup_forecast_params_and_throughput(
                burnup_data, cycle_data, done_column
            )
            if forecast_throughput_result is None:
                return None

            forecast_params, throughput_data = forecast_throughput_result

            # Setup simulation parameters
            sim_params = self._setup_simulation_parameters(
                burnup_data, backlog_column, throughput_data, forecast_params
            )
            if sim_params is None:
                return None

            return (
                backlog_column,
                done_column,
                forecast_params,
                throughput_data,
                sim_params,
            )

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error validating and preparing data: %s", e)
            return None

    def _validate_prerequisites(self, cycle_data):
        """Validate prerequisites for forecast calculation."""
        if not self._validator.validate_run_prerequisites(cycle_data):
            return False

        if self.settings.get("burnup_forecast_chart") is None:
            logger.debug(
                "Not calculating burnup forecast chart data as no output file specified"
            )
            return False

        return True

    def _validate_data_and_get_columns(self, burnup_data, cycle_data):
        """Validate data and get column names."""
        return self._validator.validate_data(burnup_data, cycle_data)

    def _setup_forecast_parameters(
        self, burnup_data, horizon_months: int = 6, freq: str = "D"
    ):
        """Setup forecast parameters."""
        # Get frequency from config if available
        throughput_frequency = self.settings.get(
            "burnup_forecast_chart_throughput_frequency", None
        )
        if throughput_frequency:
            # Convert frequency string to pandas frequency
            freq = self._convert_frequency_string(throughput_frequency)

        forecast_params = self._validator.setup_forecast_parameters(
            burnup_data, horizon_months=horizon_months, freq=freq
        )

        if forecast_params:
            # Add throughput window and smart_window settings
            forecast_params["throughput_window"] = self.settings.get(
                "burnup_forecast_chart_throughput_window", None
            )
            forecast_params["throughput_window_end"] = self.settings.get(
                "burnup_forecast_chart_throughput_window_end", None
            )
            forecast_params["smart_window"] = self.settings.get(
                "burnup_forecast_chart_smart_window", True
            )
            forecast_params["backlog_growth_window"] = self.settings.get(
                "burnup_forecast_chart_backlog_growth_window", None
            )

        return forecast_params

    def _convert_frequency_string(self, freq_str: str) -> str:
        """Convert frequency string from config to pandas frequency code.

        Args:
            freq_str: Frequency string like 'daily', 'weekly', 'monthly', 'D', 'W', 'ME'

        Returns:
            Pandas frequency code ('D', 'W', 'ME')
        """
        freq_str_lower = freq_str.lower().strip()

        # Map common strings to pandas frequencies
        if freq_str_lower in ("daily", "d", "day"):
            return "D"
        if freq_str_lower in ("weekly", "w", "week"):
            return "W"
        if freq_str_lower in ("monthly", "m", "month"):
            return "ME"  # Use 'ME' (Month End) instead of deprecated 'M'
        # Return as-is and let pandas validate
        return freq_str

    def _calculate_throughput_data(self, cycle_data, done_column, forecast_params):
        """Calculate throughput data."""
        return self._throughput_calculator.calculate_throughput(
            cycle_data, done_column, forecast_params
        )

    def _setup_forecast_params_and_throughput(
        self, burnup_data, cycle_data, done_column
    ):
        """Setup forecast parameters and calculate throughput data."""
        forecast_params = self._setup_forecast_parameters(burnup_data)
        if forecast_params is None:
            return None

        throughput_data = self._calculate_throughput_data(
            cycle_data, done_column, forecast_params
        )
        if throughput_data is None:
            return None

        return (forecast_params, throughput_data)

    def _setup_simulation_parameters(
        self,
        burnup_data: pd.DataFrame,
        backlog_column: str,
        throughput_data: pd.DataFrame,
        forecast_params: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Setup parameters for Monte Carlo simulation."""
        try:
            # Get initial values from burnup data
            last_date = burnup_data.index[-1]
            initial_backlog = burnup_data[backlog_column].iloc[-1]
            initial_done = burnup_data[self.settings["done_column"]].iloc[-1]

            # Generate forecast dates
            # Normalize frequency: convert deprecated 'M' to 'ME' (Month End)
            freq = forecast_params["freq"]
            normalized_freq = freq if freq != "M" else "ME"
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                end=forecast_params["forecast_horizon_end"],
                freq=normalized_freq,
            ).tolist()

            # Setup samplers
            throughput_sampler_func = (
                self._throughput_calculator.create_throughput_sampler(throughput_data)
            )
            backlog_growth_sampler_func = (
                self._monte_carlo_simulator.setup_backlog_growth_sampler(
                    burnup_data, backlog_column, forecast_params
                )
            )

            # Calculate target - ensure it's greater than initial_done
            target = self.settings.get("burnup_forecast_chart_target", None)
            if target is None:
                # Default to backlog + done, but ensure it exceeds current done
                target = initial_backlog + initial_done
            elif target <= initial_done:
                # If target is set but already reached, use backlog + done
                logger.warning(
                    "Target %d is already reached (current done: %d). "
                    "Using backlog + done (%d) as target instead.",
                    target,
                    initial_done,
                    initial_backlog + initial_done,
                )
                target = initial_backlog + initial_done

            return {
                "trials": self.settings.get("burnup_forecast_chart_trials", 1000),
                "forecast_dates": forecast_dates,
                "throughput_sampler": throughput_sampler_func,
                "backlog_growth_sampler": backlog_growth_sampler_func,
                "initial_backlog": initial_backlog,
                "initial_done": initial_done,
                "target": target,
            }

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error setting up simulation parameters: %s", e)
            return None

    def write(self):
        """Write the burnup forecast chart to file."""
        try:
            if not self.settings.get("burnup_forecast_chart"):
                return

            # Get burnup data for chart generation
            burnup_data = self.get_result(BurnupCalculator)
            if burnup_data is None or burnup_data.empty:
                logger.warning("No burnup data available for chart generation")
                return

            # Check if forecast results are available
            backlog_trials = self._forecast_results.get("backlog_trials")
            done_trials = self._forecast_results.get("done_trials")
            if not backlog_trials and not done_trials:
                logger.warning(
                    "No forecast trial data available. "
                    "Chart will show historical data only."
                )

            # Prepare chart data
            chart_data = {
                "forecast_dates": self._get_forecast_dates(),
                "backlog_trials": backlog_trials or [],
                "done_trials": done_trials or [],
                "trust_metrics": self._forecast_results.get("trust_metrics", {}),
                "target": self._forecast_results.get("target", 0),
                "quantile_data": self._calculate_quantile_data(),
            }

            # Generate chart
            chart_generator = BurnupChartGenerator(
                self.settings["burnup_forecast_chart"]
            )
            success = chart_generator.generate_chart(burnup_data, chart_data)

            if success:
                logger.info(
                    "Burnup forecast chart saved to %s",
                    self.settings["burnup_forecast_chart"],
                )
            else:
                logger.error("Failed to generate burnup forecast chart")

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error writing burnup forecast chart: %s", e)

    def _get_forecast_dates(self) -> list:
        """Get forecast dates for chart generation."""
        try:
            if self._forecast_results["forecast_horizon_end"] is None:
                return []

            # Get last date from burnup data
            burnup_data = self.get_result(BurnupCalculator)
            if burnup_data is None or burnup_data.empty:
                return []

            last_date = burnup_data.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                end=self._forecast_results["forecast_horizon_end"],
                freq="D",
            ).tolist()

            return forecast_dates

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error getting forecast dates: %s", e)
            return []

    def _find_completion_indices(self, done_trials: list, target: int) -> list:
        """Find completion indices when done_trial exceeds target."""
        completion_indices = []
        for done_trial in done_trials:
            if not isinstance(done_trial, list) or len(done_trial) < 2:
                continue
            # Skip index 0 (initial state) and find where we reach the target
            for idx in range(1, len(done_trial)):
                if done_trial[idx] >= target:
                    # idx-1 corresponds to forecast_dates index
                    completion_indices.append(idx - 1)
                    break
        return completion_indices

    def _extrapolate_date(self, idx: int, forecast_dates: list) -> datetime:
        """Extrapolate a date beyond the forecast_dates range.

        Args:
            idx: Index position that is beyond the forecast_dates range.
            forecast_dates: List of forecast dates to extrapolate from.

        Returns:
            Extrapolated datetime based on the interval between forecast dates.

        Raises:
            ValueError: If forecast_dates is empty or idx is negative.
        """
        # Validate forecast_dates is not empty
        if not forecast_dates:
            raise ValueError(
                "Cannot extrapolate date: forecast_dates is empty. "
                "At least one forecast date is required to calculate the interval."
            )

        # Validate idx is non-negative
        if not isinstance(idx, int) or idx < 0:
            raise ValueError(
                f"Invalid index value: {idx}. " "Index must be a non-negative integer."
            )

        last_date = forecast_dates[-1]
        interval = (
            forecast_dates[1] - forecast_dates[0]
            if len(forecast_dates) > 1
            else pd.Timedelta(days=1)
        )
        steps_beyond = idx - len(forecast_dates) + 1
        return last_date + steps_beyond * interval

    def _convert_indices_to_dates(
        self, completion_indices: list, forecast_dates: list
    ) -> list:
        """Convert completion indices to actual dates.

        Args:
            completion_indices: List of indices where completion occurred.
            forecast_dates: List of forecast dates to map indices to.

        Returns:
            List of completion dates. Returns an empty list if forecast_dates
            is empty or falsy.
        """
        if not forecast_dates:
            return []

        completion_dates = []
        for idx in completion_indices:
            if idx < len(forecast_dates):
                completion_dates.append(forecast_dates[idx])
            else:
                completion_dates.append(self._extrapolate_date(idx, forecast_dates))
        return completion_dates

    def _calculate_quantile_data(self) -> Dict[str, Any]:
        """Calculate quantile data for completion dates."""
        try:
            done_trials = self._forecast_results.get("done_trials", [])
            target = self._forecast_results.get("target", 0)
            forecast_dates = self._get_forecast_dates()

            if not done_trials or not target or not forecast_dates:
                return {}

            completion_indices = self._find_completion_indices(done_trials, target)
            if not completion_indices:
                return {}

            completion_dates = self._convert_indices_to_dates(
                completion_indices, forecast_dates
            )

            if not completion_dates:
                return {}

            # Calculate quantiles
            quantiles = [0.5, 0.75, 0.9]
            quantile_values = pd.Series(completion_dates).quantile(quantiles)

            return {
                "50%": quantile_values.get(0.5),
                "75%": quantile_values.get(0.75),
                "90%": quantile_values.get(0.9),
            }

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error calculating quantile data: %s", e)
            return {}
