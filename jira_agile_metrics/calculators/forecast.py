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
    backlog_trials: List[Dict[str, Any]]
    done_trials: List[Dict[str, Any]]
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

            # Run Monte Carlo simulation
            simulation_result = self._monte_carlo_simulator.run_simulation(sim_params)

            # Store results
            self._forecast_results["trust_metrics"] = simulation_result.get(
                "trust_metrics", {}
            )
            self._forecast_results["backlog_trials"] = list(
                simulation_result.get("trials", [])
            )
            self._forecast_results["done_trials"] = list(
                simulation_result.get("trials", [])
            )
            self._forecast_results["forecast_horizon_end"] = forecast_params[
                "forecast_horizon_end"
            ]
            self._forecast_results["target"] = sim_params.get("target", 0)

            # Convert trials to DataFrame for return
            trials = simulation_result.get("trials", [])
            if not trials:
                return None

            # Convert trials list to DataFrame with trial columns
            trials_df = self._convert_trials_to_dataframe(trials)
            return trials_df

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error running burnup forecast: %s", e)
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

        if self._forecast_results.get("forecast_horizon_end") is None:
            logger.warning("No forecast horizon available for trials conversion")
            return pd.DataFrame()

        last_date = self._get_last_burnup_date()
        if last_date is None:
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
        return self._validator.setup_forecast_parameters(
            burnup_data, horizon_months=horizon_months, freq=freq
        )

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
            forecast_horizon_end = forecast_params["forecast_horizon_end"]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                end=forecast_horizon_end,
                freq=forecast_params["freq"],
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

            return {
                "trials": self.settings.get("burnup_forecast_chart_trials", 1000),
                "forecast_dates": forecast_dates,
                "throughput_sampler": throughput_sampler_func,
                "backlog_growth_sampler": backlog_growth_sampler_func,
                "initial_backlog": initial_backlog,
                "initial_done": initial_done,
                "target": self.settings.get(
                    "burnup_forecast_chart_target", initial_backlog + initial_done
                ),
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

            # Prepare chart data
            chart_data = {
                "forecast_dates": self._get_forecast_dates(),
                "backlog_trials": self._forecast_results["backlog_trials"],
                "done_trials": self._forecast_results["done_trials"],
                "trust_metrics": self._forecast_results["trust_metrics"],
                "target": self._forecast_results["target"],
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

    def _calculate_quantile_data(self) -> Dict[str, Any]:
        """Calculate quantile data for completion dates."""
        try:
            val = self._forecast_results["done_trials"]
            if not isinstance(val, list) or not val:
                return {}

            # Create defensive copy to avoid mutation side-effects
            done_trials = val.copy()

            # Extract completion dates from trials
            completion_dates = []
            for trial in done_trials:
                if isinstance(trial, dict) and "completion_date" in trial:
                    completion_dates.append(trial["completion_date"])

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
