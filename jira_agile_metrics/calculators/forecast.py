"""Forecast calculation and burnup chart generation for Jira Agile Metrics."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Union

import pandas as pd

from ..calculator import Calculator
from ..utils import get_extension
from .burnup import BurnupCalculator
from .burnup_chart_generator import BurnupChartGenerator
from .cycletime import CycleTimeCalculator
from .forecast_utils import (
    convert_indices_to_dates,
    find_completion_indices,
)
from .forecast_validator import ForecastDataValidator, ForecastParameters
from .monte_carlo_simulator import MonteCarloSimulator
from .throughput_calculator import ThroughputCalculator

logger = logging.getLogger(__name__)

# Frequency mapping: common names to pandas frequency codes
# Note: 'ME' (Month End) is used instead of deprecated 'M' (pandas 2.1+)
FREQUENCY_MAP = {
    "daily": "D",
    "weekly": "W",
    "monthly": "ME",
}


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
            "freq": None,  # Store frequency for consistent date generation
        }

        # Initialize service dependencies
        self._validator = ForecastDataValidator(
            fallback_items_per_month=self.settings.get(
                "burnup_forecast_chart_fallback_items_per_month", None
            ),
            fallback_min_items_per_month=self.settings.get(
                "burnup_forecast_chart_fallback_min_items_per_month", 0.01
            ),
            fallback_max_items_per_month=self.settings.get(
                "burnup_forecast_chart_fallback_max_items_per_month", 5.0
            ),
        )
        self._throughput_calculator = ThroughputCalculator()
        # Initialize Monte Carlo simulator with configurable parameters
        # Note: random_seed is primarily for testing/debugging; in production
        # it should typically be None to ensure true randomness in simulations
        self._monte_carlo_simulator = MonteCarloSimulator(
            trials=self.settings.get("burnup_forecast_chart_trials", 1000),
            random_seed=self.settings.get("burnup_forecast_chart_random_seed", None),
            confidence=self.settings.get("burnup_forecast_chart_confidence", 0.8),
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
                self._forecast_results["freq"] = forecast_params.get("freq", "D")
            else:
                self._forecast_results["forecast_horizon_end"] = None
                self._forecast_results["freq"] = "D"

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

            # Get forecast horizon to pad trial to full length
            forecast_horizon_end = self._forecast_results.get("forecast_horizon_end")
            freq = self._forecast_results.get("freq", "D")
            # Normalize frequency: convert deprecated 'M' to 'ME' (Month End)
            normalized_freq = freq if freq != "M" else "ME"
            if forecast_horizon_end is None:
                logger.warning(
                    "No forecast horizon available for trial %s, using trial length",
                    trial_idx,
                )
                num_points = len(done_values)
                forecast_dates = pd.date_range(
                    start=last_date, periods=num_points, freq=normalized_freq
                )
            else:
                # Generate full forecast date range starting from last_date
                # The simulation produces [initial_done] + values for dates
                # after last_date. So we prepend last_date to the dates to
                # align with trial structure
                forecast_dates = pd.date_range(
                    start=last_date,
                    end=forecast_horizon_end,
                    freq=normalized_freq,
                )
                # Pad trial values to match full forecast horizon
                # If trial completes early, repeat the last value
                num_points = len(done_values)
                if num_points < len(forecast_dates):
                    # Pad with last value to reach target
                    last_value = numeric_values[-1] if len(numeric_values) > 0 else 0
                    padded_values = list(numeric_values) + [last_value] * (
                        len(forecast_dates) - num_points
                    )
                    numeric_values = padded_values
                elif num_points > len(forecast_dates):
                    # Truncate if somehow longer than forecast horizon
                    numeric_values = numeric_values[: len(forecast_dates)]

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

    def _normalize_target(
        self,
        target: Union[int, float],
        done_value: Union[int, float],
        backlog_value: Union[int, float],
        context: Optional[str] = "",
    ) -> Union[int, float]:
        """Normalize target value when it's already reached.

        If target <= done_value, reset to backlog-only semantics (backlog_value)
        and log an informational message.

        Args:
            target: The target value to normalize
            done_value: Current done value to compare against
            backlog_value: Backlog value to use as fallback
            context: Optional context string for logging

        Returns:
            Normalized target value (backlog_value if target <= done_value, else target)
        """
        if target <= done_value:
            original_target = target
            normalized_target = backlog_value
            logger.info(
                "Target %s <= current_done %s, resetting to backlog-only target %s "
                "(backlog=%s, done=%s)%s",
                original_target,
                done_value,
                normalized_target,
                backlog_value,
                done_value,
                f" ({context})" if context else "",
            )
            return normalized_target
        return target

    def _calculate_target_info(self, burnup_data):
        """Calculate target information from burnup data.

        Returns:
            Dict with 'target' and 'initial_done' keys, or None if unavailable
        """
        if burnup_data.empty:
            return None

        backlog_column, done_column = self._validator.validate_data(burnup_data, None)
        if not backlog_column or not done_column:
            return None

        # Latest historical point used as the initial state for the forecast
        current_backlog = burnup_data[backlog_column].iloc[-1]
        current_done = burnup_data[done_column].iloc[-1]
        # Calculate target (defaults to backlog only - work remaining to be completed)
        target = self.settings.get("burnup_forecast_chart_target", None)
        if target is None:
            # Default to backlog only (work remaining), not backlog + done
            target = current_backlog
        else:
            target = self._normalize_target(
                target, current_done, current_backlog, context="target info calculation"
            )

        return {"target": target, "initial_done": current_done}

    def _setup_forecast_parameters(
        self, burnup_data, horizon_months: int = 6, freq: str = "D"
    ):
        """Setup forecast parameters."""
        # Get frequency from config if available
        throughput_frequency = self.settings.get(
            "burnup_forecast_chart_throughput_frequency", None
        )
        if throughput_frequency:
            freq = self._convert_frequency_string(throughput_frequency)

        # Calculate target info for horizon estimation
        target_info = self._calculate_target_info(burnup_data)

        # Get horizon multiplier from settings (default: 1.5)
        horizon_multiplier = self.settings.get(
            "burnup_forecast_chart_horizon_multiplier", 1.5
        )

        forecast_params = self._validator.setup_forecast_parameters(
            ForecastParameters(
                burnup_data=burnup_data,
                horizon_months=horizon_months,
                freq=freq,
                target_info=target_info,
                horizon_multiplier=horizon_multiplier,
            )
        )

        if forecast_params:
            # Add throughput window and smart_window settings
            forecast_params["throughput_window"] = self.settings.get(
                "burnup_forecast_chart_throughput_window", None
            )
            throughput_window_end = self.settings.get(
                "burnup_forecast_chart_throughput_window_end", None
            )
            forecast_params["throughput_window_end"] = throughput_window_end

            # If throughput_window_end is provided, use it to limit forecast horizon
            # (This parameter actually limits how far the forecast projects)
            if throughput_window_end is not None:
                try:
                    # Convert to datetime (handles both string and datetime types)
                    forecast_horizon_end = pd.to_datetime(throughput_window_end)

                    # Ensure it's after the last historical date
                    last_date = forecast_params.get("last_date")
                    calculated_horizon = forecast_params.get("forecast_horizon_end")
                    if last_date and forecast_horizon_end > last_date:
                        # Check if calculated horizon is longer than
                        # user-specified limit
                        if (
                            calculated_horizon
                            and forecast_horizon_end < calculated_horizon
                        ):
                            logger.warning(
                                "Forecast horizon limited to %s by "
                                "throughput_window_end, "
                                "but calculated horizon suggests %s may be "
                                "needed to reach "
                                "target. Completion dates may be extrapolated "
                                "beyond the "
                                "forecast horizon.",
                                forecast_horizon_end,
                                calculated_horizon,
                            )
                        forecast_params["forecast_horizon_end"] = forecast_horizon_end
                        logger.debug(
                            "Forecast horizon limited to %s by throughput_window_end",
                            forecast_horizon_end,
                        )
                except (ValueError, TypeError) as e:
                    logger.warning(
                        "Invalid throughput_window_end value '%s': %s. "
                        "Using calculated forecast_horizon_end instead.",
                        throughput_window_end,
                        e,
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

        Accepts canonical pandas frequency codes ('D', 'W', 'ME') or common names
        ('daily', 'weekly', 'monthly'). All other values are returned unchanged for
        pandas validation.

        Args:
            freq_str: Frequency string (e.g., 'daily', 'weekly', 'monthly',
                'D', 'W', 'ME')

        Returns:
            Pandas frequency code ('D', 'W', 'ME') or the original string
            if unrecognized.

        Note:
            'ME' (Month End) is used instead of deprecated 'M'. This change was
            introduced in pandas 2.1 where 'M' was deprecated in favor of 'ME' to
            clarify that 'M' represents month-end frequency. Unknown strings are
            returned as-is and will be validated by pandas, allowing custom pandas
            frequency codes to be passed through directly.
        """
        if not freq_str:
            return freq_str

        freq_str_normalized = freq_str.lower().strip()

        # Check if it's a common name
        if freq_str_normalized in FREQUENCY_MAP:
            return FREQUENCY_MAP[freq_str_normalized]

        # If it's already a canonical pandas code, return as-is
        if freq_str in ("D", "W", "ME"):
            return freq_str

        # Unrecognized value - return as-is for pandas validation
        logger.debug("Unrecognized frequency string '%s' passed through", freq_str)
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

            # Generate forecast dates for simulation.
            # last_date is used only as the initial state: initial_done is computed
            # for last_date. The simulation steps/dates start the day after
            # last_date, hence forecast_dates are generated from last_date + 1 to
            # forecast_horizon_end.
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
                # Default to backlog only (work remaining), not backlog + done
                target = initial_backlog
            else:
                target = self._normalize_target(
                    target,
                    initial_done,
                    initial_backlog,
                    context="simulation parameters setup",
                )

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
        """Write the burnup forecast chart and data files to disk."""
        try:
            # Write data files if configured
            self._write_data_files()

            # Write chart if configured
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
            forecast_dates = self._get_forecast_dates()

            # Log diagnostic information
            if not backlog_trials and not done_trials:
                logger.warning(
                    "No forecast trial data available. "
                    "Chart will show historical data only."
                )
            elif not forecast_dates:
                logger.warning(
                    "No forecast dates available (forecast_horizon_end=%s). "
                    "Chart will show historical data only.",
                    self._forecast_results.get("forecast_horizon_end"),
                )
            else:
                logger.info(
                    "Forecast data: %d dates, %d backlog trials, %d done trials",
                    len(forecast_dates),
                    len(backlog_trials) if backlog_trials else 0,
                    len(done_trials) if done_trials else 0,
                )

            # The simulation produces trials with [initial_value] +
            # [simulated_values]. If forecast_dates includes last_date,
            # the trials have initial_value for last_date plus simulated
            # values. The chart generator expects trials to have
            # expected_length + 1 values (initial + forecast), which
            # matches our structure.

            # Prepare chart data
            chart_data = {
                "forecast_dates": forecast_dates,
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

    def _write_data_files(self):
        """Write forecast data to CSV/JSON/XLSX files if configured."""
        output_files = self.settings.get("burnup_forecast_chart_data")
        if not output_files:
            logger.debug("No output file specified for burnup forecast chart data")
            return

        # Support both single file path (string) and list of files
        if not isinstance(output_files, list):
            output_files = [output_files]

        # Get forecast trials data
        trials_df = self.get_result()
        if trials_df is None or trials_df.empty:
            logger.warning(
                "No forecast trial data available to write. "
                "Forecast may not have been calculated successfully."
            )
            return

        for output_file in output_files:
            output_extension = get_extension(output_file)
            logger.info("Writing burnup forecast data to %s", output_file)

            try:
                # Reset index to make date a column
                # reset_index() creates a column from the index (dates)
                # If index has no name, the column will be named "index"
                # We'll rename it to "Date" for clarity
                trials_df_reset = trials_df.reset_index()
                # Get the name of the first column (the date column created from index)
                date_col_name = trials_df_reset.columns[0]
                if date_col_name != "Date":
                    trials_df_reset.rename(
                        columns={date_col_name: "Date"}, inplace=True
                    )

                if output_extension == ".json":
                    # Convert to JSON with date as column
                    trials_df_reset.to_json(
                        output_file, date_format="iso", orient="records", index=False
                    )
                elif output_extension == ".xlsx":
                    # Write to Excel
                    trials_df_reset.to_excel(
                        output_file,
                        sheet_name="Forecast Trials",
                        header=True,
                        index=False,
                    )
                else:
                    # Default to CSV
                    trials_df_reset.to_csv(
                        output_file,
                        header=True,
                        index=False,
                        date_format="%Y-%m-%d",
                    )

            except (IOError, OSError, ValueError, TypeError) as e:
                logger.error(
                    "Error writing burnup forecast data to %s: %s",
                    output_file,
                    e,
                )

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
            # Use the same frequency as the simulation
            freq = self._forecast_results.get("freq", "D")
            # Normalize frequency: convert deprecated 'M' to 'ME' (Month End)
            normalized_freq = freq if freq != "M" else "ME"
            # Generate dates starting from last_date to connect to
            # historical data. The simulation dates start from
            # last_date+1, but we prepend last_date for chart
            forecast_dates = pd.date_range(
                start=last_date,
                end=self._forecast_results["forecast_horizon_end"],
                freq=normalized_freq,
            ).tolist()

            return forecast_dates

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error getting forecast dates: %s", e)
            return []

    def _calculate_quantile_data(self) -> Dict[str, Any]:
        """Calculate quantile data for completion dates."""
        try:
            done_trials = self._forecast_results.get("done_trials", [])
            target = self._forecast_results.get("target", 0)
            forecast_dates = self._get_forecast_dates()

            if not done_trials or not target or not forecast_dates:
                return {}

            completion_indices = find_completion_indices(done_trials, target)
            if not completion_indices:
                return {}

            completion_dates = convert_indices_to_dates(
                completion_indices, forecast_dates
            )

            if not completion_dates:
                return {}

            # Calculate quantiles
            quantiles = [0.5, 0.75, 0.85, 0.9, 0.99]
            quantile_values = pd.Series(completion_dates).quantile(quantiles)

            return {
                "50%": quantile_values.get(0.5),
                "75%": quantile_values.get(0.75),
                "85%": quantile_values.get(0.85),
                "90%": quantile_values.get(0.9),
                "99%": quantile_values.get(0.99),
            }

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error calculating quantile data: %s", e)
            return {}
