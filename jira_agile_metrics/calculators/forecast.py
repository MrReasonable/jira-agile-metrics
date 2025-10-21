"""Forecast calculation and burnup chart generation for Jira Agile Metrics."""

import calendar
import datetime
import logging
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from ..calculator import Calculator
from ..utils import set_chart_style
from .burnup import BurnupCalculator
from .cycletime import CycleTimeCalculator
from .throughput import calculate_throughput

logger = logging.getLogger(__name__)


class BurnupForecastCalculator(Calculator):
    """Draw a burn-up chart with a forecast run to completion, now sampling backlog growth."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trust_metrics = None
        self._backlog_trials = None
        self._done_trials = None
        self._forecast_horizon_end = None
        self._target = None

    def run(self):
        burnup_data = self.get_result(BurnupCalculator)
        cycle_data = self.get_result(CycleTimeCalculator)

        if len(cycle_data.index) == 0:
            return None

        # This calculation is expensive.
        # Only run it if we intend to write a file.
        if not self.settings["burnup_forecast_chart"]:
            logger.debug(
                (
                    "Not calculating burnup forecast chart data as no output file specified"
                )
            )
            return None

        # Validate data and get column names
        validation_result = self._validate_data(burnup_data, cycle_data)
        if validation_result is None:
            return None

        backlog_column, done_column = validation_result

        # Determine the last date in the data
        last_data_date = burnup_data.index.max().date()
        # Use the configured window end, but not beyond the last data date for
        # sampling
        configured_window_end = self.settings[
            "burnup_forecast_chart_throughput_window_end"
        ]
        if configured_window_end:
            configured_window_end = pd.to_datetime(
                configured_window_end
            ).date()
            sampling_window_end = min(configured_window_end, last_data_date)
        else:
            sampling_window_end = last_data_date
        # Determine forecast horizon (end date)
        forecast_horizon_end = self.settings[
            "burnup_forecast_chart_throughput_window_end"
        ]
        if forecast_horizon_end:
            forecast_horizon_end = pd.to_datetime(forecast_horizon_end).date()
        else:
            forecast_horizon_end = last_data_date

            # Use configurable frequency instead of trying all frequencies
        throughput_frequency = self.settings.get(
            "burnup_forecast_chart_throughput_frequency", "daily"
        )
        logger.info(
            "Throughput frequency setting: %s (type: %s)",
            throughput_frequency,
            type(throughput_frequency),
        )
        freq_mapping = {
            "daily": ("daily", "D"),
            "weekly": ("weekly", "W-MON"),
            "monthly": ("monthly", "M"),
        }

        if throughput_frequency not in freq_mapping:
            logger.warning(
                "Invalid throughput frequency '%s', using daily",
                throughput_frequency,
            )
            throughput_frequency = "daily"

        freq_label, freq = freq_mapping[throughput_frequency]
        logger.info(
            "Using throughput frequency: %s (%s)",
            throughput_frequency,
            freq,
        )

        # Smart window logic: start from first delivery if enabled
        smart_window = self.settings.get(
            "burnup_forecast_chart_smart_window", False
        )
        logger.info(
            "Smart window setting: %s (type: %s)",
            smart_window,
            type(smart_window),
        )
        if smart_window:
            # Find the first date when something was actually delivered
            first_delivery_date = cycle_data[done_column].min()
            logger.info(
                "Smart window enabled. First delivery date: %s (pd.NaT: %s)",
                first_delivery_date,
                pd.isna(first_delivery_date),
            )
            if pd.notna(first_delivery_date):
                # Smart window: start from first delivery, but respect the throughput window setting
                # Calculate the window start based on the throughput window setting and frequency
                throughput_window_periods = self.settings[
                    "burnup_forecast_chart_throughput_window"
                ]

                # Calculate window start based on frequency
                if freq == "D":  # daily
                    window_start_from_end = (
                        sampling_window_end
                        - datetime.timedelta(days=throughput_window_periods)
                    )
                elif freq == "W-MON":  # weekly
                    window_start_from_end = (
                        sampling_window_end
                        - datetime.timedelta(weeks=throughput_window_periods)
                    )
                elif freq == "M":  # monthly
                    # For monthly, we need to calculate months back
                    year = sampling_window_end.year
                    month = sampling_window_end.month
                    for _ in range(throughput_window_periods):
                        month -= 1
                        if month < 1:
                            month = 12
                            year -= 1
                    window_start_from_end = datetime.date(
                        year,
                        month,
                        min(
                            sampling_window_end.day,
                            calendar.monthrange(year, month)[1],
                        ),
                    )
                else:
                    # Fallback to daily
                    window_start_from_end = (
                        sampling_window_end
                        - datetime.timedelta(days=throughput_window_periods)
                    )

                # But don't start before the first delivery
                throughput_window_start = max(
                    first_delivery_date.date(), window_start_from_end
                )
                logger.info(
                    "Using smart window: from %s (max of first delivery %s and %s %s ago) to %s",
                    throughput_window_start.isoformat(),
                    first_delivery_date.date().isoformat(),
                    throughput_window_periods,
                    freq_label,
                    sampling_window_end.isoformat(),
                )
            else:
                # Fall back to fixed window if no deliveries
                fixed_window_days = self.settings[
                    "burnup_forecast_chart_throughput_window"
                ]
                throughput_window_start = (
                    sampling_window_end
                    - datetime.timedelta(days=fixed_window_days)
                )
                logger.info(
                    "No deliveries found, using fixed window: %s days",
                    fixed_window_days,
                )
        else:
            # Use fixed window as before
            throughput_window = self.settings[
                "burnup_forecast_chart_throughput_window"
            ]
            throughput_window_start = sampling_window_end - datetime.timedelta(
                days=throughput_window
            )
            logger.info(
                "Using fixed window: %s days from %s to %s",
                throughput_window,
                throughput_window_start.isoformat(),
                sampling_window_end.isoformat(),
            )

        # Calculate throughput using the specified frequency
        # For smart window, we need to filter the data to the window range
        # For fixed window, calculate_throughput will handle the window internally
        if smart_window and pd.notna(cycle_data[done_column].min()):
            # Filter cycle data to the smart window range
            filtered_cycle_data = cycle_data[
                (
                    cycle_data[done_column]
                    >= pd.Timestamp(throughput_window_start)
                )
                & (
                    cycle_data[done_column]
                    <= pd.Timestamp(sampling_window_end)
                )
            ]
            logger.info(
                "Smart window: filtered cycle data from %s to %s, %d items",
                throughput_window_start.isoformat(),
                sampling_window_end.isoformat(),
                len(filtered_cycle_data),
            )
            td = calculate_throughput(
                filtered_cycle_data,
                freq,
                window=None,  # Use all data in the filtered range
            )
        else:
            # Use fixed window - calculate_throughput will handle the window
            # Calculate window size in terms of the frequency being used
            if freq == "D":  # daily
                window_size = (
                    sampling_window_end - throughput_window_start
                ).days
            elif freq == "W-MON":  # weekly
                window_size = (
                    sampling_window_end - throughput_window_start
                ).days // 7
            elif freq == "M":  # monthly
                window_size = (
                    (sampling_window_end.year - throughput_window_start.year)
                    * 12
                    + sampling_window_end.month
                    - throughput_window_start.month
                )
            else:
                # Fallback to daily calculation
                window_size = (
                    sampling_window_end - throughput_window_start
                ).days

            td = calculate_throughput(
                cycle_data,
                freq,
                window=window_size,
            )
        mean_throughput = td["count"].mean() if len(td) > 0 else 0

        logger.info(
            "Throughput calculation result: %d periods, mean=%.2f, total=%d",
            len(td),
            mean_throughput,
            td["count"].sum() if len(td) > 0 else 0,
        )

        if mean_throughput > 0:
            throughput_data = td
        else:
            print(
                f"[ERROR] No completed items in the throughput window using "
                f"{freq_label} frequency. Cannot run forecast.\n"
                "Try increasing the 'Burnup forecast chart throughput window' "
                "in your config or changing the 'Burnup forecast chart "
                "throughput frequency'."
            )
            return None

        # Backlog growth window (optional, defaults to throughput window)
        backlog_growth_window = self.settings.get(
            "burnup_forecast_chart_backlog_growth_window",
            (sampling_window_end - throughput_window_start).days,
        )
        backlog_growth_window_end = sampling_window_end
        backlog_growth_window_start = (
            backlog_growth_window_end
            - datetime.timedelta(days=backlog_growth_window)
        )

        # Determine if a target number is set
        target = self.settings.get("burnup_forecast_chart_target")
        if target is not None:
            target = int(target)
        else:
            # If no target is set, use the current backlog as the target
            target = int(burnup_data[backlog_column].iloc[-1])
            # Optionally, log this behavior
            logger.info(
                "No target specified; using current backlog (%d) as target.",
                target,
            )
        start_value = burnup_data[done_column].max()
        start_backlog = burnup_data[backlog_column].max()
        trials = self.settings["burnup_forecast_chart_trials"]

        backlog_growth_data = calculate_daily_backlog_growth(
            burnup_data,
            backlog_column,
            backlog_growth_window_start,
            backlog_growth_window_end,
        )

        # --- Throughput sampling window ---
        if len(throughput_data) == 0:
            warnings.warn(
                "No throughput samples available, aborting forecast simulations",
                RuntimeWarning,
            )
            return None

        # If all backlog growth is zero, just use zeros
        if backlog_growth_data.sum() == 0:

            def backlog_growth_sampler_fn():
                return 0

        else:
            backlog_growth_sampler_fn = backlog_growth_sampler(
                backlog_growth_data
            )

        mean_throughput = throughput_data["count"].mean()
        std_throughput = throughput_data["count"].std()
        # Calculate meaningful statistics
        total_time_periods = len(throughput_data)
        actual_completed_items = throughput_data["count"].sum()
        non_zero_periods = (throughput_data["count"] > 0).sum()
        if mean_throughput == 0:
            print(
                "[ERROR] No completed items in the throughput window. Cannot run forecast.\n"
                "Try increasing the 'Burnup forecast chart throughput window' in your config."
            )
            return None

        # Compute sample buffer size for throughput sampler
        if target is not None:
            sample_buffer_size = int(
                2 * (target - start_value) / mean_throughput
            )
            if sample_buffer_size < 10:
                sample_buffer_size = 10
        else:
            sample_buffer_size = max(
                2
                * (forecast_horizon_end - burnup_data.index.max().date()).days,
                100,
            )

        # Calculate number of days to simulate
        sim_days = (forecast_horizon_end - burnup_data.index.max().date()).days
        if sim_days < 1:
            logger.warning(
                "Forecast horizon is not after last data point; nothing to simulate."
            )
            return None

        # Monte Carlo simulation: always run for sim_days
        max_iterations = self.settings.get(
            "burnup_forecast_chart_max_iterations", 9999
        )
        mc_trials, backlog_trials = burnup_monte_carlo_horizon(
            start_value=start_value,
            start_backlog=start_backlog,
            start_date=burnup_data.index.max(),
            days=sim_days,
            frequency=throughput_data.index.freq,
            draw_sample=throughput_sampler(
                throughput_data, sample_buffer_size
            ),
            draw_backlog_growth=backlog_growth_sampler_fn,
            trials=trials,
            target=target,
            max_iterations=max_iterations,
        )
        # Trustworthiness metrics
        if mc_trials is not None and len(mc_trials) > 0:
            forecast_std = mc_trials.iloc[-1].std()
            forecast_var = mc_trials.iloc[-1].var()
        else:
            forecast_std = float("nan")
            forecast_var = float("nan")
        # Traffic light logic - use actual completed items and non-zero periods
        rel_std = (
            std_throughput / mean_throughput
            if mean_throughput
            else float("inf")
        )
        if actual_completed_items >= 30 and rel_std < 0.5:
            trust_level = "green"
        elif actual_completed_items >= 10 and rel_std < 1.0:
            trust_level = "yellow"
        else:
            trust_level = "red"
        self._trust_metrics = {
            "total_time_periods": total_time_periods,
            "throughput_frequency": throughput_frequency,
            "actual_completed_items": actual_completed_items,
            "non_zero_periods": non_zero_periods,
            "throughput_mean": mean_throughput,
            "throughput_std": std_throughput,
            "forecast_std": forecast_std,
            "forecast_var": forecast_var,
            "trust_level": trust_level,
            "rel_std": rel_std,
        }
        self._backlog_trials = backlog_trials
        self._done_trials = mc_trials
        self._forecast_horizon_end = forecast_horizon_end
        self._target = target
        return mc_trials

    def _validate_data(self, burnup_data, cycle_data):
        """Validate input data and required columns."""
        backlog_column = self.settings["backlog_column"]
        done_column = self.settings["done_column"]

        # Debug: Log cycle data info
        logger.info(
            "Cycle data shape: %s, columns: %s, done_column: %s",
            cycle_data.shape,
            list(cycle_data.columns),
            done_column,
        )
        if done_column in cycle_data.columns:
            logger.info(
                "Done column stats: min=%s, max=%s, count=%s",
                cycle_data[done_column].min(),
                cycle_data[done_column].max(),
                cycle_data[done_column].count(),
            )
        else:
            logger.error(
                "Done column '%s' not found in cycle data", done_column
            )
            return None

        if backlog_column not in burnup_data.columns:
            logger.error("Backlog column %s does not exist", backlog_column)
            return None
        if done_column not in burnup_data.columns:
            logger.error("Backlog column %s does not exist", done_column)
            return None

        if pd.isna(cycle_data[done_column].max()):
            logger.warning(
                (
                    "Unable to draw burnup forecast chart with zero completed items."
                )
            )
            return None

        return backlog_column, done_column

    def write(self):
        output_file = self.settings["burnup_forecast_chart"]
        if not output_file:
            logger.debug("No output file specified for burnup forecast chart")
            return

        burnup_data = self.get_result(BurnupCalculator)
        if burnup_data is None or len(burnup_data.index) == 0:
            logger.warning("Cannot draw burnup forecast chart with zero items")
            return

        window = self.settings["burnup_forecast_window"]
        if window:
            start = burnup_data.index.max() - pd.Timedelta(window, "D")
            burnup_data = burnup_data[start:]

            if len(burnup_data.index) == 0:
                logger.warning(
                    "Cannot draw burnup forecast chart with zero items"
                )
                return

        mc_trials = self._done_trials
        backlog_trials = self._backlog_trials
        forecast_horizon_end = self._forecast_horizon_end
        target = self._target
        trust_metrics = getattr(self, "_trust_metrics", None)

        fig, ax = plt.subplots()

        if self.settings["burnup_forecast_chart_title"]:
            ax.set_title(self.settings["burnup_forecast_chart_title"])

        fig.autofmt_xdate()

        # --- Plot historical backlog and done ---
        burnup_data.plot.line(ax=ax, legend=False)

        # --- Plot forecast fans (backlog and done) beyond last historical date ---
        # Determine the full forecast date range
        forecast_dates = pd.date_range(
            start=burnup_data.index.max(), end=forecast_horizon_end, freq="D"
        )
        # Plot backlog fan
        if backlog_trials is not None:
            backlog_quantiles = backlog_trials.quantile(
                [0.1, 0.5, 0.9], axis=1
            ).transpose()
            backlog_quantiles = backlog_quantiles.reindex(forecast_dates)
            backlog_quantiles = backlog_quantiles.interpolate(
                method="index"
            ).ffill()
            ax.fill_between(
                backlog_quantiles.index,
                backlog_quantiles[0.1],
                backlog_quantiles[0.9],
                color="#e0e0e0",
                alpha=0.5,
                label="Backlog growth fan (10-90%)",
            )
            ax.fill_between(
                backlog_quantiles.index,
                backlog_quantiles[0.5],
                backlog_quantiles[0.9],
                color="#b0b0b0",
                alpha=0.3,
            )
            ax.fill_between(
                backlog_quantiles.index,
                backlog_quantiles[0.1],
                backlog_quantiles[0.5],
                color="#b0b0b0",
                alpha=0.3,
            )
        # Plot done fan
        if mc_trials is not None:
            done_quantiles = mc_trials.quantile(
                [0.1, 0.5, 0.9], axis=1
            ).transpose()
            done_quantiles = done_quantiles.reindex(forecast_dates)
            done_quantiles = done_quantiles.interpolate(method="index").ffill()
            ax.fill_between(
                done_quantiles.index,
                done_quantiles[0.1],
                done_quantiles[0.9],
                color="#c0e0ff",
                alpha=0.5,
                label="Done fan (10-90%)",
            )
            ax.fill_between(
                done_quantiles.index,
                done_quantiles[0.5],
                done_quantiles[0.9],
                color="#80bfff",
                alpha=0.3,
            )
            ax.fill_between(
                done_quantiles.index,
                done_quantiles[0.1],
                done_quantiles[0.5],
                color="#80bfff",
                alpha=0.3,
            )
        # --- Plot target line and quantile completion dates if target is set ---
        if target is not None:
            left, right = ax.get_xlim()
            ax.hlines(
                target,
                left,
                right,
                linestyles="--",
                linewidths=1,
                color="#888888",
                label=f"Target: {target}",
            )
            # Find for each trial the first date when done >= target
            finish_dates = []
            for col in mc_trials:
                reached = mc_trials[col][mc_trials[col] >= target]
                if not reached.empty:
                    finish_dates.append(reached.index[0])
            if finish_dates:
                finish_dates = pd.Series(finish_dates)
                quantiles = [0.5, 0.85, 0.95]
                finish_date_quantiles = finish_dates.quantile(quantiles)
                quantile_handles = []
                quantile_labels = []
                for q, value in zip(quantiles, finish_date_quantiles):
                    vline = ax.axvline(
                        value, linestyle="--", color="#444444", linewidth=1
                    )
                    date_str = pd.to_datetime(value).strftime(
                        self.settings["date_format"]
                    )
                    label = f"{int(q * 100)}% ({date_str})"
                    quantile_handles.append(vline)
                    quantile_labels.append(label)
        # --- Clean up legend: only show key elements ---
        handles, labels = ax.get_legend_handles_labels()
        keep_labels = set(
            [
                "Backlog",
                "Done",
                "Backlog growth fan (10-90%)",
                "Backlog median",
                "Done fan (10-90%)",
                "Done median",
            ]
        )
        if target is not None:
            keep_labels.add(f"Target: {target}")
        filtered = [
            (h, label)
            for h, label in zip(handles, labels)
            if label in keep_labels
        ]
        if filtered:
            handles, labels = zip(*filtered)
        # Add quantile completion date lines to legend
        if target is not None and "quantile_handles" in locals():
            handles = list(handles) + quantile_handles
            labels = list(labels) + quantile_labels
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=2,
        )
        set_chart_style()
        # Ensure x-axis covers the full forecast period
        ax.set_xlim([burnup_data.index.min(), forecast_horizon_end])
        # Annotate trustworthiness metrics
        if trust_metrics is not None:
            trust_text = (
                f"{trust_metrics['total_time_periods']} "
                f"{trust_metrics['throughput_frequency']} periods\n"
                f"Actual completed items: {trust_metrics['actual_completed_items']}\n"
                f"Non-zero periods: {trust_metrics['non_zero_periods']}\n"
                f"Throughput mean: {trust_metrics['throughput_mean']:.2f}\n"
                f"Throughput std: {trust_metrics['throughput_std']:.2f}\n"
                f"Forecast std: {trust_metrics['forecast_std']:.2f}\n"
                f"Trust level: {trust_metrics['trust_level'].capitalize()}\n"
            )
            color_map = {
                "green": "#b6e3b6",
                "yellow": "#fff7b2",
                "red": "#f7b2b2",
            }
            ax.annotate(
                trust_text,
                xy=(0.01, 0.99),
                xycoords="axes fraction",
                fontsize=8,
                ha="left",
                va="top",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc=color_map.get(trust_metrics["trust_level"], "white"),
                    ec="gray",
                    lw=1,
                ),
            )
        # Write file
        logger.info("Writing burnup forecast chart to %s", output_file)
        fig.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close(fig)


def calculate_daily_throughput(
    cycle_data, done_column, window_start, window_end
):
    """Calculate daily throughput from cycle data within a specified window."""
    return (
        cycle_data[[done_column, "key"]]
        .rename(columns={"key": "count", done_column: "completed_timestamp"})
        .groupby("completed_timestamp")
        .count()
        .resample("1D")
        .sum()
        .reindex(
            index=pd.date_range(start=window_start, end=window_end, freq="D")
        )
        .fillna(0)
    )


def throughput_sampler(throughput_data, sample_buffer_size=100):
    """Return a function that can efficiently
    draw samples from `throughput_data`"""
    sample_buffer = dict(idx=0, buffer=None)

    def get_throughput_sample():
        if sample_buffer["buffer"] is None or sample_buffer["idx"] >= len(
            sample_buffer["buffer"].index
        ):
            sample_buffer["buffer"] = throughput_data["count"].sample(
                sample_buffer_size, replace=True
            )
            sample_buffer["idx"] = 0

        sample_buffer["idx"] += 1
        return sample_buffer["buffer"].iloc[sample_buffer["idx"] - 1]

    return get_throughput_sample


def calculate_daily_backlog_growth(
    burnup_data, backlog_column, window_start, window_end
):
    """Calculate daily backlog growth from burnup data within a specified window."""
    # Calculate the daily change in backlog (new items added)
    backlog_series = (
        burnup_data[backlog_column]
        .reindex(pd.date_range(start=window_start, end=window_end, freq="D"))
        .ffill()
        .fillna(0)
    )
    # Daily growth is the positive difference (new items only)
    backlog_growth = backlog_series.diff().fillna(0)
    backlog_growth[backlog_growth < 0] = 0
    return backlog_growth


def backlog_growth_sampler(backlog_growth_data, sample_buffer_size=100):
    """Return a function that efficiently draws samples from backlog growth data."""
    sample_buffer = dict(idx=0, buffer=None)

    def get_backlog_growth_sample():
        if sample_buffer["buffer"] is None or sample_buffer["idx"] >= len(
            sample_buffer["buffer"]
        ):
            sample_buffer["buffer"] = backlog_growth_data.sample(
                sample_buffer_size, replace=True
            )
            sample_buffer["idx"] = 0
        sample_buffer["idx"] += 1
        return sample_buffer["buffer"].iloc[sample_buffer["idx"] - 1]

    return get_backlog_growth_sample


def burnup_monte_carlo_horizon(
    start_value,
    start_backlog,
    start_date,
    days,
    frequency,
    draw_sample,
    draw_backlog_growth=None,
    trials=100,
    target=None,
    max_iterations=9999,
):
    """Run Monte Carlo simulation for burnup forecast with specified parameters."""
    series = {}
    backlog_series = {}
    for t in range(trials):
        current_date = start_date
        current_value = start_value
        current_backlog = start_backlog
        dates = [current_date]
        done_steps = [current_value]
        backlog_steps = [current_backlog]
        for _ in range(min(days, max_iterations)):
            current_date += frequency
            current_value += draw_sample()
            if draw_backlog_growth is not None:
                current_backlog += draw_backlog_growth()
            dates.append(current_date)
            done_steps.append(current_value)
            backlog_steps.append(current_backlog)
            # If target is set and median done reaches/exceeds target, stop
            # early
            if target is not None and current_value >= target:
                break
        # Pad to full horizon if stopped early
        while len(dates) < days + 1:
            dates.append(dates[-1] + frequency)
            done_steps.append(done_steps[-1])
            backlog_steps.append(backlog_steps[-1])
        series[f"Trial {t}"] = pd.Series(
            done_steps, index=dates, name=f"Trial {t}"
        )
        backlog_series[f"Trial {t}"] = pd.Series(
            backlog_steps, index=dates, name=f"Trial {t}"
        )
    return pd.DataFrame(series), pd.DataFrame(backlog_series)
