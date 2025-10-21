"""Forecast calculation and burnup chart generation for Jira Agile Metrics."""

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

    def run(self):
        burnup_data = self.get_result(BurnupCalculator)
        cycle_data = self.get_result(CycleTimeCalculator)

        if len(cycle_data.index) == 0:
            return None

        # This calculation is expensive.
        # Only run it if we intend to write a file.
        if not self.settings["burnup_forecast_chart"]:
            logger.debug(("Not calculating burnup forecast chart data as no output file specified"))
            return None

        backlog_column = self.settings["backlog_column"]
        done_column = self.settings["done_column"]

        if backlog_column not in burnup_data.columns:
            logger.error("Backlog column %s does not exist", backlog_column)
            return None
        if done_column not in burnup_data.columns:
            logger.error("Backlog column %s does not exist", done_column)
            return None

        if cycle_data[done_column].max() is pd.NaT:
            logger.warning(("Unable to draw burnup forecast chart with zero completed items."))
            return None

        # Determine the last date in the data
        last_data_date = burnup_data.index.max().date()
        # Use the configured window end, but not beyond the last data date for
        # sampling
        configured_window_end = self.settings["burnup_forecast_chart_throughput_window_end"]
        if configured_window_end:
            configured_window_end = pd.to_datetime(configured_window_end).date()
            sampling_window_end = min(configured_window_end, last_data_date)
        else:
            sampling_window_end = last_data_date
        # Determine forecast horizon (end date)
        forecast_horizon_end = self.settings["burnup_forecast_chart_throughput_window_end"]
        if forecast_horizon_end:
            forecast_horizon_end = pd.to_datetime(forecast_horizon_end).date()
        else:
            forecast_horizon_end = last_data_date

        throughput_window = self.settings["burnup_forecast_chart_throughput_window"]
        throughput_window_start = sampling_window_end - datetime.timedelta(days=throughput_window)
        logger.info(
            "Sampling throughput between %s and %s",
            throughput_window_start.isoformat(),
            sampling_window_end.isoformat(),
        )

        # Backlog growth window (optional, defaults to throughput window)
        backlog_growth_window = self.settings.get(
            "burnup_forecast_chart_backlog_growth_window", throughput_window
        )
        backlog_growth_window_end = sampling_window_end
        backlog_growth_window_start = backlog_growth_window_end - datetime.timedelta(
            days=backlog_growth_window
        )

        # Determine if a target number is set
        target = self.settings.get("burnup_forecast_chart_target")
        if target is not None:
            target = int(target)
        else:
            # If no target is set, use the current backlog as the target
            target = int(burnup_data[backlog_column].iloc[-1])
            # Optionally, log this behavior
            logger.info(f"No target specified; using current backlog ({target}) as target.")

        # Set up simulation parameters
        start_value = burnup_data[done_column].max()
        start_backlog = burnup_data[backlog_column].max()
        trials = self.settings["burnup_forecast_chart_trials"]

        # Try daily, then weekly, then monthly throughput until we get nonzero
        # mean
        for freq_label, freq in [
            ("daily", "D"),
            ("weekly", "W-MON"),
            ("monthly", "M"),
        ]:
            td = calculate_throughput(
                cycle_data,
                freq,
                window=(sampling_window_end - throughput_window_start).days,
            )
            mean_throughput = td["count"].mean() if len(td) > 0 else 0
            if mean_throughput > 0:
                throughput_data = td
                break
        else:
            print(
                "[ERROR] No completed items in the throughput window at any frequency. "
                "Cannot run forecast.\n"
                "Try increasing the 'Burnup forecast chart throughput window' in your config."
            )
            return None

        backlog_growth_data = calculate_daily_backlog_growth(
            burnup_data,
            backlog_column,
            backlog_growth_window_start,
            backlog_growth_window_end,
        )
        # Debug: print last 60 days of backlog growth data
        # print("[DEBUG] Backlog growth data (last 60 days):\n", backlog_growth_data.tail(60))
        # print("[DEBUG] Backlog growth data index (last 5):\n", backlog_growth_data.index[-5:])

        # Debug: print last 60 days of backlog column from burnup_data
        # print(
        #     "[DEBUG] burnup_data[backlog_column] (last 60 days):\n",
        #     burnup_data[backlog_column].tail(60),
        # )
        # print(
        #     "[DEBUG] burnup_data[backlog_column] index (last 5):\n",
        #     burnup_data[backlog_column].index[-5:],
        # )

        # --- Throughput sampling window ---
        # print(f"[DEBUG] Throughput window: {throughput_window_start} to {sampling_window_end}")
        # print(f"[DEBUG] Throughput samples found: {len(throughput_data)}")
        # print(f"[DEBUG] Throughput samples: {throughput_data}")
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
            backlog_growth_sampler_fn = backlog_growth_sampler(backlog_growth_data)

        mean_throughput = throughput_data["count"].mean()
        std_throughput = throughput_data["count"].std()
        throughput_sample_count = len(throughput_data)
        if mean_throughput == 0:
            print(
                "[ERROR] No completed items in the throughput window. Cannot run forecast.\n"
                "Try increasing the 'Burnup forecast chart throughput window' in your config."
            )
            return None

        # Compute sample buffer size for throughput sampler
        if target is not None:
            sample_buffer_size = int(2 * (target - start_value) / mean_throughput)
            if sample_buffer_size < 10:
                sample_buffer_size = 10
        else:
            sample_buffer_size = max(
                2 * (forecast_horizon_end - burnup_data.index.max().date()).days, 100
            )

        # Calculate number of days to simulate
        sim_days = (forecast_horizon_end - burnup_data.index.max().date()).days
        if sim_days < 1:
            logger.warning("Forecast horizon is not after last data point; nothing to simulate.")
            return None

        # Monte Carlo simulation: always run for sim_days
        mc_trials, backlog_trials = burnup_monte_carlo_horizon(
            start_value=start_value,
            start_backlog=start_backlog,
            start_date=burnup_data.index.max(),
            days=sim_days,
            frequency=throughput_data.index.freq,
            draw_sample=throughput_sampler(
                throughput_data, start_value, start_backlog, sample_buffer_size
            ),
            draw_backlog_growth=backlog_growth_sampler_fn,
            trials=trials,
            target=target,
        )
        # Trustworthiness metrics
        if mc_trials is not None and len(mc_trials) > 0:
            forecast_std = mc_trials.iloc[-1].std()
            forecast_var = mc_trials.iloc[-1].var()
        else:
            forecast_std = float("nan")
            forecast_var = float("nan")
        # Traffic light logic
        rel_std = std_throughput / mean_throughput if mean_throughput else float("inf")
        if throughput_sample_count >= 30 and rel_std < 0.5:
            trust_level = "green"
        elif throughput_sample_count >= 10 and rel_std < 1.0:
            trust_level = "yellow"
        else:
            trust_level = "red"
        self._trust_metrics = {
            "throughput_sample_count": throughput_sample_count,
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
                logger.warning("Cannot draw burnup forecast chart with zero items")
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
            backlog_quantiles = backlog_trials.quantile([0.1, 0.5, 0.9], axis=1).transpose()
            backlog_quantiles = backlog_quantiles.reindex(forecast_dates)
            backlog_quantiles = backlog_quantiles.interpolate(method="index").ffill()
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
            done_quantiles = mc_trials.quantile([0.1, 0.5, 0.9], axis=1).transpose()
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
                    vline = ax.axvline(value, linestyle="--", color="#444444", linewidth=1)
                    date_str = pd.to_datetime(value).strftime(self.settings["date_format"])
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
        filtered = [(h, label) for h, label in zip(handles, labels) if label in keep_labels]
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
        if trust_metrics:
            trust_text = (
                f"Throughput samples: {trust_metrics['throughput_sample_count']}\n"
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


def calculate_daily_throughput(cycle_data, done_column, window_start, window_end):
    return (
        cycle_data[[done_column, "key"]]
        .rename(columns={"key": "count", done_column: "completed_timestamp"})
        .groupby("completed_timestamp")
        .count()
        .resample("1D")
        .sum()
        .reindex(index=pd.date_range(start=window_start, end=window_end, freq="D"))
        .fillna(0)
    )


def throughput_sampler(throughput_data, start_value, target, sample_buffer_size=100):
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


def calculate_daily_backlog_growth(burnup_data, backlog_column, window_start, window_end):
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
    sample_buffer = dict(idx=0, buffer=None)

    def get_backlog_growth_sample():
        if sample_buffer["buffer"] is None or sample_buffer["idx"] >= len(sample_buffer["buffer"]):
            sample_buffer["buffer"] = backlog_growth_data.sample(sample_buffer_size, replace=True)
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
    series = {}
    backlog_series = {}
    for t in range(trials):
        current_date = start_date
        current_value = start_value
        current_backlog = start_backlog
        dates = [current_date]
        done_steps = [current_value]
        backlog_steps = [current_backlog]
        for i in range(days):
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
        series[f"Trial {t}"] = pd.Series(done_steps, index=dates, name=f"Trial {t}")
        backlog_series[f"Trial {t}"] = pd.Series(backlog_steps, index=dates, name=f"Trial {t}")
    return pd.DataFrame(series), pd.DataFrame(backlog_series)
