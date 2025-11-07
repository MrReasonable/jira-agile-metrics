"""Histogram calculator for Jira Agile Metrics.

This module provides functionality to calculate histogram metrics from JIRA data.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..calculator import Calculator
from ..chart_styling_utils import set_chart_style
from ..utils import get_extension
from .cycletime import CycleTimeCalculator

logger = logging.getLogger(__name__)


class HistogramCalculator(Calculator):
    """Build histogram data for the cycle times in `cycle_data`. Returns
    a dictionary with keys `bin_values` and `bin_edges` of numpy arrays
    """

    CSV_HISTOGRAM_COLUMNS = ["Range", "Items"]
    ITEMS_COLUMN_INDEX = 1

    def run(self):
        cycle_data = self.get_result(CycleTimeCalculator)

        # Check if cycle_time column exists and has timedelta data
        if "cycle_time" not in cycle_data.columns:
            return pd.Series(
                [], name=self.CSV_HISTOGRAM_COLUMNS[self.ITEMS_COLUMN_INDEX], index=[]
            )

        cycle_time_series = cycle_data["cycle_time"]

        # Check if the series is empty or not timedelta type
        if cycle_time_series.empty or not pd.api.types.is_timedelta64_dtype(
            cycle_time_series
        ):
            return pd.Series(
                [], name=self.CSV_HISTOGRAM_COLUMNS[self.ITEMS_COLUMN_INDEX], index=[]
            )

        cycle_times = cycle_time_series.dt.days.dropna().tolist()

        if not cycle_times:
            bins = range(11)
        else:
            bins = range(int(max(cycle_times)) + 2)

        values, edges = np.histogram(cycle_times, bins=bins, density=False)

        index = []
        for i, _ in enumerate(edges):
            if i == 0:
                continue
            index.append(f"{edges[i - 1]:.01f} to {edges[i]:.01f}")

        return pd.Series(
            values,
            name=self.CSV_HISTOGRAM_COLUMNS[self.ITEMS_COLUMN_INDEX],
            index=index,
        )

    def write(self):
        data = self.get_result()

        logger.debug(
            "[DEBUG] lead_time_histogram_data: %s",
            self.settings.get("lead_time_histogram_data"),
        )
        logger.debug(
            "[DEBUG] lead_time_histogram_chart: %s",
            self.settings.get("lead_time_histogram_chart"),
        )

        if self.settings["histogram_data"]:
            self.write_file(data, self.settings["histogram_data"])
        else:
            logger.debug("No output file specified for histogram data")

        if self.settings["histogram_chart"]:
            self.write_chart(data, self.settings["histogram_chart"])
        else:
            logger.debug("No output file specified for histogram chart")

        # Lead time histogram outputs
        if self.settings.get("lead_time_histogram_data"):
            self.write_lead_time_file(data, self.settings["lead_time_histogram_data"])
        else:
            logger.debug("No output file specified for lead time histogram data")

        if self.settings.get("lead_time_histogram_chart"):
            for output_file in self.settings["lead_time_histogram_chart"]:
                self.write_lead_time_chart(data, output_file)
        else:
            logger.debug("No output file specified for lead time histogram chart")

    def write_file(self, _data, output_files):
        """Write histogram data to output files."""
        file_data = self.get_result()

        for output_file in output_files:
            output_extension = get_extension(output_file)

            logger.info("Writing histogram data to %s", output_file)
            if output_extension == ".json":
                file_data.to_json(output_file, date_format="iso")
            elif output_extension == ".xlsx":
                file_data.to_frame(name="histogram").to_excel(
                    output_file, sheet_name="Histogram", header=True
                )
            else:
                # Reset index to convert it to a named column
                file_data_to_write = file_data.reset_index()
                file_data_to_write.columns = self.CSV_HISTOGRAM_COLUMNS
                file_data_to_write.to_csv(output_file, header=True, index=False)

    def write_chart(self, _data, output_file):
        """Write histogram chart to output file.

        Args:
            _data: Unused parameter (kept for interface compatibility)
            output_file: Path to output file
        """
        cycle_data = self.get_result(CycleTimeCalculator)
        chart_data = cycle_data[["cycle_time", "completed_timestamp"]].dropna(
            subset=["cycle_time"]
        )

        # Check if we have any data before trying to use .dt accessor
        if len(chart_data.index) == 0:
            logger.warning("Need at least 2 completed items to draw histogram")
            return

        # The `window` calculation and the chart output will fail if we don't
        # have at least two valid data points.
        # Check if cycle_time is a timedelta type before using .dt accessor
        if not pd.api.types.is_timedelta64_dtype(chart_data["cycle_time"]):
            logger.warning("Cycle time column must be timedelta type to draw histogram")
            return

        ct_days = chart_data["cycle_time"].dt.days
        if len(ct_days.index) < 2:
            logger.warning("Need at least 2 completed items to draw histogram")
            return

        # Apply window filtering if specified
        chart_data = self._apply_window_filter(chart_data)
        if chart_data is None:
            return

        ct_days = chart_data["cycle_time"].dt.days
        self._create_histogram_chart(ct_days, output_file)

    def _apply_window_filter(self, chart_data):
        """Apply window filtering to chart data."""
        window = self.settings["histogram_window"]
        if window:
            start = chart_data["completed_timestamp"].max().normalize() - pd.Timedelta(
                window, "D"
            )
            chart_data = chart_data[chart_data.completed_timestamp >= start]

            # Re-check that we have enough data
            ct_days = chart_data["cycle_time"].dt.days
            if len(ct_days.index) < 2:
                logger.warning("Need at least 2 completed items to draw histogram")
                return None
        return chart_data

    def _create_histogram_chart(self, ct_days, output_file):
        """Create and save histogram chart."""
        quantiles = self.settings["quantiles"]
        logger.debug(
            "Showing histogram at quantiles %s",
            ", ".join([f"{q * 100.0:.2f}" for q in quantiles]),
        )

        fig, ax = plt.subplots()
        bins = range(int(ct_days.max()) + 2)

        sns.histplot(ct_days, bins=bins, ax=ax, kde=False)
        ax.set_xlabel("Cycle time (days)")

        if self.settings["histogram_chart_title"]:
            ax.set_title(self.settings["histogram_chart_title"])

        _, right = ax.get_xlim()
        ax.set_xlim(0, right)

        # Add quantiles
        self._add_quantile_lines(ax, ct_days, quantiles, metric_name="Cycle time")
        ax.set_ylabel("Frequency")
        set_chart_style()

        # Write file
        logger.info("Writing histogram chart to %s", output_file)
        fig.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def _add_quantile_lines(self, ax, time_days, quantiles, metric_name=None):
        """Add quantile lines and labels to the chart.

        Args:
            ax: Matplotlib Axes object
            time_days: Series or array of time values in days
            quantiles: List of quantile values (e.g., [0.5, 0.85, 0.95])
            metric_name: Optional metric name prefix for labels
        """
        bottom, top = ax.get_ylim()
        quantile_labels = []
        for quantile, value in time_days.quantile(quantiles).items():
            ax.vlines(value, bottom, top - 0.001, linestyles="--", linewidths=1)
            prefix = f"{metric_name}: " if metric_name else ""
            label = f"{prefix}{quantile * 100:.0f}% ({value:.0f} days)"
            quantile_labels.append(label)

        # Add quantile labels as a block of text below the chart
        if quantile_labels:
            fig = ax.get_figure()
            fig.text(
                0.5,
                -0.03,
                " | ".join(quantile_labels),
                ha="center",
                va="top",
                fontsize="small",
            )

    def write_lead_time_file(self, _data, output_files):
        """Write lead time histogram data to output files."""
        cycle_data = self.get_result(CycleTimeCalculator)
        lead_times = cycle_data["lead_time"].dt.days.dropna().tolist()
        if not lead_times:
            bins = range(11)
        else:
            bins = range(int(max(lead_times)) + 2)
        values, edges = np.histogram(lead_times, bins=bins, density=False)
        index = []
        for i, _ in enumerate(edges):
            if i == 0:
                continue
            index.append(f"{edges[i - 1]:.01f} to {edges[i]:.01f}")
        file_data = pd.Series(
            values,
            name=self.CSV_HISTOGRAM_COLUMNS[self.ITEMS_COLUMN_INDEX],
            index=index,
        )
        for output_file in output_files:
            output_extension = get_extension(output_file)
            logger.info("Writing lead time histogram data to %s", output_file)
            if output_extension == ".json":
                file_data.to_json(output_file, date_format="iso")
            elif output_extension == ".xlsx":
                file_data.to_frame(name="lead_time_histogram").to_excel(
                    output_file, sheet_name="LeadTimeHistogram", header=True
                )
            else:
                # Reset index to convert it to a named column
                file_data_to_write = file_data.reset_index()
                file_data_to_write.columns = self.CSV_HISTOGRAM_COLUMNS
                file_data_to_write.to_csv(output_file, header=True, index=False)

    def write_lead_time_chart(self, _data, output_file):
        """Write lead time histogram chart to output file.

        Args:
            _data: Unused parameter (kept for interface compatibility)
            output_file: Path to output file
        """
        cycle_data = self.get_result(CycleTimeCalculator)
        chart_data = cycle_data[["lead_time", "completed_timestamp"]].dropna(
            subset=["lead_time"]
        )
        lt_days = chart_data["lead_time"].dt.days
        if len(lt_days.index) < 2:
            logger.warning(
                "Need at least 2 completed items to draw lead time histogram"
            )
            return

        # Apply window filtering if specified
        chart_data = self._apply_lead_time_window_filter(chart_data)
        if chart_data is None:
            return

        lt_days = chart_data["lead_time"].dt.days
        self._create_lead_time_histogram_chart(lt_days, output_file)

    def _apply_lead_time_window_filter(self, chart_data):
        """Apply window filtering to lead time chart data."""
        window = self.settings.get("histogram_window")
        if window:
            start = chart_data["completed_timestamp"].max().normalize() - pd.Timedelta(
                window, "D"
            )
            chart_data = chart_data[chart_data.completed_timestamp >= start]
            lt_days = chart_data["lead_time"].dt.days
            if len(lt_days.index) < 2:
                logger.warning(
                    "Need at least 2 completed items to draw lead time histogram"
                )
                return None
        return chart_data

    def _create_lead_time_histogram_chart(self, lt_days, output_file):
        """Create and save lead time histogram chart."""
        quantiles = self.settings["quantiles"]
        logger.debug(
            "Showing lead time histogram at quantiles %s",
            ", ".join([f"{q * 100.0:.2f}" for q in quantiles]),
        )
        fig, ax = plt.subplots()
        bins = range(int(lt_days.max()) + 2)
        sns.histplot(lt_days, bins=bins, ax=ax, kde=False)
        ax.set_xlabel("Lead time (days)")
        if self.settings.get("lead_time_histogram_chart_title"):
            ax.set_title(self.settings["lead_time_histogram_chart_title"])
        _, right = ax.get_xlim()
        ax.set_xlim(0, right)
        self._add_quantile_lines(ax, lt_days, quantiles, metric_name="Lead time")
        ax.set_ylabel("Frequency")
        set_chart_style()
        logger.info("Writing lead time histogram chart to %s", output_file)
        fig.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close(fig)
