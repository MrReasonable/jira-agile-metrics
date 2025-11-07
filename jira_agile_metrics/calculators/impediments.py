"""Impediments calculator for Jira Agile Metrics.

This module provides functionality to calculate impediments metrics from JIRA data.
"""

import logging
import os

import matplotlib.pyplot as plt
import pandas as pd

from ..chart_styling_utils import _format_index_labels, set_chart_style
from ..utils import (
    breakdown_by_month,
    breakdown_by_month_sum_days,
    get_extension,
)
from .base_calculator import BaseCalculator
from .cycletime import CycleTimeCalculator

logger = logging.getLogger(__name__)


class ImpedimentsCalculator(BaseCalculator):
    """Calculate impediments, charted by month and workflow status,
    either as a count of tickets that were blocked in that month,
    or as a sum of the total number of days of blockage for all
    tickets in that month.

    Writes to `impediments_chart`, `impediments_days_chart`,
    `impediments_tatus_chart`, and `impediments_status_days_chart`,
    respectively, with corresponding titles. The number of months to
    output can be restricted with `impediments_window`.
    Raw data can be written to `impediments_data`.
    """

    def _has_valid_output_config(self, setting_name):
        """Check if a setting is configured with a non-empty value.

        Args:
            setting_name: The name of the setting to check.

        Returns:
            bool: True if the setting exists and has a non-empty value,
                False otherwise. For lists, returns True if any element
                is truthy. For strings, returns True if the string is
                non-empty.
        """
        value = self.settings.get(setting_name)
        if not value:
            return False
        if isinstance(value, list):
            return any(value)
        return True

    def run(self):
        # This calculation is expensive.
        # Only run it if we are going to write something
        # Check that settings are explicitly configured (not None and not empty)
        has_output = (
            self._has_valid_output_config("impediments_data")
            or self._has_valid_output_config("impediments_chart")
            or self._has_valid_output_config("impediments_days_chart")
            or self._has_valid_output_config("impediments_status_chart")
            or self._has_valid_output_config("impediments_status_days_chart")
        )
        if not has_output:
            logger.debug(
                "Not calculating impediments data as no output files specified"
            )
            return None

        cycle_data = self.get_result(CycleTimeCalculator)
        cycle_data = cycle_data[cycle_data.blocked_days > 0][["key", "impediments"]]

        data = []

        active_columns = self.get_active_cycle_columns()

        for row in cycle_data.itertuples():
            for _, event in enumerate(row.impediments):
                # Ignore things that were impeded whilst
                # in the backlog and/or done column
                # (these are mostly nonsensical,
                # and don't really indicate blocked/wasted time)

                if event["status"] not in active_columns:
                    continue
                data.append(
                    {
                        "key": row.key,
                        "status": event["status"],
                        "flag": event["flag"],
                        "start": pd.Timestamp(event["start"]),
                        "end": (pd.Timestamp(event["end"]) if event["end"] else pd.NaT),
                    }
                )

        return pd.DataFrame(data, columns=["key", "status", "flag", "start", "end"])

    def write(self):
        data = self.get_result()
        if data is None:
            return

        # Only write files if explicitly configured (not None and not empty)
        if self.settings.get("impediments_data"):
            output_files = self.settings["impediments_data"]
            # Handle both string and list formats
            if isinstance(output_files, str):
                output_files = [output_files]
            # Filter out empty strings
            output_files = [f for f in output_files if f]
            if output_files:
                self.write_data(data, output_files)

        if self.settings.get("impediments_chart"):
            output_file = self.settings["impediments_chart"]
            if output_file:  # Ensure it's not empty string
                logger.debug(
                    "Writing impediments chart to: %s (cwd: %s)",
                    output_file,
                    os.getcwd(),
                )
                self.write_impediments_chart(data, output_file)

        if self.settings.get("impediments_days_chart"):
            output_file = self.settings["impediments_days_chart"]
            if output_file:  # Ensure it's not empty string
                logger.debug(
                    "Writing impediments days chart to: %s (cwd: %s)",
                    output_file,
                    os.getcwd(),
                )
                self.write_impediments_days_chart(data, output_file)

        if self.settings.get("impediments_status_chart"):
            output_file = self.settings["impediments_status_chart"]
            if output_file:  # Ensure it's not empty string
                logger.debug(
                    "Writing impediments status chart to: %s (cwd: %s)",
                    output_file,
                    os.getcwd(),
                )
                self.write_impediments_status_chart(data, output_file)

        if self.settings.get("impediments_status_days_chart"):
            output_file = self.settings["impediments_status_days_chart"]
            if output_file:  # Ensure it's not empty string
                logger.debug(
                    "Writing impediments status days chart to: %s (cwd: %s)",
                    output_file,
                    os.getcwd(),
                )
                self.write_impediments_status_days_chart(data, output_file)

    def write_data(self, data, output_files):
        """Write impediments data to output files."""
        for output_file in output_files:
            output_extension = get_extension(output_file)

            logger.info("Writing impediments data to %s", output_file)
            if output_extension == ".json":
                data.to_json(output_file, date_format="iso")
            elif output_extension == ".xlsx":
                data.to_excel(output_file, sheet_name="Impediments", header=True)
            else:
                data.to_csv(
                    output_file,
                    header=True,
                    date_format="%Y-%m-%d",
                    index=False,
                )

    def write_impediments_chart(self, chart_data, output_file):
        """Write impediments chart showing impediment trends over time."""
        if len(chart_data.index) == 0:
            logger.warning("Cannot draw impediments chart with zero items")
            return

        window = self.settings.get("impediments_window")
        breakdown = breakdown_by_month(
            chart_data,
            {
                "start_column": "start",
                "end_column": "end",
                "key_column": "key",
                "value_column": "flag",
            },
        )

        if window:
            breakdown = breakdown[-window:]

        if len(breakdown.index) == 0:
            logger.warning("Cannot draw impediments chart with zero items")
            return

        fig, ax = plt.subplots()

        breakdown.plot.bar(ax=ax, stacked=True)

        if self.settings.get("impediments_chart_title"):
            ax.set_title(self.settings["impediments_chart_title"])

        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_xlabel("Month", labelpad=20)
        ax.set_ylabel("Number of impediments", labelpad=10)

        # Set ticks to match data positions to ensure proper label alignment
        num_data_points = len(breakdown.index)
        tick_positions = list(range(num_data_points))
        ax.set_xticks(tick_positions)
        labels = _format_index_labels(breakdown.index)
        ax.set_xticklabels(labels, rotation=90, size="small")

        set_chart_style()

        # Write file
        logger.info("Writing impediments chart to %s", output_file)
        fig.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def write_impediments_days_chart(self, chart_data, output_file):
        """Write impediments days chart to output file."""
        if len(chart_data.index) == 0:
            logger.warning("Cannot draw impediments days chart with zero items")
            return

        window = self.settings.get("impediments_window")
        breakdown = breakdown_by_month_sum_days(
            chart_data,
            {"start_column": "start", "end_column": "end", "value_column": "flag"},
        )

        if window:
            breakdown = breakdown[-window:]

        if len(breakdown.index) == 0:
            logger.warning("Cannot draw impediments chart with zero items")
            return

        fig, ax = plt.subplots()

        breakdown.plot.bar(ax=ax, stacked=True)

        if self.settings.get("impediments_days_chart_title"):
            ax.set_title(self.settings["impediments_days_chart_title"])

        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_xlabel("Month", labelpad=20)
        ax.set_ylabel("Total impeded days", labelpad=10)

        # Set ticks to match data positions to ensure proper label alignment
        num_data_points = len(breakdown.index)
        tick_positions = list(range(num_data_points))
        ax.set_xticks(tick_positions)
        labels = _format_index_labels(breakdown.index)
        ax.set_xticklabels(labels, rotation=90, size="small")

        set_chart_style()

        # Write file
        logger.info("Writing impediments days chart to %s", output_file)
        fig.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def write_impediments_status_chart(self, chart_data, output_file):
        """Write impediments status chart to output file."""
        if len(chart_data.index) == 0:
            logger.warning("Cannot draw impediments status chart with zero items")
            return

        window = self.settings.get("impediments_window")
        cycle_names = [s["name"] for s in self.settings["cycle"]]

        breakdown = breakdown_by_month(
            chart_data,
            {
                "start_column": "start",
                "end_column": "end",
                "key_column": "key",
                "value_column": "status",
                "output_columns": cycle_names,
            },
        )

        if window:
            breakdown = breakdown[-window:]

        if len(breakdown.index) == 0:
            logger.warning("Cannot draw impediments status chart with zero items")
            return

        fig, ax = plt.subplots()

        breakdown.plot.bar(ax=ax, stacked=True)

        if self.settings.get("impediments_status_chart_title"):
            ax.set_title(self.settings["impediments_status_chart_title"])

        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_xlabel("Month", labelpad=20)
        ax.set_ylabel("Number of impediments", labelpad=10)

        # Set ticks to match data positions to ensure proper label alignment
        num_data_points = len(breakdown.index)
        tick_positions = list(range(num_data_points))
        ax.set_xticks(tick_positions)
        labels = _format_index_labels(breakdown.index)
        ax.set_xticklabels(labels, rotation=90, size="small")

        set_chart_style()

        # Write file
        logger.info("Writing impediments status chart to %s", output_file)
        fig.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def write_impediments_status_days_chart(self, chart_data, output_file):
        """Write impediments status days chart to output file."""
        if len(chart_data.index) == 0:
            logger.warning("Cannot draw impediments status days chart with zero items")
            return

        window = self.settings.get("impediments_window")
        cycle_names = [s["name"] for s in self.settings["cycle"]]

        breakdown = breakdown_by_month_sum_days(
            chart_data,
            {
                "start_column": "start",
                "end_column": "end",
                "value_column": "status",
                "output_columns": cycle_names,
            },
        )

        if window:
            breakdown = breakdown[-window:]

        if len(breakdown.index) == 0:
            logger.warning("Cannot draw impediments status days chart with zero items")
            return

        fig, ax = plt.subplots()

        breakdown.plot.bar(ax=ax, stacked=True)

        if self.settings.get("impediments_status_days_chart_title"):
            ax.set_title(self.settings["impediments_status_days_chart_title"])

        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_xlabel("Month", labelpad=20)
        ax.set_ylabel("Total impeded days", labelpad=10)

        # Set ticks to match data positions to ensure proper label alignment
        num_data_points = len(breakdown.index)
        tick_positions = list(range(num_data_points))
        ax.set_xticks(tick_positions)
        labels = _format_index_labels(breakdown.index)
        ax.set_xticklabels(labels, rotation=90, size="small")

        set_chart_style()

        # Write file
        logger.info("Writing impediments status days chart to %s", output_file)
        fig.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close(fig)
