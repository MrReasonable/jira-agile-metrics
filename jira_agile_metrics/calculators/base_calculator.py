"""Base calculator class with common functionality for Jira Agile Metrics.

This module provides a base calculator class that contains common functionality
shared across multiple calculator classes to reduce code duplication.
"""

import logging

import pandas as pd

from ..calculator import Calculator
from ..utils import breakdown_by_month, create_monthly_bar_chart

logger = logging.getLogger(__name__)


class BaseCalculator(Calculator):
    """Base calculator class with common functionality."""

    def create_dataframe_from_series(self, series, columns):
        """Create a DataFrame from series data with specified columns."""
        data = {}
        for k, v in series.items():
            data[k] = pd.Series(v["data"], dtype=v["dtype"])

        return pd.DataFrame(data, columns=columns)

    def check_chart_data_empty(self, chart_data, chart_name):
        """Check if chart data is empty and log warning if so."""
        if chart_data is None:
            return True

        if len(chart_data.index) == 0:
            logger.warning("Cannot draw %s chart with zero items", chart_name)
            return True

        return False

    def get_active_cycle_columns(self):
        """Get active cycle columns between committed and done columns."""
        cycle_names = [s["name"] for s in self.settings["cycle"]]
        committed_column = self.settings["committed_column"]
        done_column = self.settings["done_column"]

        # Validate that committed_column exists in cycle_names
        if committed_column not in cycle_names:
            raise ValueError(
                f"committed_column '{committed_column}' not found in cycle_names. "
                f"Available cycle_names: {cycle_names}"
            )

        # Validate that done_column exists in cycle_names
        if done_column not in cycle_names:
            raise ValueError(
                f"done_column '{done_column}' not found in cycle_names. "
                f"Available cycle_names: {cycle_names}"
            )

        # Validate that committed_column comes before done_column
        committed_index = cycle_names.index(committed_column)
        done_index = cycle_names.index(done_column)
        if committed_index >= done_index:
            raise ValueError(
                f"committed_column '{committed_column}' (index {committed_index}) "
                f"must come before done_column '{done_column}' (index {done_index}) "
                f"in cycle_names: {cycle_names}"
            )

        return cycle_names[committed_index:done_index]

    def create_common_series_structure(self, fields):
        """Create a common series structure for data collection."""
        series = {}
        for field in fields:
            if field == "key":
                series[field] = {"data": [], "dtype": "object"}
            elif field in ["created", "resolved", "start", "end"]:
                series[field] = {"data": [], "dtype": "datetime64[ns]"}
            else:
                series[field] = {"data": [], "dtype": "object"}
        return series

    def create_monthly_breakdown_chart(self, chart_data, output_file, config):
        """Create a monthly breakdown chart with common patterns.

        Args:
            chart_data: DataFrame with chart data
            output_file: Path to save the chart
            config: Dictionary containing:
                - created_field: Field name for created date
                - resolved_field: Field name for resolved date
                - group_field: Field name for grouping
                - group_values: List of group values
                - window: Optional time window
                - chart_title: Optional chart title
        """
        if self.check_chart_data_empty(chart_data, "monthly breakdown"):
            return

        breakdown = breakdown_by_month(
            chart_data,
            {
                "start_column": config["created_field"],
                "end_column": config["resolved_field"],
                "key_column": "key",
                "value_column": config["group_field"],
                "output_columns": config["group_values"],
            },
        )

        if config.get("window"):
            breakdown = breakdown[-config["window"] :]

        create_monthly_bar_chart(
            breakdown, "monthly breakdown", output_file, config.get("chart_title")
        )

    def add_issue_data_to_series(self, series, issue, fields_mapping):
        """Add issue data to series with common field mapping."""
        for field, field_id in fields_mapping.items():
            if field == "key":
                series[field]["data"].append(issue.key)
            elif field in ["created", "resolved"]:
                value = getattr(issue.fields, field, None)
                series[field]["data"].append(value)
            else:
                value = self.query_manager.resolve_field_value(issue, field_id)
                series[field]["data"].append(value)

    def add_quantile_annotations(self, ax, chart_data, quantiles):
        """Add quantile annotations to a chart."""
        left, right = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        offset = 0.02 * (ymax - ymin)
        for quantile, value in chart_data["cycle_time"].quantile(quantiles).items():
            ax.hlines(value, left, right, linestyles="--", linewidths=1)
            ax.annotate(
                f"{quantile * 100:.0f}% ({value:.0f} days)",
                xy=(left, value),
                xytext=(left, value + offset),
                fontsize="x-small",
                ha="left",
            )
