"""Defects calculator for Jira Agile Metrics.

This module provides functionality to calculate defect metrics from JIRA data.
"""

import logging

import dateutil.parser

from ..common_constants import COMMON_ISSUE_DATA_PATTERN
from ..utils import (
    breakdown_by_month,
    create_common_defects_columns,
    create_monthly_bar_chart,
)
from .base_calculator import BaseCalculator

logger = logging.getLogger(__name__)


class DefectsCalculator(BaseCalculator):
    """Calculate defect concentration

    Queries JIRA with JQL set in `defects_query` and creates three stacked
    bar charts, presuming their file name values are set. Each shows the
    concentration of defects by month. The number of months to show can be
    limited with `defects_window`.

    - `defects_by_priority_chart`: Grouped by priority
      (`defects_priority_field`), optionally limited to a list of known values
      in order (`defects_priority_values`) and with title
      `defects_by_priority_chart_title`.
    - `defects_by_type_chart`: Grouped by type
      (`defects_type_field`), optionally limited to a list of known values
      in order (`defects_type_values`) and with title
      `defects_by_type_chart_title`.
    - `defects_by_environment_chart`: Grouped by environment
      (`defects_environment_field`), optionally limited to a list of known
      values in order (`defects_environment_values`) and with title
      `defects_by_environment_chart_title`.
    """

    def run(self):
        query = self.settings["defects_query"]

        # This calculation is expensive. Only run it if we have a query.
        if not query:
            logger.debug("Not calculating defects chart data as no query specified")
            return None

        # Get the fields
        priority_field = self.settings["defects_priority_field"]
        priority_field_id = (
            self.query_manager.field_name_to_id(priority_field)
            if priority_field
            else None
        )

        type_field = self.settings["defects_type_field"]
        type_field_id = (
            self.query_manager.field_name_to_id(type_field) if type_field else None
        )

        environment_field = self.settings["defects_environment_field"]
        environment_field_id = (
            self.query_manager.field_name_to_id(environment_field)
            if environment_field
            else None
        )

        # Build data frame
        columns = create_common_defects_columns()
        series = self.create_common_series_structure(columns)

        for issue in self.query_manager.find_issues(query, expand=None):
            # Add common fields
            issue_data_pattern = COMMON_ISSUE_DATA_PATTERN.copy()
            issue_data_pattern.update(
                {
                    "priority": priority_field_id,
                    "type": type_field_id,
                    "environment": environment_field_id,
                }
            )
            self.add_issue_data_to_series(series, issue, issue_data_pattern)

            # Add date fields
            series["created"]["data"].append(
                dateutil.parser.parse(issue.fields.created)
            )
            series["resolved"]["data"].append(
                dateutil.parser.parse(issue.fields.resolutiondate)
                if issue.fields.resolutiondate
                else None
            )

        return self.create_dataframe_from_series(series, columns)

    def write(self):
        chart_data = self.get_result()
        if chart_data is None:
            return

        if self.check_chart_data_empty(chart_data, "defect"):
            return

        if self.settings["defects_by_priority_chart"]:
            self.write_defects_by_priority_chart(
                chart_data, self.settings["defects_by_priority_chart"]
            )

        if self.settings["defects_by_type_chart"]:
            self.write_defects_by_type_chart(
                chart_data, self.settings["defects_by_type_chart"]
            )

        if self.settings["defects_by_environment_chart"]:
            self.write_defects_by_environment_chart(
                chart_data, self.settings["defects_by_environment_chart"]
            )

    def write_defects_by_priority_chart(self, chart_data, output_file):
        """Write defects by priority chart showing monthly breakdown."""
        window = self.settings["defects_window"]
        priority_values = self.settings["defects_priority_values"]

        breakdown = breakdown_by_month(
            chart_data,
            {
                "start_column": "created",
                "end_column": "resolved",
                "key_column": "key",
                "value_column": "priority",
                "output_columns": priority_values,
            },
        )

        if window:
            breakdown = breakdown[-window:]

        if len(breakdown.index) == 0 or len(breakdown.columns) == 0:
            logger.warning("Cannot draw defects by priority chart with zero items")
            return

        create_monthly_bar_chart(
            breakdown,
            "defects by priority",
            output_file,
            self.settings.get("defects_by_priority_chart_title"),
        )

    def write_defects_by_type_chart(self, chart_data, output_file):
        """Write defects by type chart showing monthly breakdown of defects by type."""
        window = self.settings["defects_window"]
        type_values = self.settings["defects_type_values"]

        breakdown = breakdown_by_month(
            chart_data,
            {
                "start_column": "created",
                "end_column": "resolved",
                "key_column": "key",
                "value_column": "type",
                "output_columns": type_values,
            },
        )

        if window:
            breakdown = breakdown[-window:]

        if len(breakdown.index) == 0 or len(breakdown.columns) == 0:
            logger.warning("Cannot draw defects by type chart with zero items")
            return

        create_monthly_bar_chart(
            breakdown,
            "defects by type",
            output_file,
            self.settings.get("defects_by_type_chart_title"),
        )

    def write_defects_by_environment_chart(self, chart_data, output_file):
        """Write defects by environment chart showing monthly breakdown."""
        window = self.settings["defects_window"]
        environment_values = self.settings["defects_environment_values"]

        breakdown = breakdown_by_month(
            chart_data,
            {
                "start_column": "created",
                "end_column": "resolved",
                "key_column": "key",
                "value_column": "environment",
                "output_columns": environment_values,
            },
        )

        if window:
            breakdown = breakdown[-window:]

        if len(breakdown.index) == 0 or len(breakdown.columns) == 0:
            logger.warning("Cannot draw defects by environment chart with zero items")
            return

        create_monthly_bar_chart(
            breakdown,
            "defects by environment",
            output_file,
            self.settings.get("defects_by_environment_chart_title"),
        )
