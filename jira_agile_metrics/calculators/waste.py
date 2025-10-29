"""Waste calculator for Jira Agile Metrics.

This module provides functionality to calculate waste metrics from JIRA data.
"""

import logging

import dateutil
import pandas as pd

from ..utils import create_monthly_bar_chart
from .base_calculator import BaseCalculator

logger = logging.getLogger(__name__)


class WasteCalculator(BaseCalculator):
    """Calculate stories withdrawn, grouped by the time of withdrawal and
    stage prior to withdrawal.

    Chart is drawn in `waste_chart`, with title `waste_chart_title`, using
    tickets fetched with `waste_chart_query`, and grouped by month, limited to
    `waste_chart_window` months (if given).
    """

    def run(self):
        query = self.settings["waste_query"]
        if not query:
            logger.debug("Not calculating waste chart data as no query specified")
            return None

        active_columns = self.get_active_cycle_columns()

        cycle_lookup = {}
        for idx, cycle_step in enumerate(self.settings["cycle"]):
            for status in cycle_step["statuses"]:
                cycle_lookup[status.lower()] = {
                    "index": idx,
                    "name": cycle_step["name"],
                }

        columns = ["key", "last_status", "resolution", "withdrawn_date"]
        series = {
            "key": {"data": [], "dtype": "str"},
            "last_status": {"data": [], "dtype": "str"},
            "resolution": {"data": [], "dtype": "str"},
            "withdrawn_date": {"data": [], "dtype": "datetime64[ns]"},
        }

        for issue in self.query_manager.find_issues(query):
            # Assume all waste items are resolved somehow
            if not issue.fields.resolution:
                continue

            last_status = None
            status_changes = list(self.query_manager.iter_changes(issue, ["status"]))
            if len(status_changes) > 0:
                last_status = status_changes[-1].from_string

            if last_status is not None and last_status.lower() in cycle_lookup:
                last_status = cycle_lookup.get(last_status.lower())["name"]
            else:
                logger.warning(
                    "Issue %s transitioned from unknown JIRA status %s",
                    issue.key,
                    last_status,
                )

            # Skip if last_status was not in one of the active columns
            if last_status not in active_columns:
                continue

            series["key"]["data"].append(issue.key)
            series["last_status"]["data"].append(last_status)
            series["resolution"]["data"].append(issue.fields.resolution.name)
            series["withdrawn_date"]["data"].append(
                dateutil.parser.parse(issue.fields.resolutiondate)
            )

        return self.create_dataframe_from_series(series, columns)

    def write(self):
        chart_data = self.get_result()
        if chart_data is None:
            return

        output_file = self.settings["waste_chart"]
        if not output_file:
            logger.debug("No output file specified for waste chart")
            return

        if self.check_chart_data_empty(chart_data, "waste"):
            return

        frequency = self.settings["waste_frequency"]
        window = self.settings["waste_window"]

        active_columns = self.get_active_cycle_columns()

        breakdown = (
            chart_data.pivot_table(
                index="withdrawn_date",
                columns="last_status",
                values="key",
                aggfunc="count",
            )
            .groupby(pd.Grouper(freq=frequency, closed="left", label="left"))
            .sum()
            .reindex(active_columns, axis=1)
        )

        if window:
            breakdown = breakdown[-window:]

        if len(breakdown.index) == 0 or len(breakdown.columns) == 0:
            logger.warning("Cannot draw waste chart with zero items")
            return

        create_monthly_bar_chart(
            breakdown,
            "waste",
            output_file,
            self.settings.get("waste_chart_title"),
        )
