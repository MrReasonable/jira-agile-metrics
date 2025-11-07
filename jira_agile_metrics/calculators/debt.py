"""Debt calculator for Jira Agile Metrics.

This module provides functionality to calculate technical debt metrics from JIRA data.
"""

import datetime
import logging

import dateutil.parser
import matplotlib.pyplot as plt
import pandas as pd

from ..chart_styling_utils import set_chart_style
from ..columns import DEBT_COLUMNS
from ..common_constants import COMMON_ISSUE_DATA_PATTERN
from ..utils import to_bin
from .base_calculator import BaseCalculator

logger = logging.getLogger(__name__)


class DebtCalculator(BaseCalculator):
    """Calculate technical debt over time.

    Queries JIRA with JQL set in `debt_query` and draws a stacked bar
    chart in the file `debt_chart` with title `debt_chart_title`. The bars are
    the last 6 months (or another window set in `debt_window`), grouped by
    priority. The field giving the priority is set with `debt_chart_priority`.
    To force the list of valid values and their order, provide a list of
    strings in `debt_priority_values`.

    Also draw a stacked bar chart in the file `debt_age_chart`, with title
    `debt_age_chart_title`, grouping by item age.
    """

    def run(self, now=None):
        query = self.settings["debt_query"]

        # Allows unit testing to use a fixed date
        if now is None:
            now = datetime.datetime.now(datetime.timezone.utc)

        # This calculation is expensive. Only run it if we have a query.
        if not query:
            logger.debug("Not calculating debt chart data as no query specified")
            return None

        # Resolve field name to field id for later lookup
        priority_field_id = self._get_priority_field_id()

        # Build data frame
        series = self._initialize_series()

        for issue in self.query_manager.find_issues(query, expand=None):
            self._process_issue(issue, series, priority_field_id, now)

        return self._create_dataframe(series)

    def _get_priority_field_id(self):
        """Get priority field ID for lookup."""
        priority_field = self.settings["debt_priority_field"]
        return (
            self.query_manager.field_name_to_id(priority_field)
            if priority_field
            else None
        )

    def _initialize_series(self):
        """Initialize series structure for debt data."""
        series = self.create_common_series_structure(
            [
                "key",
                "priority",
                "created",
                "resolved",
                "type",
                "environment",
            ]
        )
        series["age"] = {"data": [], "dtype": "timedelta64[ns]"}
        return series

    def _process_issue(self, issue, series, priority_field_id, now):
        """Process a single issue and add to series."""
        created_date = dateutil.parser.parse(issue.fields.created)
        resolved_date = (
            dateutil.parser.parse(issue.fields.resolutiondate)
            if issue.fields.resolutiondate
            else None
        )

        # Add common fields
        issue_data_pattern = COMMON_ISSUE_DATA_PATTERN.copy()
        issue_data_pattern.update({"priority": priority_field_id})
        self.add_issue_data_to_series(series, issue, issue_data_pattern)

        # Add date fields
        series["created"]["data"].append(created_date)
        series["resolved"]["data"].append(resolved_date)

        # Calculate age
        age = self._calculate_age(created_date, resolved_date, now)
        series["age"]["data"].append(age)

    def _calculate_age(self, created_date, resolved_date, now):
        """Calculate age of issue."""

        def normalize_to_utc(dt):
            """Normalize datetime to UTC-aware datetime.

            Args:
                dt: datetime (naive or aware)

            Returns:
                datetime with UTC timezone, or None if dt is None
            """
            if dt is None:
                return None
            if dt.tzinfo is not None:
                return dt.astimezone(datetime.timezone.utc)
            # Treat naive datetime as UTC
            return dt.replace(tzinfo=datetime.timezone.utc)

        # Normalize both datetimes to UTC before computing difference
        created_utc = normalize_to_utc(created_date)

        if resolved_date is not None:
            resolved_utc = normalize_to_utc(resolved_date)
        else:
            resolved_utc = normalize_to_utc(now)

        # Compute subtraction while both are UTC-aware
        return resolved_utc - created_utc

    def _create_dataframe(self, series):
        """Create DataFrame from series data."""

        columns = DEBT_COLUMNS
        data = {}
        for k, v in series.items():
            data[k] = pd.Series(v["data"], dtype=v["dtype"])
        return pd.DataFrame(data, columns=columns)

    def write(self):
        chart_data = self.get_result()
        if chart_data is None:
            return

        if len(chart_data.index) == 0:
            logger.warning("Cannot draw debt chart with zero items")
            return

        if self.settings["debt_chart"]:
            self.write_debt_chart(chart_data, self.settings["debt_chart"])

        if self.settings["debt_age_chart"]:
            self.write_debt_age_chart(chart_data, self.settings["debt_age_chart"])

    def write_debt_chart(self, chart_data, output_file):
        """Write debt chart using common monthly breakdown pattern."""
        config = {
            "created_field": "created",
            "resolved_field": "resolved",
            "group_field": "priority",
            "group_values": self.settings["debt_priority_values"],
            "window": self.settings["debt_window"],
            "chart_title": self.settings.get("debt_chart_title"),
        }
        self.create_monthly_breakdown_chart(chart_data, output_file, config)

    def write_debt_age_chart(self, chart_data, output_file):
        """Write debt age chart showing debt distribution by priority and age bins."""
        priority_values = self.settings["debt_priority_values"]
        bins = self.settings["debt_age_chart_bins"]

        def generate_bin_label(v):
            low, high = to_bin(v, bins)
            return f"> {low} days" if high is None else f"{low}-{high} days"

        def day_grouper(value):
            if isinstance(value, pd.Timedelta):
                return generate_bin_label(value.days)
            return None

        bin_labels = list(map(generate_bin_label, bins + [bins[-1] + 1]))
        breakdown = (
            chart_data.pivot_table(
                index="age", columns="priority", values="key", aggfunc="count"
            )
            .groupby(day_grouper)
            .sum()
            .reindex(bin_labels)
            .T
        )

        if priority_values:
            breakdown = breakdown.reindex(priority_values)

        fig, ax = plt.subplots()

        breakdown.plot.barh(ax=ax, stacked=True)

        if self.settings["debt_age_chart_title"]:
            ax.set_title(self.settings["debt_age_chart_title"])

        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_xlabel("Number of items", labelpad=20)
        ax.set_ylabel("Priority", labelpad=10)

        set_chart_style()

        # Write file
        logger.info("Writing debt age chart to %s", output_file)
        fig.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close(fig)
