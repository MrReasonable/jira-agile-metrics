import datetime
import json
import logging
import pprint
import traceback

import dateutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..calculator import Calculator
from ..trello import TrelloClient
from ..utils import get_extension, to_json_string

logger = logging.getLogger(__name__)


class CycleTimeCalculator(Calculator):
    """Basic cycle time data, fetched from JIRA.

    Builds a numerically indexed data frame with the following 'fixed'
    columns: `key`, 'url', 'issue_type', `summary`, `status`, and
    `resolution` from JIRA, as well as the value of any fields set in
    the `fields` dict in `settings`. If `known_values` is set (a dict of
    lists, with field names as keys and a list of known values for each
    field as values) and a field in `fields` contains a list of values,
    only the first value in the list of known values will be used.

    If 'query_attribute' is set in `settings`, a column with this name
    will be added, and populated with the `value` key, if any, from each
    criteria block under `queries` in settings.

    In addition, `cycle_time` will be set to the time delta between the
    first `accepted`-type column and the first `complete` column, or None.

    The remaining columns are the names of the items in the configured
    cycle, in order.

    Each cell contains the last date/time stamp when the relevant status
    was set.

    If an item moves backwards through the cycle, subsequent date/time
    stamps in the cycle are erased.
    """

    def run(self, now=None):
        return calculate_cycle_times(
            self.query_manager,
            self.settings["cycle"],
            self.settings["attributes"],
            self.settings["committed_column"],
            self.settings["done_column"],
            self.settings["queries"],
            self.settings["query_attribute"],
            lead_time_start_column=self.settings.get("lead_time_start_column"),
            now=now,
            reset_on_backwards=self.settings.get("reset_on_backwards", True),
        )

    def write(self):
        output_files = self.settings["cycle_time_data"]

        if not output_files:
            logger.debug("No output file specified for cycle time data")
            return

        cycle_data = self.get_result()
        cycle_names = [s["name"] for s in self.settings["cycle"]]
        attribute_names = sorted(self.settings["attributes"].keys())
        query_attribute_names = (
            [self.settings["query_attribute"]] if self.settings["query_attribute"] else []
        )

        header = (
            ["ID", "Link", "Name"]
            + cycle_names
            + ["Type", "Status", "Resolution"]
            + attribute_names
            + query_attribute_names
            + ["Blocked Days"]
        )
        columns = (
            ["key", "url", "summary"]
            + cycle_names
            + ["issue_type", "status", "resolution"]
            + attribute_names
            + query_attribute_names
            + ["blocked_days"]
        )

        for output_file in output_files:
            logger.info("Writing cycle time data to %s", output_file)
            output_extension = get_extension(output_file)

            if output_extension == ".json":
                values = [header] + [
                    list(map(to_json_string, row)) for row in cycle_data[columns].values.tolist()
                ]
                with open(output_file, "w") as out:
                    out.write(json.dumps(values))
            elif output_extension == ".xlsx":
                cycle_data.to_excel(
                    output_file,
                    "Cycle data",
                    columns=columns,
                    header=header,
                    index=False,
                )
            else:
                cycle_data.to_csv(
                    output_file,
                    columns=columns,
                    header=header,
                    date_format="%Y-%m-%d",
                    index=False,
                )


def calculate_cycle_times(
    query_manager,
    cycle,  # [{name:"", statuses:[""], type:""}]
    attributes,  # [{key:value}]
    committed_column,  # "" in `cycle`
    done_column,  # "" in `cycle`
    queries,  # [{jql:"", value:""}]
    query_attribute=None,  # ""
    lead_time_start_column=None,  # Optional: name of the column to use as lead time start
    now=None,
    reset_on_backwards=True,  # New option
):
    # Allows unit testing to use a fixed date
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)

    cycle_names = [s["name"] for s in cycle]
    if lead_time_start_column is None:
        lead_time_start_column = cycle_names[0]
    active_columns = cycle_names[
        cycle_names.index(committed_column) : cycle_names.index(done_column)
    ]

    cycle_lookup = {}
    for idx, cycle_step in enumerate(cycle):
        for status in cycle_step["statuses"]:
            cycle_lookup[status.lower()] = dict(
                index=idx,
                name=cycle_step["name"],
            )

    unmapped_statuses = set()

    series = {
        "key": {"data": [], "dtype": "str"},
        "url": {"data": [], "dtype": "str"},
        "issue_type": {"data": [], "dtype": "str"},
        "summary": {"data": [], "dtype": "str"},
        "status": {"data": [], "dtype": "str"},
        "resolution": {"data": [], "dtype": "str"},
        "cycle_time": {"data": [], "dtype": "timedelta64[ns]"},
        "lead_time": {"data": [], "dtype": "timedelta64[ns]"},
        "completed_timestamp": {"data": [], "dtype": "datetime64[ns]"},
        "blocked_days": {"data": [], "dtype": "int"},
        "impediments": {
            "data": [],
            "dtype": "object",
        },  # list of {'start', 'end', 'status', 'flag'}
    }

    for cycle_name in cycle_names:
        series[cycle_name] = {"data": [], "dtype": "datetime64[ns]"}

    for name in attributes:
        series[name] = {"data": [], "dtype": "object"}

    if query_attribute:
        series[query_attribute] = {"data": [], "dtype": "str"}

    for criteria in queries:
        for issue in query_manager.find_issues(criteria["jql"]):
            if isinstance(query_manager.jira, TrelloClient):
                issue_url = issue.url
            else:
                issue_url = "%s/browse/%s" % (
                    query_manager.jira._options["server"],
                    issue.key,
                )
            item = {
                "key": issue.key,
                "url": issue_url,
                "issue_type": issue.fields.issuetype.name,
                "summary": issue.fields.summary,
                "status": issue.fields.status.name,
                "resolution": issue.fields.resolution.name if issue.fields.resolution else None,
                "cycle_time": None,
                "lead_time": None,
                "completed_timestamp": None,
                "blocked_days": 0,
                "impediments": [],
            }

            for name in attributes:
                item[name] = query_manager.resolve_attribute_value(issue, name)

            if query_attribute:
                item[query_attribute] = criteria.get("value", None)

            for cycle_name in cycle_names:
                item[cycle_name] = None

            last_status = None
            impediment_flag = None
            impediment_start_status = None
            impediment_start = None

            # Record date of status and impediments flag changes
            for snapshot in query_manager.iter_changes(issue, ["status", "Flagged"]):
                if snapshot.change == "status":
                    snapshot_cycle_step = cycle_lookup.get(snapshot.to_string.lower(), None)
                    if snapshot_cycle_step is None:
                        logger.info(
                            "Issue %s transitioned to unknown JIRA status %s",
                            issue.key,
                            snapshot.to_string,
                        )
                        unmapped_statuses.add(snapshot.to_string)
                        continue

                    last_status = snapshot_cycle_step_name = snapshot_cycle_step["name"]

                    # Keep the first time we entered a step
                    if item[snapshot_cycle_step_name] is None:
                        item[snapshot_cycle_step_name] = snapshot.date.date()

                    # Wipe any subsequent dates,
                    # in case this was a move backwards
                    if reset_on_backwards:
                        found_cycle_name = False
                        for cycle_name in cycle_names:
                            if not found_cycle_name and cycle_name == snapshot_cycle_step_name:
                                found_cycle_name = True
                                continue
                            elif found_cycle_name and item[cycle_name] is not None:
                                logger.info(
                                    (
                                        "Issue %s moved backwards to %s "
                                        "[JIRA: %s -> %s], wiping data "
                                        "for subsequent step %s"
                                    ),
                                    issue.key,
                                    snapshot_cycle_step_name,
                                    snapshot.from_string,
                                    snapshot.to_string,
                                    cycle_name,
                                )
                                item[cycle_name] = None
                elif snapshot.change == "Flagged":
                    if snapshot.from_string == snapshot.to_string is None:
                        # Initial state from None -> None
                        continue
                    elif snapshot.to_string is not None and snapshot.to_string != "":
                        impediment_flag = snapshot.to_string
                        impediment_start = snapshot.date.date()
                        impediment_start_status = last_status
                    elif snapshot.to_string is None or snapshot.to_string == "":
                        if impediment_start is None:
                            logger.warning(
                                (
                                    "Issue %s had impediment flag "
                                    "cleared before being set. "
                                    "This should not happen."
                                ),
                                issue.key,
                            )
                            continue

                        if impediment_start_status in active_columns:
                            item["blocked_days"] += (snapshot.date.date() - impediment_start).days
                        item["impediments"].append(
                            {
                                "start": impediment_start,
                                "end": snapshot.date.date(),
                                "status": impediment_start_status,
                                "flag": impediment_flag,
                            }
                        )

                        # Reset for next time
                        impediment_flag = None
                        impediment_start = None
                        impediment_start_status = None

            # If an impediment flag was set but never cleared:
            # treat as resolved on the ticket
            # resolution date if the ticket was resolved,
            # else as still open until today.
            if impediment_start is not None:
                if issue.fields.resolutiondate:
                    resolution_date = dateutil.parser.parse(issue.fields.resolutiondate).date()
                    if impediment_start_status in active_columns:
                        item["blocked_days"] += (resolution_date - impediment_start).days
                    item["impediments"].append(
                        {
                            "start": impediment_start,
                            "end": resolution_date,
                            "status": impediment_start_status,
                            "flag": impediment_flag,
                        }
                    )
                else:
                    if impediment_start_status in active_columns:
                        item["blocked_days"] += (now.date() - impediment_start).days
                    item["impediments"].append(
                        {
                            "start": impediment_start,
                            "end": None,
                            "status": impediment_start_status,
                            "flag": impediment_flag,
                        }
                    )
                impediment_flag = None
                impediment_start = None
                impediment_start_status = None

            # calculate cycle time and lead time

            previous_timestamp = None
            committed_timestamp = None
            done_timestamp = None
            lead_time_start_timestamp = None

            for cycle_name in reversed(cycle_names):
                if item[cycle_name] is not None:
                    previous_timestamp = item[cycle_name]

                if previous_timestamp is not None:
                    item[cycle_name] = previous_timestamp
                    if cycle_name == done_column:
                        done_timestamp = previous_timestamp
                    if cycle_name == committed_column:
                        committed_timestamp = previous_timestamp
                    if cycle_name == lead_time_start_column:
                        lead_time_start_timestamp = previous_timestamp

            if committed_timestamp is not None and done_timestamp is not None:
                item["cycle_time"] = done_timestamp - committed_timestamp
                item["completed_timestamp"] = done_timestamp
            else:
                item["cycle_time"] = None

            if lead_time_start_timestamp is not None and done_timestamp is not None:
                item["lead_time"] = done_timestamp - lead_time_start_timestamp
            else:
                item["lead_time"] = None

            for k, v in item.items():
                series[k]["data"].append(v)

    if len(unmapped_statuses) > 0:
        logger.warn(
            (
                "The following JIRA statuses were found, "
                "but not mapped to a workflow state, "
                "and have been ignored: %s"
            ),
            ", ".join(sorted(unmapped_statuses)),
        )

    data = {}
    for k, v in series.items():
        data[k] = pd.Series(v["data"], dtype=v["dtype"])

    return pd.DataFrame(
        data,
        columns=["key", "url", "issue_type", "summary", "status", "resolution"]
        + sorted(attributes.keys())
        + ([query_attribute] if query_attribute else [])
        + ["cycle_time", "lead_time", "completed_timestamp", "blocked_days", "impediments"]
        + cycle_names,
    )


def calculate_column_durations(cycle_data, cycle_names):
    # Returns a DataFrame: rows=issues, columns=cycle columns, values=duration in days
    durations = []
    for _, row in cycle_data.iterrows():
        times = [row.get(col) for col in cycle_names]
        # Convert to pandas Timestamp if not already
        times = [pd.Timestamp(t) if not pd.isnull(t) else pd.NaT for t in times]
        durations_row = []
        for i in range(len(times) - 1):
            if pd.isnull(times[i]) or pd.isnull(times[i + 1]):
                durations_row.append(np.nan)
            else:
                durations_row.append((times[i + 1] - times[i]).days)
        durations.append(durations_row)
    duration_cols = [f"{cycle_names[i]}→{cycle_names[i + 1]}" for i in range(len(cycle_names) - 1)]
    return pd.DataFrame(durations, columns=duration_cols, index=cycle_data["key"])


def calculate_column_durations_per_column(
    cycle_data, cycle_names, negative_duration_handling="zero"
):
    # Returns a DataFrame: rows=issues, columns=cycle columns (except last),
    # values=duration in days spent in each column
    durations = []
    for _, row in cycle_data.iterrows():
        times = [row.get(col) for col in cycle_names]
        times = [pd.Timestamp(t) if not pd.isnull(t) else pd.NaT for t in times]
        durations_row = []
        for i in range(len(times) - 1):
            if pd.isnull(times[i]) or pd.isnull(times[i + 1]):
                durations_row.append(np.nan)
            else:
                val = (times[i + 1] - times[i]).days
                if val < 0:
                    if negative_duration_handling == "zero":
                        val = 0
                    elif negative_duration_handling == "nan":
                        val = np.nan
                    elif negative_duration_handling == "abs":
                        val = abs(val)
                durations_row.append(val)
        durations.append(durations_row)
    # Use column names (except last)
    return pd.DataFrame(durations, columns=cycle_names[:-1], index=cycle_data["key"])


class BottleneckChartsCalculator(Calculator):
    """
    Generates bottleneck visualizations: per-issue stacked bar, aggregate stacked bar,
    and box/violin plots.
    """

    def run(self):
        cycle_data = self.get_result(CycleTimeCalculator)
        cycle_names = [s["name"] for s in self.settings["cycle"]]
        negative_duration_handling = self.settings.get("negative_duration_handling", "zero")
        # Return both transition durations and per-column durations
        return {
            "transitions": calculate_column_durations(cycle_data, cycle_names),
            "columns": calculate_column_durations_per_column(
                cycle_data, cycle_names, negative_duration_handling
            ),
        }

    def write(self):
        results = self.get_result()
        # durations_transitions = results["transitions"]  # Unused
        durations_columns = results["columns"]
        output_settings = self.settings
        logger.debug(
            "[BottleneckChartsCalculator] output_settings: %s",
            pprint.pformat(output_settings),
        )
        for key in [
            "bottleneck_stacked_per_issue_chart",
            "bottleneck_stacked_aggregate_mean_chart",
            "bottleneck_stacked_aggregate_median_chart",
            "bottleneck_boxplot_chart",
            "bottleneck_violin_chart",
        ]:
            logger.debug("[BottleneckChartsCalculator] %s: %s", key, output_settings.get(key))
        # 1. Stacked bar per issue (now uses per-column durations)
        if output_settings.get("bottleneck_stacked_per_issue_chart"):
            self.write_stacked_per_issue(
                durations_columns, output_settings["bottleneck_stacked_per_issue_chart"]
            )
        # 2. Aggregated stacked bar (mean, by column)
        if output_settings.get("bottleneck_stacked_aggregate_mean_chart"):
            self.write_stacked_aggregate(
                durations_columns,
                output_settings["bottleneck_stacked_aggregate_mean_chart"],
                aggfunc="mean",
            )
        # 3. Aggregated stacked bar (median, by column)
        if output_settings.get("bottleneck_stacked_aggregate_median_chart"):
            self.write_stacked_aggregate(
                durations_columns,
                output_settings["bottleneck_stacked_aggregate_median_chart"],
                aggfunc="median",
            )
        # 4. Boxplot (by column)
        if output_settings.get("bottleneck_boxplot_chart"):
            self.write_boxplot(durations_columns, output_settings["bottleneck_boxplot_chart"])
        # 5. Violin plot (by column)
        if output_settings.get("bottleneck_violin_chart"):
            self.write_violin(durations_columns, output_settings["bottleneck_violin_chart"])

    def write_stacked_per_issue(self, durations, output_file):
        logger = logging.getLogger(__name__)
        try:
            logger.info(f"Writing bottleneck stacked per issue chart to {output_file}")
            N = 30
            plot_data = durations.dropna(how="all").iloc[:N]
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_data.plot(kind="bar", stacked=True, ax=ax)
            ax.set_xlabel("Issue key")
            ax.set_ylabel("Days in column")
            ax.set_title(f"Time spent in each column (per issue, first {N} issues)")
            plt.xticks(rotation=90)
            plt.tight_layout()
            fig.savefig(output_file, bbox_inches="tight", dpi=300)
            plt.close(fig)
            logger.info(f"Successfully wrote {output_file}")
        except Exception as e:
            logger.error(
                "Error writing bottleneck stacked per issue chart to %s: %s\n%s"
                % (output_file, e, traceback.format_exc())
            )

    def write_stacked_aggregate(self, durations, output_file, aggfunc="mean"):
        logger = logging.getLogger(__name__)
        try:
            logger.info(f"Writing bottleneck stacked aggregate chart to {output_file}")
            if aggfunc == "mean":
                agg = durations.mean(skipna=True)
                title = "Average time spent in each column (all issues)"
            else:
                agg = durations.median(skipna=True)
                title = "Median time spent in each column (all issues)"
            fig, ax = plt.subplots(figsize=(10, 5))
            agg.plot(kind="bar", stacked=False, ax=ax, color=sns.color_palette("tab10"))
            ax.set_xlabel("Column")
            ax.set_ylabel("Days")
            ax.set_title(title)
            plt.xticks(rotation=45)
            plt.tight_layout()
            fig.savefig(output_file, bbox_inches="tight", dpi=300)
            plt.close(fig)
            logger.info(f"Successfully wrote {output_file}")
        except Exception as e:
            logger.error(
                "Error writing bottleneck stacked aggregate chart to %s: %s\n%s"
                % (output_file, e, traceback.format_exc())
            )

    def write_boxplot(self, durations, output_file):
        logger = logging.getLogger(__name__)
        try:
            logger.info(f"Writing bottleneck boxplot chart to {output_file}")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=durations, ax=ax)
            ax.set_xlabel("Column")
            ax.set_ylabel("Days")
            ax.set_title("Distribution of time spent in each column (boxplot)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            fig.savefig(output_file, bbox_inches="tight", dpi=300)
            plt.close(fig)
            logger.info(f"Successfully wrote {output_file}")
        except Exception as e:
            logger.error(
                "Error writing bottleneck boxplot chart to %s: %s\n%s"
                % (output_file, e, traceback.format_exc())
            )

    def write_violin(self, durations, output_file):
        logger = logging.getLogger(__name__)
        try:
            logger.info(f"Writing bottleneck violin chart to {output_file}")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.violinplot(data=durations, ax=ax, cut=0)
            ax.set_xlabel("Column")
            ax.set_ylabel("Days")
            ax.set_title("Distribution of time spent in each column (violin plot)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            fig.savefig(output_file, bbox_inches="tight", dpi=300)
            plt.close(fig)
            logger.info(f"Successfully wrote {output_file}")
        except Exception as e:
            logger.error(
                "Error writing bottleneck violin chart to %s: %s\n%s"
                % (output_file, e, traceback.format_exc())
            )
