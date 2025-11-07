"""Cycle time calculator for Jira Agile Metrics.

This module provides functionality to calculate cycle time metrics from JIRA data.
"""

import datetime
import json
import logging

import pandas as pd

from ..calculator import Calculator
from ..common_constants import COMMON_CYCLE_TIME_FIELDS
from ..trello import TrelloClient
from ..utils import get_extension, to_json_string
from .impediments_utils import _process_impediments

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
        config = CycleTimeConfig(
            {
                "cycle": self.settings["cycle"],
                "attributes": self.settings["attributes"],
                "committed_column": self.settings["committed_column"],
                "done_column": self.settings["done_column"],
                "query_attribute": self.settings["query_attribute"],
                "lead_time_start_column": self.settings.get("lead_time_start_column"),
                "now": now,
                "reset_on_backwards": self.settings.get("reset_on_backwards", True),
                "keep_first_entry_time": self.settings.get(
                    "keep_first_entry_time", True
                ),
            }
        )
        params = CycleTimeParams(
            query_manager=self.query_manager,
            config=config,
            queries=self.settings["queries"],
        )
        return calculate_cycle_times(params)

    def write(self):
        output_files = self.settings.get("cycle_time_data", [])

        if not output_files:
            logger.debug("No output file specified for cycle time data")
            return

        cycle_data = self.get_result()
        cycle_names = [s["name"] for s in self.settings["cycle"]]
        attribute_names = sorted(self.settings["attributes"].keys())
        query_attribute_names = (
            [self.settings["query_attribute"]]
            if self.settings["query_attribute"]
            else []
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
                    list(map(to_json_string, row))
                    for row in cycle_data[columns].values.tolist()
                ]
                with open(output_file, "w", encoding="utf-8") as out:
                    out.write(json.dumps(values))
            elif output_extension == ".xlsx":
                cycle_data.to_excel(
                    output_file,
                    sheet_name="Cycle data",
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


def calculate_cycle_times(params):
    """Calculate cycle times for issues based on status transitions.

    Args:
        params: CycleTimeParams object containing all necessary parameters

    Returns:
        DataFrame with cycle time calculations
    """
    # Allows unit testing to use a fixed date
    if params.config.config["now"] is None:
        params.config.config["now"] = datetime.datetime.now(datetime.timezone.utc)

    cycle_names = [s["name"] for s in params.config.config["cycle"]]
    if params.config.config["lead_time_start_column"] is None:
        params.config.config["lead_time_start_column"] = cycle_names[0]
    active_columns = cycle_names[
        cycle_names.index(params.config.config["committed_column"]) : cycle_names.index(
            params.config.config["done_column"]
        )
    ]

    cycle_lookup = _build_cycle_lookup(params.config.config["cycle"])
    unmapped_statuses = set()

    series = _initialize_series(
        params.config.config["cycle"],
        params.config.config["attributes"],
        params.config.config["query_attribute"],
    )

    total_issues_found = 0
    for criteria in params.queries:
        issues = params.query_manager.find_issues(criteria["jql"])
        issues_count = len(issues)
        total_issues_found += issues_count

        if issues_count == 0:
            logger.warning(
                "No issues found for query: %s. "
                "Check your JQL query, authentication, "
                "and ensure there are matching issues in JIRA.",
                criteria["jql"],
            )

        for issue in issues:
            processing_params = IssueProcessingParams(
                {
                    "criteria": criteria,
                    "params": params,
                    "cycle_names": cycle_names,
                    "cycle_lookup": cycle_lookup,
                    "unmapped_statuses": unmapped_statuses,
                    "active_columns": active_columns,
                }
            )
            item = _process_single_issue(issue, processing_params)
            for k, v in item.items():
                series[k]["data"].append(v)

    if total_issues_found == 0:
        logger.warning(
            "No issues were found for any query. "
            "This will result in empty CSV files and no charts. "
            "Please verify: "
            "1. JIRA credentials are correct (check .env file or config.yml) "
            "2. The JQL query matches issues in your JIRA instance "
            "3. You have permission to view the issues"
        )

    if len(unmapped_statuses) > 0:
        logger.warning(
            (
                "The following JIRA statuses were found, "
                "but not mapped to a workflow state, "
                "and have been ignored: %s"
            ),
            ", ".join(sorted(unmapped_statuses)),
        )

    return _build_result_dataframe(series, params, cycle_names)


class CycleTimeConfig:
    """Configuration for cycle time calculation."""

    def __init__(self, config_dict):
        """Initialize with a dictionary of configuration parameters."""
        self.config = {
            "cycle": config_dict.get("cycle"),
            "attributes": config_dict.get("attributes"),
            "committed_column": config_dict.get("committed_column"),
            "done_column": config_dict.get("done_column"),
            "query_attribute": config_dict.get("query_attribute"),
            "lead_time_start_column": config_dict.get("lead_time_start_column"),
            "now": config_dict.get("now"),
            "reset_on_backwards": config_dict.get("reset_on_backwards", True),
            "keep_first_entry_time": config_dict.get("keep_first_entry_time", True),
        }

    def get_config(self):
        """Get a copy of the configuration dictionary.

        This is the canonical method for accessing the configuration.
        Returns a copy to prevent external modification of internal state.
        """
        return self.config.copy()

    def get_config_dict(self):
        """Get configuration as a dictionary.

        Deprecated: Use get_config() instead. This method is maintained
        for backward compatibility only.
        """
        return self.get_config()

    def __str__(self):
        """Return string representation."""
        return (
            f"CycleTimeConfig(done_column={self.config['done_column']}, "
            f"committed_column={self.config['committed_column']})"
        )

    def __repr__(self):
        return (
            f"CycleTimeConfig(cycle={self.config['cycle']}, "
            f"committed_column='{self.config['committed_column']}', "
            f"done_column='{self.config['done_column']}')"
        )


class CycleTimeParams:
    """Parameters for cycle time calculation."""

    def __init__(self, query_manager, config, queries):
        self.query_manager = query_manager
        self.config = config
        self.queries = queries

    def __repr__(self):
        return (
            f"CycleTimeParams(query_manager={self.query_manager}, "
            f"config={self.config}, queries={len(self.queries)} queries)"
        )

    def get_queries(self):
        """Get the queries list."""
        return self.queries


def _build_cycle_lookup(cycle):
    """Build lookup dictionary for cycle statuses."""
    cycle_lookup = {}
    for idx, cycle_step in enumerate(cycle):
        for status in cycle_step["statuses"]:
            cycle_lookup[status.lower()] = {
                "index": idx,
                "name": cycle_step["name"],
            }
    return cycle_lookup


def _initialize_series(cycle, attributes, query_attribute):
    """Initialize series dictionary for data collection."""
    cycle_names = [s["name"] for s in cycle]
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
        "impediments": {"data": [], "dtype": "object"},
    }

    for cycle_name in cycle_names:
        series[cycle_name] = {"data": [], "dtype": "datetime64[ns]"}

    for name in attributes:
        series[name] = {"data": [], "dtype": "object"}

    if query_attribute:
        series[query_attribute] = {"data": [], "dtype": "str"}

    return series


class IssueProcessingParams:
    """Parameters for processing a single issue."""

    def __init__(self, params_dict):
        """Initialize with a dictionary of parameters."""
        self.criteria = params_dict.get("criteria")
        self.params = params_dict.get("params")
        self.cycle_names = params_dict.get("cycle_names")
        self.cycle_lookup = params_dict.get("cycle_lookup")
        self.unmapped_statuses = params_dict.get("unmapped_statuses")
        self.active_columns = params_dict.get("active_columns")

    def get_params_dict(self):
        """Get parameters as a dictionary."""
        return {
            "criteria": self.criteria,
            "params": self.params,
            "cycle_names": self.cycle_names,
            "cycle_lookup": self.cycle_lookup,
            "unmapped_statuses": self.unmapped_statuses,
            "active_columns": self.active_columns,
        }

    def __str__(self):
        """Return string representation."""
        return f"IssueProcessingParams(criteria={self.criteria})"

    def __repr__(self):
        return (
            f"IssueProcessingParams(criteria={self.criteria}, "
            f"cycle_names={self.cycle_names})"
        )

    def get_criteria(self):
        """Get the criteria."""
        return self.criteria


def _process_single_issue(issue, processing_params):
    """Process a single issue and return its data."""
    item = _create_base_item(
        issue,
        processing_params.criteria,
        processing_params.params,
        processing_params.cycle_names,
    )
    status_params = StatusChangeParams(
        processing_params.params,
        processing_params.cycle_lookup,
        processing_params.unmapped_statuses,
        processing_params.active_columns,
    )
    item = _process_status_changes(issue, item, status_params)
    item = _process_impediments(
        issue, item, processing_params.params, processing_params.active_columns
    )
    item = _calculate_times(
        item, processing_params.cycle_names, processing_params.params
    )
    return item


def _extract_resolution_value(issue_fields):
    """Extract a readable resolution value from an issue's fields.

    Returns the resolution name when available, otherwise a string cast of the
    resolution object, or None if no resolution is present.
    """
    resolution = getattr(issue_fields, "resolution", None)
    if resolution is None:
        return None
    if hasattr(resolution, "name"):
        return resolution.name
    if isinstance(resolution, dict):
        return resolution.get("name")
    return str(resolution)


def _extract_name(value):
    """Extract a readable name from JIRA objects or dicts.

    Handles PropertyHolder objects with a `name` attribute as well as
    plain dictionaries like `{ 'name': 'Story' }` that may come from
    file-backed fixtures.
    """
    if value is None:
        return None
    if hasattr(value, "name"):
        return value.name
    if isinstance(value, dict):
        return value.get("name")
    return str(value)


def _create_base_item(issue, criteria, params, cycle_names):
    """Create base item dictionary for an issue."""
    if isinstance(params.query_manager.jira, TrelloClient):
        issue_url = issue.url
    else:
        jira_client = params.query_manager.jira
        # Use the standard JIRA client options dict for the server URL
        base_server_url = getattr(jira_client, "_options", {}).get("server", "")
        issue_url = f"{base_server_url}/browse/{issue.key}"
    item = {
        "key": issue.key,
        "url": issue_url,
        "issue_type": _extract_name(issue.fields.issuetype),
        "summary": issue.fields.summary,
        "status": _extract_name(issue.fields.status),
        "resolution": _extract_resolution_value(issue.fields),
        "cycle_time": None,
        "lead_time": None,
        "completed_timestamp": None,
        "blocked_days": 0,
        "impediments": [],
    }

    for name in params.config.config["attributes"]:
        item[name] = params.query_manager.resolve_attribute_value(issue, name)

    if params.config.config["query_attribute"]:
        item[params.config.config["query_attribute"]] = criteria.get("value", None)

    for cycle_name in cycle_names:
        item[cycle_name] = None

    return item


class StatusChangeParams:
    """Parameters for processing status changes."""

    def __init__(self, params, cycle_lookup, unmapped_statuses, active_columns):
        self.params = params
        self.cycle_lookup = cycle_lookup
        self.unmapped_statuses = unmapped_statuses
        self.active_columns = active_columns

    def __repr__(self):
        return (
            f"StatusChangeParams(cycle_lookup={len(self.cycle_lookup)} statuses, "
            f"active_columns={self.active_columns})"
        )

    def get_active_columns(self):
        """Get the active columns."""
        return self.active_columns


def _process_status_changes(issue, item, status_params):
    """Process status changes for an issue."""
    last_status = None

    for snapshot in status_params.params.query_manager.iter_changes(
        issue, ["status", "Flagged"]
    ):
        if snapshot.change == "status":
            last_status = _process_status_change_legacy(
                snapshot,
                item,
                status_params.params,
                status_params.cycle_lookup,
                status_params.unmapped_statuses,
            )
        elif snapshot.change == "Flagged":
            _process_flagged_change_legacy(
                snapshot, item, last_status, status_params.active_columns
            )

    return item


def _process_status_change_legacy(
    snapshot, item, params, cycle_lookup, unmapped_statuses
):
    """Process a single status change (legacy version for cycle time processing)."""
    snapshot_cycle_step = cycle_lookup.get(snapshot.to_string.lower(), None)
    if snapshot_cycle_step is None:
        logger.info(
            "Issue %s transitioned to unknown JIRA status %s",
            snapshot.key,
            snapshot.to_string,
        )
        unmapped_statuses.add(snapshot.to_string)
        return None

    last_status = snapshot_cycle_step_name = snapshot_cycle_step["name"]

    # Record entry time based on configuration
    # Default behavior: keep first entry time (important for cycle time)
    if not params.config.config["keep_first_entry_time"]:
        # Use the latest observed timestamp
        item[snapshot_cycle_step_name] = snapshot.date.date()
    elif item[snapshot_cycle_step_name] is None:
        # Keep the first time we entered a step
        item[snapshot_cycle_step_name] = snapshot.date.date()

    # Wipe any subsequent dates, in case this was a move backwards
    if params.config.config["reset_on_backwards"]:
        _reset_subsequent_dates(
            item,
            snapshot_cycle_step_name,
            cycle_lookup,
            snapshot.from_string,
            snapshot.to_string,
        )

    return last_status


def _reset_subsequent_dates(
    item, snapshot_cycle_step_name, cycle_lookup, from_string, to_string
):
    """Reset subsequent dates when moving backwards.

    Args:
        item: The issue data item
        snapshot_cycle_step_name: Name of the current cycle step
        cycle_lookup: Dictionary mapping status names to cycle steps
        from_string: JIRA status name before the transition
        to_string: JIRA status name after the transition
    """
    # Get cycle names in the correct order from the cycle_lookup
    # We need to sort by index to maintain the cycle order
    cycle_steps = sorted(cycle_lookup.values(), key=lambda x: x["index"])
    cycle_names = [step["name"] for step in cycle_steps]

    found_cycle_name = False
    for cycle_name in cycle_names:
        if not found_cycle_name and cycle_name == snapshot_cycle_step_name:
            found_cycle_name = True
            continue
        if found_cycle_name and item[cycle_name] is not None:
            # Only log if we have actual JIRA status values
            if from_string is not None and to_string is not None:
                logger.info(
                    (
                        "Issue %s moved backwards to %s "
                        "[JIRA: %s -> %s], wiping data "
                        "for subsequent step %s"
                    ),
                    item["key"],
                    snapshot_cycle_step_name,
                    from_string,
                    to_string,
                    cycle_name,
                )
            else:
                logger.info(
                    (
                        "Issue %s moved backwards to %s, "
                        "wiping data for subsequent step %s"
                    ),
                    item["key"],
                    snapshot_cycle_step_name,
                    cycle_name,
                )
            item[cycle_name] = None


def _process_flagged_change_legacy(snapshot, item, last_status, active_columns):
    """Process flagged/impediment changes (legacy version for cycle time processing)."""
    # NOTE: Impediment tracking is not yet implemented
    # This is a placeholder for the full impediment processing logic
    _ = snapshot, item, last_status, active_columns  # Suppress unused argument warnings


# Impediment processing helpers moved to `impediments.py`


def _calculate_times(item, cycle_names, params):
    """Calculate cycle time and lead time for an issue."""
    previous_timestamp = None
    committed_timestamp = None
    done_timestamp = None
    lead_time_start_timestamp = None

    for cycle_name in reversed(cycle_names):
        if item[cycle_name] is not None:
            previous_timestamp = item[cycle_name]

        if previous_timestamp is not None:
            item[cycle_name] = previous_timestamp
            if cycle_name == params.config.config["done_column"]:
                done_timestamp = previous_timestamp
            if cycle_name == params.config.config["committed_column"]:
                committed_timestamp = previous_timestamp
            if cycle_name == params.config.config["lead_time_start_column"]:
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

    return item


def _build_result_dataframe(series, params, cycle_names):
    """Build the final result DataFrame."""
    data = {}
    for k, v in series.items():
        data[k] = pd.Series(v["data"], dtype=v["dtype"])

    return pd.DataFrame(
        data,
        columns=["key", "url", "issue_type", "summary", "status", "resolution"]
        + COMMON_CYCLE_TIME_FIELDS
        + sorted(params.config.config["attributes"].keys())
        + (
            [params.config.config["query_attribute"]]
            if params.config.config["query_attribute"]
            else []
        )
        + cycle_names,
    )
