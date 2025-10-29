"""Helper functions for impediment processing used by cycle time calculations."""

import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..querymanager import IssueSnapshot


@dataclass(frozen=True)
class ImpedimentConfig:
    """Read-only configuration for processing flagged impediment changes."""

    active_columns: List[str]
    impediment_flags: List[str]


@dataclass
class ImpedimentProcessingContext:
    """Processing context containing read-only config and mutable accumulated state."""

    config: ImpedimentConfig
    impediments: List[Dict[str, Any]] = field(default_factory=list)
    done_column: Optional[str] = None


def _create_status_to_column_mapping(params):
    status_to_column = {}
    for cycle_step in params.config.config["cycle"]:
        column_name = cycle_step["name"]
        for status in cycle_step["statuses"]:
            status_to_column[status] = column_name
    return status_to_column


def _end_impediment(
    impediment_state_dict,
    end_date,
    status_to_column,
    active_columns,
):
    impediment_start_date = impediment_state_dict["impediment_start_date"]
    impediment_status = impediment_state_dict["impediment_status"]
    impediment_flag = impediment_state_dict["impediment_flag"]

    impediment_days = (end_date - impediment_start_date).days
    impediment_column = status_to_column.get(impediment_status)

    impediment_record = {
        "start": impediment_start_date,
        "end": end_date,
        "status": impediment_column,
        "flag": impediment_flag,
    }

    blocked_days = 0
    if impediment_column in active_columns:
        blocked_days = impediment_days

    return impediment_record, blocked_days


def _process_status_change(
    snapshot: IssueSnapshot,
    status_to_column: Dict[str, str],
    impediment_state: Dict[str, Any],
    context: ImpedimentProcessingContext,
) -> Dict[str, Any]:
    active_columns = context.config.active_columns
    impediments = context.impediments
    done_column = context.done_column

    current_status = snapshot.to_string
    update_state = {
        "current_impediment": impediment_state["current_impediment"],
        "impediment_start_date": impediment_state["impediment_start_date"],
        "impediment_flag": impediment_state["impediment_flag"],
        "impediment_status": impediment_state["impediment_status"],
        "current_status": current_status,
        "blocked_days": impediment_state["blocked_days"],
    }

    if impediment_state["current_impediment"] is not None:
        configured_done = (done_column or "Done").strip().casefold()
        current_column = status_to_column.get(current_status, current_status)
        if str(current_column).strip().casefold() == configured_done:
            impediment_end_date = snapshot.date.date()
            impediment_state_subset = {
                "impediment_start_date": impediment_state["impediment_start_date"],
                "impediment_status": impediment_state["impediment_status"],
                "impediment_flag": impediment_state["impediment_flag"],
            }
            record, blocked = _end_impediment(
                impediment_state_subset,
                impediment_end_date,
                status_to_column,
                active_columns,
            )
            impediments.append(record)
            update_state["blocked_days"] = impediment_state["blocked_days"] + blocked
            update_state["current_impediment"] = None
            update_state["impediment_start_date"] = None
            update_state["impediment_flag"] = None
            update_state["impediment_status"] = None

    return update_state


def _process_flagged_change(
    snapshot: IssueSnapshot,
    status_to_column: Dict[str, str],
    impediment_state: Dict[str, Any],
    context: ImpedimentProcessingContext,
) -> Dict[str, Any]:
    flag_value = snapshot.to_string
    update_state = {
        "current_impediment": impediment_state["current_impediment"],
        "impediment_start_date": impediment_state["impediment_start_date"],
        "impediment_flag": impediment_state["impediment_flag"],
        "impediment_status": impediment_state["impediment_status"],
        "current_status": impediment_state["current_status"],
        "blocked_days": impediment_state["blocked_days"],
    }

    active_columns = context.config.active_columns
    impediments = context.impediments
    impediment_flags = context.config.impediment_flags

    if flag_value and flag_value in impediment_flags:
        if impediment_state["current_impediment"] is None:
            update_state["current_impediment"] = flag_value
            update_state["impediment_start_date"] = snapshot.date.date()
            update_state["impediment_flag"] = flag_value
            update_state["impediment_status"] = impediment_state["current_status"]
    else:
        if impediment_state["current_impediment"] is not None:
            impediment_end_date = snapshot.date.date()
            impediment_state_subset = {
                "impediment_start_date": impediment_state["impediment_start_date"],
                "impediment_status": impediment_state["impediment_status"],
                "impediment_flag": impediment_state["impediment_flag"],
            }
            record, blocked = _end_impediment(
                impediment_state_subset,
                impediment_end_date,
                status_to_column,
                active_columns,
            )
            impediments.append(record)
            update_state["blocked_days"] = impediment_state["blocked_days"] + blocked
            update_state["current_impediment"] = None
            update_state["impediment_start_date"] = None
            update_state["impediment_flag"] = None
            update_state["impediment_status"] = None

    return update_state


def _process_impediments(
    issue: Any, item: Dict[str, Any], params: Any, active_columns: List[str]
) -> Dict[str, Any]:
    impediments = []
    blocked_days = 0

    status_to_column = _create_status_to_column_mapping(params)

    impediment_state = {
        "current_impediment": None,
        "impediment_start_date": None,
        "impediment_flag": None,
        "impediment_status": None,
        "current_status": None,
        "blocked_days": 0,
    }

    # Get impediment_flags from config or use default
    impediment_flags = params.config.config.get(
        "impediment_flags", ["Impediment", "Awaiting input"]
    )

    # Create impediment config and context once before the loop
    impediment_config = ImpedimentConfig(
        active_columns=active_columns,
        impediment_flags=impediment_flags,
    )
    impediment_context = ImpedimentProcessingContext(
        config=impediment_config,
        impediments=impediments,
        done_column=params.config.config.get("done_column", None),
    )

    now = params.config.config["now"]
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)

    for snapshot in params.query_manager.iter_changes(issue, ["status", "Flagged"]):
        if snapshot.change == "status":
            impediment_state = _process_status_change(
                snapshot,
                status_to_column,
                impediment_state,
                impediment_context,
            )
            blocked_days = impediment_state["blocked_days"]
        elif snapshot.change == "Flagged":
            impediment_state = _process_flagged_change(
                snapshot,
                status_to_column,
                impediment_state,
                impediment_context,
            )
            blocked_days = impediment_state["blocked_days"]

    if impediment_state["current_impediment"] is not None:
        impediment_column = status_to_column.get(impediment_state["impediment_status"])
        impediments.append(
            {
                "start": impediment_state["impediment_start_date"],
                "end": None,
                "status": impediment_column,
                "flag": impediment_state["impediment_flag"],
            }
        )

        if impediment_column in active_columns:
            blocked_days += (
                now.date() - impediment_state["impediment_start_date"]
            ).days

    item["impediments"] = impediments
    item["blocked_days"] = blocked_days
    return item
