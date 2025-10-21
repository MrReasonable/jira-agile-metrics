from typing import Any, Dict, List

from .type_utils import expand_key, force_date, force_int


def to_progress_report_teams_list(value: Any) -> List[Dict]:
    """
    Convert value to a list of progress report teams dicts.
    """
    return [
        {
            "name": (
                val[expand_key("name")] if expand_key("name") in val else None
            ),
            "wip": (
                force_int("wip", val[expand_key("wip")])
                if expand_key("wip") in val
                else 1
            ),
            "min_throughput": (
                force_int("min_throughput", val[expand_key("min_throughput")])
                if expand_key("min_throughput") in val
                else None
            ),
            "max_throughput": (
                force_int("max_throughput", val[expand_key("max_throughput")])
                if expand_key("max_throughput") in val
                else None
            ),
            "throughput_samples": (
                val[expand_key("throughput_samples")]
                if expand_key("throughput_samples") in val
                else None
            ),
            "throughput_samples_window": (
                force_int(
                    "throughput_samples_window",
                    val[expand_key("throughput_samples_window")],
                )
                if expand_key("throughput_samples_window") in val
                else None
            ),
        }
        for val in value
    ]


def to_progress_report_outcomes_list(value: Any) -> List[Dict]:
    """
    Convert value to a list of progress report outcomes dicts.
    """
    return [
        {
            "name": (
                val[expand_key("name")] if expand_key("name") in val else None
            ),
            "key": (
                val[expand_key("key")] if expand_key("key") in val else None
            ),
            "deadline": (
                force_date("deadline", val[expand_key("deadline")])
                if expand_key("deadline") in val
                else None
            ),
            "epic_query": (
                val[expand_key("epic_query")]
                if expand_key("epic_query") in val
                else None
            ),
        }
        for val in value
    ]
