"""Progress report utilities for Jira Agile Metrics.

This module provides utilities for processing progress report configuration data.
"""

from typing import Any, Dict, List

from .type_utils import expand_key, force_date, force_int


def to_progress_report_teams_list(value: Any) -> List[Dict]:
    """
    Convert value to a list of progress report teams dicts.
    """
    result = []
    for val in value:
        name_key = expand_key("name")
        wip_key = expand_key("wip")
        min_throughput_key = expand_key("min_throughput")
        max_throughput_key = expand_key("max_throughput")
        throughput_samples_key = expand_key("throughput_samples")
        throughput_samples_window_key = expand_key("throughput_samples_window")

        result.append(
            {
                "name": (val[name_key] if name_key in val else None),
                "wip": (force_int("wip", val[wip_key]) if wip_key in val else 1),
                "min_throughput": (
                    force_int("min_throughput", val[min_throughput_key])
                    if min_throughput_key in val
                    else None
                ),
                "max_throughput": (
                    force_int("max_throughput", val[max_throughput_key])
                    if max_throughput_key in val
                    else None
                ),
                "throughput_samples": (
                    val[throughput_samples_key]
                    if throughput_samples_key in val
                    else None
                ),
                "throughput_samples_window": (
                    force_int(
                        "throughput_samples_window",
                        val[throughput_samples_window_key],
                    )
                    if throughput_samples_window_key in val
                    else None
                ),
            }
        )
    return result


def to_progress_report_outcomes_list(value: Any) -> List[Dict]:
    """
    Convert value to a list of progress report outcomes dicts.
    """
    result = []
    for val in value:
        name_k = expand_key("name")
        key_k = expand_key("key")
        deadline_k = expand_key("deadline")
        epic_query_k = expand_key("epic_query")

        result.append(
            {
                "name": (val[name_k] if name_k in val else None),
                "key": (val[key_k] if key_k in val else None),
                "deadline": (
                    force_date("deadline", val[deadline_k])
                    if deadline_k in val
                    else None
                ),
                "epic_query": (val[epic_query_k] if epic_query_k in val else None),
            }
        )
    return result
