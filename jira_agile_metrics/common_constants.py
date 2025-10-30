"""Common constants used across Jira Agile Metrics modules."""

from typing import Dict, Final, List

# Common bottleneck chart settings list
BOTTLENECK_CHART_SETTINGS: Final[List[str]] = [
    "bottleneck_stacked_per_issue_chart",
    "bottleneck_stacked_aggregate_mean_chart",
    "bottleneck_stacked_aggregate_median_chart",
    "bottleneck_boxplot_chart",
    "bottleneck_violin_chart",
]

# Common issue data pattern for add_issue_data_to_series
COMMON_ISSUE_DATA_PATTERN: Final[Dict[str, str]] = {
    "key": "key",
    "priority": "priority",
    "type": "Issue type",
    "environment": "environment",
    "created": "created",
    "resolved": "resolved",
}

# Common cycle time fields used across multiple modules
COMMON_CYCLE_TIME_FIELDS: Final[List[str]] = [
    "cycle_time",
    "lead_time",
    "completed_timestamp",
    "blocked_days",
    "impediments",
]
