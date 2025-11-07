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

# Chart filename keys used in config parsing and webapp helpers
CHART_FILENAME_KEYS: Final[List[str]] = [
    "scatterplot_chart",
    "histogram_chart",
    "cfd_chart",
    "throughput_chart",
    "burnup_chart",
    "burnup_forecast_chart",
    "wip_chart",
    "ageing_wip_chart",
    "net_flow_chart",
    "impediments_chart",
    "impediments_days_chart",
    "impediments_status_chart",
    "impediments_status_days_chart",
    "defects_by_priority_chart",
    "defects_by_type_chart",
    "defects_by_environment_chart",
    "debt_chart",
    "debt_age_chart",
    "waste_chart",
    "progress_report",
]

# Data filename keys used in config parsing and webapp helpers
DATA_FILENAME_KEYS: Final[List[str]] = [
    "cycle_time_data",
    "cfd_data",
    "scatterplot_data",
    "histogram_data",
    "throughput_data",
    "percentiles_data",
    "impediments_data",
    "burnup_forecast_chart_data",
    "lead_time_histogram_data",
    "lead_time_histogram_chart",
]
