"""Common constants used across Jira Agile Metrics modules."""


def get_bottleneck_chart_settings():
    """Get common bottleneck chart settings list."""
    return [
        "bottleneck_stacked_per_issue_chart",
        "bottleneck_stacked_aggregate_mean_chart",
        "bottleneck_stacked_aggregate_median_chart",
        "bottleneck_boxplot_chart",
        "bottleneck_violin_chart",
    ]


def get_common_issue_data_pattern():
    """Get common issue data pattern for add_issue_data_to_series."""
    return {
        "key": "key",
        "priority": "priority",
        "type": "Issue type",
        "environment": "environment",
        "created": "created",
        "resolved": "resolved",
    }


def get_common_cycle_time_fields():
    """Get common cycle time fields used across multiple modules."""
    return [
        "cycle_time",
        "lead_time",
        "completed_timestamp",
        "blocked_days",
        "impediments",
    ]
