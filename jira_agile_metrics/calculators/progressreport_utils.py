"""Utility functions for progress report calculator.

This module re-exports utility functions from specialized modules for
backward compatibility. The module has been refactored into smaller
focused modules:
- progressreport_teams.py: Team throughput functions
- progressreport_queries.py: Query functions
- progressreport_forecasting.py: Forecasting functions
- progressreport_plotting.py: Chart plotting functions
"""

# Import and re-export all functions for backward compatibility
from .progressreport_forecasting import (
    calculate_epic_target,
    forecast_to_complete,
    forward_weeks,
)
from .progressreport_plotting import (
    plot_cfd,
    plot_scatterplot,
    plot_throughput,
)
from .progressreport_queries import (
    date_value,
    find_epics,
    find_outcomes,
    int_or_none,
    update_story_counts,
)
from .progressreport_teams import (
    calculate_team_throughput,
    throughput_range_sampler,
    update_team_sampler,
)

__all__ = [
    # Query functions
    "find_outcomes",
    "find_epics",
    "update_story_counts",
    "int_or_none",
    "date_value",
    # Team functions
    "throughput_range_sampler",
    "update_team_sampler",
    "calculate_team_throughput",
    # Forecasting functions
    "forecast_to_complete",
    "calculate_epic_target",
    "forward_weeks",
    # Plotting functions
    "plot_cfd",
    "plot_throughput",
    "plot_scatterplot",
]
