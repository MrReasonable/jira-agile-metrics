"""Team throughput and sampler functions for progress reports.

This module contains functions for calculating team throughput and managing
throughput samplers for Monte Carlo simulations.
"""

import logging
import random

from .cycletime import CycleTimeParams, calculate_cycle_times
from .forecast_utils import throughput_sampler

logger = logging.getLogger(__name__)


def escape_jql_string(value):
    """Escape a string value for use in JQL queries to prevent injection.

    Args:
        value: The string value to escape. None is accepted and will be
               converted to an empty string.

    Returns:
        Escaped string safe for use in JQL double-quoted strings. Returns
        empty string if value is None, or returns value unchanged if it's
        already an empty string.

    Examples:
        >>> escape_jql_string('Team "Special"')
        'Team \\"Special\\"'
        >>> escape_jql_string('Team\\Backslash')
        'Team\\\\Backslash'
        >>> escape_jql_string(None)
        ''
        >>> escape_jql_string('')
        ''
    """
    if value is None:
        return ""
    if not value:
        return value
    # Escape backslashes first, then quotes
    # This prevents injection via quotes or backslashes in team names
    return value.replace("\\", "\\\\").replace('"', '\\"')


def throughput_range_sampler(min_val, max_val):
    """Create a sampler that returns values between min_val and max_val."""
    if min_val is None or max_val is None:
        return None

    def sampler():
        return random.uniform(min_val, max_val)

    return sampler


def _get_throughput_config(team, config):
    """Get throughput config from team or construct from global config.

    Args:
        team: Team object (dict or object with throughput_config attribute)
        config: Configuration dict or object with get method

    Returns:
        dict: Throughput configuration with throughput_samples, min_throughput,
            and max_throughput keys. Returns team.throughput_config if available,
            otherwise constructs a dict using config defaults.
    """
    if hasattr(team, "throughput_config"):
        return team.throughput_config

    # If team is a dict, create a mock throughput_config
    # Default throughput values are configurable via config keys
    min_throughput = (
        config.get("default_min_throughput", 0) if isinstance(config, dict) else 0
    )
    max_throughput = (
        config.get("default_max_throughput", 10) if isinstance(config, dict) else 10
    )
    return {
        "throughput_samples": [],
        "min_throughput": min_throughput,
        "max_throughput": max_throughput,
    }


def update_team_sampler(team, query_manager, config):
    """Update team sampler based on configuration.

    When team is a dict, default min_throughput and max_throughput values
    are read from config keys 'default_min_throughput' and 'default_max_throughput'
    respectively, with fallbacks to 0 and 10.
    """
    throughput_config = _get_throughput_config(team, config)

    if throughput_config.get("throughput_samples"):
        # Use throughput samples
        cycle_times = calculate_cycle_times(
            CycleTimeParams(
                query_manager,
                config,
                [throughput_config["throughput_samples"]],
            )
        )
        throughput_config["throughput_samples_cycle_times"] = cycle_times
        throughput_config["sampler"] = throughput_sampler(cycle_times)
    else:
        # Use min/max throughput
        throughput_config["sampler"] = throughput_range_sampler(
            throughput_config.get("min_throughput", 0),
            throughput_config.get("max_throughput", 10),
        )


def calculate_team_throughput(team, query_manager, config):
    """Calculate team throughput based on configuration.

    Returns team throughput in one of two ways depending on configuration:

    1. **With throughput_samples**: Calculates actual historical throughput by
       fetching issues, computing cycle times, and analyzing completion data.
       Returns the cycle time data.

    2. **With min_throughput/max_throughput** (range-based): Returns the midpoint
       ``(min_throughput + max_throughput) / 2`` as a static estimate, even when
       historical data exists. This is a design choice made for:
       - Performance: Avoiding expensive full data analysis
       - Configuration intent: Using a simple range-based estimate
       - Consistency: Aligning with the probabilistic range sampler
       - Efficiency: Leveraging lightweight data availability checks

    When team is a dict, default min_throughput and max_throughput values
    are read from config keys 'default_min_throughput' and 'default_max_throughput'
    respectively, with fallbacks to 0 and 10.

    Args:
        team: Team object with optional throughput_config or name attribute
        query_manager: QueryManager instance for checking data availability
        config: Configuration dict with defaults and query templates

    Returns:
        Cycle time data (when using throughput_samples) or float midpoint
        estimate (when using min/max range), or 0 if no data is available.

    Note:
        When using min/max throughput, ``has_issues_for_jql()`` confirms that
        historical data exists (by fetching one issue with maxResults=1), but
        the function does not analyze the actual historical throughput. Instead,
        it returns the midpoint of the configured range. To get actual historical
        throughput, configure ``throughput_samples`` in the team configuration.
    """
    throughput_config = _get_throughput_config(team, config)

    if throughput_config.get("throughput_samples"):
        # Use throughput samples
        cycle_times = calculate_cycle_times(
            CycleTimeParams(
                query_manager,
                config,
                [throughput_config["throughput_samples"]],
            )
        )
        throughput_config["throughput_samples_cycle_times"] = cycle_times
        throughput_config["sampler"] = throughput_sampler(cycle_times)
        return cycle_times

    # Use min/max throughput
    min_throughput = throughput_config.get("min_throughput", 0)
    max_throughput = throughput_config.get("max_throughput", 10)
    throughput_config["sampler"] = throughput_range_sampler(
        min_throughput, max_throughput
    )

    # Check if there's actual data available using a team-scoped JQL query
    # Return 0 as a clear sentinel when no data is available
    # Build a team-scoped JQL query
    # Prefer team's throughput_samples query if available (even if not using it),
    # otherwise try to use story query template from config,
    # or fall back to basic query
    team_name = (
        team.name
        if hasattr(team, "name")
        else (team.get("name") if isinstance(team, dict) else None)
    )
    team_jql = None

    # Option 1: Use team's throughput_samples JQL if available (team-scoped)
    if throughput_config.get("throughput_samples"):
        team_jql = throughput_config["throughput_samples"]
        if team_name and "{team}" in team_jql:
            # Escape quotes in team name to prevent JQL injection
            escaped_name = escape_jql_string(team_name)
            team_jql = team_jql.format(team=f'"{escaped_name}"')
    # Option 2: Use story query template from config if available
    elif (
        isinstance(config, dict)
        and config.get("progress_report_story_query_template")
        and team_name
    ):
        story_template = config["progress_report_story_query_template"]
        if "{team}" in story_template:
            escaped_name = escape_jql_string(team_name)
            team_jql = story_template.format(team=f'"{escaped_name}"')
        else:
            # If template doesn't use {team}, try appending team filter
            # This is a fallback - may not work for all JIRA setups
            # Note: Hardcoded "Team" field assumes specific JIRA configuration
            escaped_name = escape_jql_string(team_name)
            team_jql = f'{story_template} AND Team = "{escaped_name}"'
    else:
        # No team-scoped query available - return 0 to indicate no data
        # rather than querying all issues
        logger.debug(
            "No team-scoped query available for team %s, returning 0", team_name
        )
        return 0

    # Use the safe has_issues_for_jql method to check data availability
    # without mutating global settings
    if query_manager.has_issues_for_jql(team_jql):
        # Return midpoint of min/max throughput as a static estimate.
        #
        # Note: Although has_issues_for_jql confirms that historical data exists
        # for this team, we use the midpoint of the configured min/max throughput
        # range rather than calculating actual historical throughput. This design
        # decision is made for several reasons:
        #
        # 1. Performance: Calculating actual throughput requires fetching and
        #    processing all relevant issues, determining cycle times, grouping by
        #    completion periods, and aggregating - which is computationally
        #    expensive for large datasets.
        #
        # 2. Configuration intent: When min_throughput/max_throughput are configured
        #    without throughput_samples, it indicates the user wants to use a
        #    simple range-based estimate rather than historical analysis.
        #
        # 3. Consistent with sampler behavior: The throughput sampler for this team
        #    (created above) uses throughput_range_sampler, which samples uniformly
        #    between min/max. Using the midpoint here provides a deterministic
        #    point estimate that aligns with the probabilistic range used in
        #    simulations.
        #
        # 4. Lightweight check: has_issues_for_jql only fetches one issue (maxResults=1)
        #    to verify data availability. Using this as a quick validation without
        #    expensive full data analysis keeps this function efficient.
        #
        # For teams that need actual historical throughput, configure throughput_samples
        # in the team configuration, which will trigger full cycle time calculation
        # and actual throughput analysis.
        return (min_throughput + max_throughput) / 2
    # No issues available, return 0 as clear sentinel for no data
    return 0
