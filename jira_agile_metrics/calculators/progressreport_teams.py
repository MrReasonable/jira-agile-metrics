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

    if min_val > max_val:
        raise ValueError(f"min_val ({min_val}) must be <= max_val ({max_val})")

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


def _build_team_jql(team, config, throughput_config):
    """Build team-scoped JQL query for data availability check.

    Args:
        team: Team object (dict or object with name attribute)
        config: Configuration dict with progress_report.templates.story
        throughput_config: Throughput configuration dict with throughput_samples

    Returns:
        str or None: Team-scoped JQL query (or non-scoped if throughput_samples
                     exists but isn't team-scoped), or None if unavailable.
    """
    team_name = (
        team.name
        if hasattr(team, "name")
        else (team.get("name") if isinstance(team, dict) else None)
    )

    # Option 1: Use team's throughput_samples JQL if available (team-scoped)
    if throughput_config.get("throughput_samples"):
        team_jql = throughput_config["throughput_samples"]
        if team_name and "{team}" in team_jql:
            # Escape quotes in team name to prevent JQL injection
            escaped_name = escape_jql_string(team_name)
            return team_jql.format(team=f'"{escaped_name}"')
        # If throughput_samples exists but isn't team-scoped, return as-is
        # (original behavior allows non-team-scoped queries)
        return team_jql

    # Option 2: Use story query template from config if available
    if isinstance(config, dict) and team_name:
        pr_config = config.get("progress_report", {})
        story_template = (
            pr_config.get("templates", {}).get("story") if pr_config else None
        )
    else:
        story_template = None

    if story_template:
        escaped_name = escape_jql_string(team_name)
        if "{team}" in story_template:
            return story_template.format(team=f'"{escaped_name}"')
        # If template doesn't use {team}, try appending team filter
        # This is a fallback - may not work for all JIRA setups
        # Note: Hardcoded "Team" field assumes specific JIRA configuration
        return f'{story_template} AND Team = "{escaped_name}"'

    return None


def update_team_sampler(team, query_manager, config):
    """Update team sampler based on configuration.

    When team is a dict, default min_throughput and max_throughput values
    are read from config keys 'default_min_throughput' and 'default_max_throughput'
    respectively, with fallbacks to 0 and 10.
    """
    throughput_config = _get_throughput_config(team, config)

    # Get team identifier for error messages
    team_name = (
        team.name
        if hasattr(team, "name")
        else (team.get("name") if isinstance(team, dict) else None)
    )
    team_identifier = team_name if team_name else "unknown team"

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
        # Validate sampler was created successfully
        if not throughput_config["sampler"]:
            logger.error(
                "Failed to create sampler for team '%s' using throughput_samples: %s",
                team_identifier,
                throughput_config["throughput_samples"],
            )
            raise ValueError(
                f"Sampler creation failed for team '{team_identifier}': "
                f"throughput_sampler returned None/False for throughput_samples: "
                f"{throughput_config['throughput_samples']}"
            )
    else:
        # Use min/max throughput
        min_throughput = throughput_config.get("min_throughput", 0)
        max_throughput = throughput_config.get("max_throughput", 10)
        throughput_config["sampler"] = throughput_range_sampler(
            min_throughput,
            max_throughput,
        )
        # Validate sampler was created successfully
        if not throughput_config["sampler"]:
            logger.error(
                (
                    "Failed to create sampler for team '%s' "
                    "using min_throughput=%s, max_throughput=%s"
                ),
                team_identifier,
                min_throughput,
                max_throughput,
            )
            raise ValueError(
                f"Sampler creation failed for team '{team_identifier}': "
                f"throughput_range_sampler returned None/False for "
                f"min_throughput={min_throughput}, max_throughput={max_throughput}"
            )


def calculate_team_throughput(team, query_manager, config):
    """Calculate team throughput based on configuration.

    Returns team throughput in one of two ways depending on configuration:

    1. **With throughput_samples**: Calculates actual historical throughput by
       fetching issues, computing cycle times, and analyzing completion data.
       Returns the cycle time data.

    2. **With min_throughput/max_throughput** (range-based): Returns the midpoint
       ``(min_throughput + max_throughput) / 2`` as a static estimate. This uses
       a lightweight data availability check without full historical analysis for
       performance. The midpoint aligns with the probabilistic range sampler used
       in simulations.

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
    team_jql = _build_team_jql(team, config, throughput_config)
    if team_jql is None:
        # No team-scoped query available - return 0 to indicate no data
        # rather than querying all issues
        team_name = (
            team.name
            if hasattr(team, "name")
            else (team.get("name") if isinstance(team, dict) else None)
        )
        logger.debug(
            "No team-scoped query available for team %s, returning 0", team_name
        )
        return 0

    # Use the safe has_issues_for_jql method to check data availability
    # without mutating global settings
    if query_manager.has_issues_for_jql(team_jql):
        # Return midpoint of min/max throughput range as a static estimate.
        # Note: When historical data is needed, use throughput_samples configuration
        # instead to trigger full cycle time analysis. See function docstring for
        # details on why the midpoint is used instead of historical calculation.
        return (min_throughput + max_throughput) / 2
    # No issues available
    return 0
