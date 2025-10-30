"""Forecasting and simulation functions for progress reports.

This module contains functions for Monte Carlo simulation and forecasting
of epic completion times.
"""

import datetime

import numpy as np


def _simulate_single_trial(team, epics, max_iterations):
    """Simulate a single trial of epic completion.

    Args:
        team: Team object
        epics: List of Epic objects
        max_iterations: Maximum number of iterations

    Returns:
        Dictionary with trial results
    """
    # Initialize trial state
    trial_state = {
        "epics": {epic.key: {"completed": False, "iterations": 0} for epic in epics},
        "team_wip": team.wip,
        "completed_epics": [],
    }

    # Simulate iterations
    for _ in range(max_iterations):
        # Get available WIP slots
        available_wip = team.wip - sum(
            1
            for epic in epics
            if not trial_state["epics"][epic.key]["completed"]
            and trial_state["epics"][epic.key]["iterations"] > 0
        )

        # Process epics
        for epic in epics:
            if trial_state["epics"][epic.key]["completed"]:
                continue

            # Check if epic can be worked on
            if available_wip > 0:
                trial_state["epics"][epic.key]["iterations"] += 1
                available_wip -= 1

                # Check if epic is completed
                if (
                    trial_state["epics"][epic.key]["iterations"]
                    >= epic.data["max_stories"]
                ):
                    trial_state["epics"][epic.key]["completed"] = True
                    trial_state["completed_epics"].append(epic.key)

        # Check if all epics are completed
        if all(trial_state["epics"][epic.key]["completed"] for epic in epics):
            break

    return trial_state


def forecast_to_complete(team, epics, forecast_config):
    """Forecast completion time for epics.

    Args:
        team: Team object
        epics: List of Epic objects
        forecast_config: Forecast configuration

    Returns:
        Dictionary with forecast results
    """
    trials = forecast_config.get("trials", 1000)
    quantiles = forecast_config.get("quantiles", [0.5, 0.75, 0.85, 0.95])
    max_iterations = forecast_config.get("max_iterations", 1000)

    # Run Monte Carlo simulation
    trial_results = []
    for _ in range(trials):
        trial_result = _simulate_single_trial(team, epics, max_iterations)
        trial_results.append(trial_result)

    # Calculate quantiles for each epic
    epic_forecasts = {}
    for epic in epics:
        epic_trials = [
            result["epics"][epic.key]["iterations"] for result in trial_results
        ]
        quantile_values = {}
        for quantile in quantiles:
            quantile_values[quantile] = np.percentile(epic_trials, quantile * 100)
        epic_forecasts[epic.key] = quantile_values

    return epic_forecasts


def _validate_stories_count(value, default=0):
    """Validate and convert stories count.

    Args:
        value: Raw value to validate
        default: Default value if missing

    Returns:
        Validated float value or None if invalid
    """
    try:
        count = float(value if value is not None else default)
        return count if count >= 0 else None
    except (ValueError, TypeError):
        return None


def _parse_start_date(date_str):
    """Parse first story started date string.

    Args:
        date_str: Date string in YYYY-MM-DD format

    Returns:
        Parsed date object or None if invalid
    """
    if not date_str:
        return None
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None


def calculate_epic_target(epic):
    """Calculate target completion date for an epic.

    Args:
        epic: Epic object

    Returns:
        Target completion date or None
    """
    # Validate all inputs first
    first_story_started_str = epic.data.get("first_story_started")
    first_story_started = _parse_start_date(first_story_started_str)
    stories_done = _validate_stories_count(epic.data.get("stories_done", 0))
    stories_total = _validate_stories_count(epic.data.get("stories_raised", 0))

    # Early validation check - consolidate multiple checks
    is_valid = (
        first_story_started is not None
        and stories_done is not None
        and stories_total is not None
        and isinstance(stories_total, (int, float))
        and isinstance(stories_done, (int, float))
        and stories_total > 0
        and 0 <= stories_done <= stories_total
    )
    if not is_valid:
        return None

    # Calculate completion rate
    completion_rate = stories_done / stories_total
    if completion_rate == 0:
        return None

    # Estimate remaining time and calculate target date
    remaining_stories = stories_total - stories_done
    estimated_remaining_time = remaining_stories / completion_rate

    # Validate, convert estimated time, and calculate target date
    try:
        estimated_remaining_time = int(round(estimated_remaining_time))
        if estimated_remaining_time < 0:
            return None
        target_date = first_story_started + datetime.timedelta(
            days=estimated_remaining_time
        )
        return target_date
    except (ValueError, TypeError, OverflowError):
        return None


def forward_weeks(date, weeks):
    """Add weeks to a date.

    Args:
        date: Base date
        weeks: Number of weeks to add

    Returns:
        New date
    """
    return date + datetime.timedelta(weeks=weeks)
