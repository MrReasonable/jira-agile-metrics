"""Query functions for progress reports.

This module contains functions for finding outcomes and epics, and for
updating story counts.
"""

import dateutil
import pandas as pd

from .cycletime import CycleTimeParams, calculate_cycle_times
from .progressreport_models import Outcome


def find_outcomes(query_manager, query, outcome_deadline_field, epic_query_template):
    """Find outcomes using configuration.

    Args:
        query_manager: Query manager instance
        query: Query for finding outcomes
        outcome_deadline_field: Field name for outcome deadline
        epic_query_template: Template for epic queries

    Yields:
        Outcome objects
    """
    for issue in query_manager.find_issues(query):
        outcome_data = {
            "key": issue.key,
            "name": issue.fields.summary,
            "deadline": date_value(
                query_manager,
                issue,
                outcome_deadline_field,
            ),
            "epic_query": (
                epic_query_template.format(outcome=issue.key)
                if epic_query_template
                else None
            ),
        }
        yield Outcome(outcome_data)


def find_epics(query_manager, epic_config, outcome):
    """Find epics for an outcome using configuration.

    Args:
        query_manager: Query manager instance
        epic_config: Dictionary containing epic field configurations
        outcome: Outcome object

    Yields:
        Dictionary containing epic data with keys such as key, summary,
        status, resolution, resolution_date, min_stories, max_stories,
        team_name, deadline, and story_query
    """
    epic_min_stories_field = epic_config["min_stories_field"]
    epic_max_stories_field = epic_config["max_stories_field"]
    epic_team_field = epic_config["team_field"]
    epic_deadline_field = epic_config["deadline_field"]

    for issue in query_manager.find_issues(outcome.epic_query):
        yield {
            "key": issue.key,
            "summary": issue.fields.summary,
            "status": issue.fields.status.name,
            "resolution": (
                issue.fields.resolution.name if issue.fields.resolution else None
            ),
            "resolution_date": _parse_date_field(issue.fields.resolutiondate),
            "min_stories": (
                int_or_none(
                    query_manager.resolve_field_value(issue, epic_min_stories_field)
                )
                if epic_min_stories_field
                else None
            ),
            "max_stories": (
                int_or_none(
                    query_manager.resolve_field_value(issue, epic_max_stories_field)
                )
                if epic_max_stories_field
                else None
            ),
            "team_name": (
                query_manager.resolve_field_value(issue, epic_team_field)
                if epic_team_field
                else None
            ),
            "deadline": date_value(
                query_manager,
                issue,
                epic_deadline_field,
            ),
            "story_query": None,
        }


def _update_story_status_counts(epic, row, backlog_column, done_column):
    """Update story status counts based on row status.

    Args:
        epic: Epic object to update
        row: Cycle time row
        backlog_column: Backlog column name
        done_column: Done column name
    """
    if row["status"] == backlog_column:
        epic.data["stories_in_backlog"] += 1
    elif row["status"] == done_column:
        epic.data["stories_done"] += 1
    else:
        epic.data["stories_in_progress"] += 1


def _update_story_dates(epic, row):
    """Update first/last story dates from cycle time row.

    Args:
        epic: Epic object to update
        row: Cycle time row
    """
    if not pd.isna(row["started"]):
        if (
            epic.data["first_story_started"] is None
            or row["started"] < epic.data["first_story_started"]
        ):
            epic.data["first_story_started"] = row["started"]

    if not pd.isna(row["completed"]):
        if (
            epic.data["last_story_finished"] is None
            or row["completed"] > epic.data["last_story_finished"]
        ):
            epic.data["last_story_finished"] = row["completed"]


def update_story_counts(epic, query_manager, cycle, backlog_column, done_column):
    """Update story counts for an epic.

    Args:
        epic: Epic object to update
        query_manager: Query manager instance
        cycle: Cycle configuration
        backlog_column: Backlog column name
        done_column: Done column name
    """
    if not epic.data["story_query"]:
        return

    # Get stories for this epic
    stories = list(query_manager.find_issues(epic.data["story_query"]))

    # Calculate story counts
    epic.data["stories_raised"] = len(stories)
    epic.data["stories_in_backlog"] = 0
    epic.data["stories_in_progress"] = 0
    epic.data["stories_done"] = 0
    epic.data["first_story_started"] = None
    epic.data["last_story_finished"] = None

    story_cycle_times = []

    if not stories:
        epic.data["story_cycle_times"] = pd.DataFrame(story_cycle_times)
        return

    # Collect all story keys first
    story_keys = [story.key for story in stories]

    # Build a single JQL filter for all stories
    # JQL format: key in ("KEY1", "KEY2", "KEY3", ...)
    # Escape any internal double quotes by doubling them
    quoted_keys = ['"' + key.replace('"', '""') + '"' for key in story_keys]
    keys_jql = f"key in ({', '.join(quoted_keys)})"

    # Calculate cycle times for all stories in a single query
    all_cycle_times = calculate_cycle_times(
        CycleTimeParams(
            query_manager,
            {
                "cycle": cycle,
                "backlog_column": backlog_column,
                "done_column": done_column,
            },
            [{"jql": keys_jql}],
        )
    )

    # Create a lookup dictionary mapping story keys to their cycle time data
    cycle_times_by_key = {}
    if not all_cycle_times.empty:
        for _, row in all_cycle_times.iterrows():
            cycle_times_by_key[row["key"]] = row

    # Iterate through stories and map cycle time results
    for story in stories:
        cycle_time_data = cycle_times_by_key.get(story.key)

        if cycle_time_data is not None:
            # Store row for reuse and readability
            row = cycle_time_data
            story_cycle_times.append(row)

            # Update story counts based on status
            _update_story_status_counts(epic, row, backlog_column, done_column)

            # Update first/last dates
            _update_story_dates(epic, row)

    epic.data["story_cycle_times"] = pd.DataFrame(story_cycle_times)


def int_or_none(value):
    """Convert value to int or return None.

    Args:
        value: Value to convert

    Returns:
        Integer value or None
    """
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _parse_date_field(value, default=None):
    """Safely parse a date field value.

    Args:
        value: Date string value to parse
        default: Default value to return if parsing fails or value is falsy

    Returns:
        Parsed date value or default
    """
    if not value:
        return default

    try:
        return dateutil.parser.parse(value)
    except (ValueError, TypeError):
        return default


def date_value(query_manager, issue, field_name, default=None):
    """Get date value from issue field.

    Args:
        query_manager: Query manager instance
        issue: Issue object
        field_name: Field name
        default: Default value

    Returns:
        Date value or default
    """
    if not field_name:
        return default

    value = query_manager.resolve_field_value(issue, field_name)
    return _parse_date_field(value, default)
