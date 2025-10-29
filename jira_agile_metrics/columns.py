"""Shared column definitions for Jira Agile Metrics.

This module centralizes reusable helpers that return column name lists
used across calculators and tests.
"""

from typing import List


def create_debt_columns() -> List[str]:
    """Create debt columns list to eliminate duplication.

    Returns the canonical column ordering for the debt calculator's
    DataFrame so both production code and tests remain consistent.
    """

    return [
        "key",
        "priority",
        "created",
        "resolved",
        "age",
        "type",
        "environment",
    ]
