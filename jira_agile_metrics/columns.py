"""Shared column definitions for Jira Agile Metrics.

This module centralizes reusable helpers that return column name lists
used across calculators and tests.
"""

DEBT_COLUMNS = [
    "key",
    "priority",
    "created",
    "resolved",
    "age",
    "type",
    "environment",
]
