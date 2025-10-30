"""Common test data factory to eliminate code duplication."""

from datetime import date, datetime
from typing import Any, Dict, List

import pandas as pd
from pandas import NaT

from .calculators.cycletime import CycleTimeCalculator


def _ts(date_string: str) -> datetime:
    """Convert date string to timestamp."""
    return pd.to_datetime(date_string)


def _determine_status(issue: Dict[str, Any]) -> str:
    """Determine status from issue data based on timestamp fields.

    Checks status fields in order: Done, Test, Build, Committed, else Backlog.
    """
    if issue["Done"] is not NaT:
        return "Done"
    if issue["Test"] is not NaT:
        return "Test"
    if issue["Build"] is not NaT:
        return "Build"
    if issue["Committed"] is not NaT:
        return "Committed"
    return "Backlog"


def _issues(issues: List[Dict[str, Any]]):
    """Convert issue data to proper format matching conftest.py."""
    return [
        {
            "key": f"A-{idx + 1}",
            "url": f"https://example.org/browse/A-{idx + 1}",
            "issue_type": "Story",
            "summary": f"Generated issue A-{idx + 1}",
            "status": _determine_status(i),
            "resolution": "Done" if i["Done"] is not NaT else None,
            "completed_timestamp": i["Done"] if i["Done"] is not NaT else None,
            "cycle_time": (
                (i["Done"] - i["Committed"])
                if (i["Done"] is not NaT and i["Committed"] is not NaT)
                else None
            ),
            "lead_time": (
                (i["Done"] - i["Backlog"])
                if (i["Done"] is not NaT and i["Backlog"] is not NaT)
                else None
            ),
            "blocked_days": i.get("blocked_days", 0),
            "impediments": i.get("impediments", []),
            "Backlog": i["Backlog"],
            "Committed": i["Committed"],
            "Build": i["Build"],
            "Test": i["Test"],
            "Done": i["Done"],
        }
        for idx, i in enumerate(issues)
    ]


def create_cycle_time_test_data() -> Dict[str, Any]:
    """Create common cycle time test data to eliminate duplication."""
    return {
        "minimal_data": [
            {
                "key": "A-1",
                "Backlog": _ts("2018-01-01"),
                "Committed": NaT,
                "Build": NaT,
                "Test": NaT,
                "Done": NaT,
                "blocked_days": 0,
                "impediments": [],
            },
            {
                "key": "A-2",
                "Backlog": _ts("2018-01-02"),
                "Committed": _ts("2018-01-03"),
                "Build": NaT,
                "Test": NaT,
                "Done": NaT,
                "blocked_days": 0,
                "impediments": [],
            },
            {
                "key": "A-3",
                "Backlog": _ts("2018-01-03"),
                "Committed": _ts("2018-01-03"),
                "Build": _ts("2018-01-04"),
                "Test": _ts("2018-01-05"),
                "Done": _ts("2018-01-06"),
                "blocked_days": 0,
                "impediments": [],
            },
            {
                "key": "A-4",
                "Backlog": _ts("2018-01-04"),
                "Committed": _ts("2018-01-04"),
                "Build": NaT,
                "Test": NaT,
                "Done": NaT,
                "blocked_days": 0,
                "impediments": [],
            },
        ],
        "extended_data": [
            {
                "key": "A-1",
                "Backlog": _ts("2018-01-01"),
                "Committed": NaT,
                "Build": NaT,
                "Test": NaT,
                "Done": NaT,
                "blocked_days": 0,
                "impediments": [],
            },
            {
                "key": "A-2",
                "Backlog": _ts("2018-01-02"),
                "Committed": _ts("2018-01-03"),
                "Build": NaT,
                "Test": NaT,
                "Done": NaT,
                "blocked_days": 0,
                "impediments": [],
            },
            {
                "key": "A-3",
                "Backlog": _ts("2018-01-03"),
                "Committed": _ts("2018-01-03"),
                "Build": _ts("2018-01-04"),
                "Test": _ts("2018-01-05"),
                "Done": _ts("2018-01-06"),
                "blocked_days": 0,
                "impediments": [],
            },
            {
                "key": "A-4",
                "Backlog": _ts("2018-01-04"),
                "Committed": _ts("2018-01-04"),
                "Build": NaT,
                "Test": NaT,
                "Done": NaT,
                "blocked_days": 0,
                "impediments": [],
            },
            {
                "key": "A-5",
                "Backlog": _ts("2018-01-05"),
                "Committed": _ts("2018-01-05"),
                "Build": _ts("2018-01-06"),
                "Test": _ts("2018-01-07"),
                "Done": _ts("2018-01-08"),
                "blocked_days": 0,
                "impediments": [],
            },
        ],
    }


def create_cycle_time_dataframe(data_type: str, columns: List[str]) -> pd.DataFrame:
    """Create cycle time DataFrame from common test data."""
    test_data = create_cycle_time_test_data()
    data = test_data.get(data_type, test_data["minimal_data"])

    return pd.DataFrame(
        _issues(data),
        columns=columns,
    )


def create_minimal_cycle_time_results(columns: List[str]) -> Dict[Any, pd.DataFrame]:
    """Create minimal cycle time results to eliminate duplication."""
    return {CycleTimeCalculator: create_cycle_time_dataframe("minimal_data", columns)}


def create_large_cycle_time_results(columns: List[str]) -> Dict[Any, pd.DataFrame]:
    """Create large cycle time results to eliminate duplication."""
    return {CycleTimeCalculator: create_cycle_time_dataframe("extended_data", columns)}


def create_impediments_test_data() -> List[Dict[str, Any]]:
    """Create impediments test data to eliminate duplication."""
    return [
        {
            "Backlog": _ts("2018-01-01"),
            "Committed": NaT,
            "Build": NaT,
            "Test": NaT,
            "Done": NaT,
            "blocked_days": 0,
            "impediments": [],
        },
        {
            "Backlog": _ts("2018-01-02"),
            "Committed": _ts("2018-01-03"),
            "Build": NaT,
            "Test": NaT,
            "Done": NaT,
            "blocked_days": 4,
            "impediments": [
                {
                    "start": date(2018, 1, 5),
                    "end": date(2018, 1, 7),
                    "status": "Backlog",
                    "flag": "Impediment",
                },  # ignored because it was blocked in backlog
                {
                    "start": date(2018, 1, 10),
                    "end": date(2018, 1, 12),
                    "status": "Committed",
                    "flag": "Impediment",
                },  # included
            ],
        },
        {
            "Backlog": _ts("2018-01-03"),
            "Committed": _ts("2018-01-03"),
            "Build": _ts("2018-01-04"),
            "Test": _ts("2018-01-05"),
            "Done": _ts("2018-01-06"),
            "blocked_days": 4,
            "impediments": [
                {
                    "start": date(2018, 1, 4),
                    "end": date(2018, 1, 5),
                    "status": "Build",
                    "flag": "Impediment",
                },  # included
                {
                    "start": date(2018, 1, 7),
                    "end": date(2018, 1, 10),
                    "status": "Done",
                    "flag": "Impediment",
                },  # ignored because it was blocked in done
            ],
        },
        {
            "Backlog": _ts("2018-01-04"),
            "Committed": _ts("2018-01-04"),
            "Build": NaT,
            "Test": NaT,
            "Done": NaT,
            "blocked_days": 100,
            "impediments": [
                {
                    "start": date(2018, 1, 5),
                    "end": None,
                    "status": "Committed",
                    "flag": "Awaiting input",
                },  # open ended, still included
            ],
        },
    ]


def create_impediments_cycle_time_results(
    columns: List[str],
) -> Dict[Any, pd.DataFrame]:
    """Create impediments cycle time results to eliminate duplication."""
    return {
        CycleTimeCalculator: pd.DataFrame(
            _issues(create_impediments_test_data()),
            columns=columns,
        )
    }
