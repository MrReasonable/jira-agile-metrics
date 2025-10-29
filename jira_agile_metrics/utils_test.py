"""Tests for utility functions in Jira Agile Metrics.

This module contains unit tests for the utility functions used across the metrics
calculations.
"""

import datetime

import numpy as np
import pandas as pd

from .utils import (
    breakdown_by_month,
    breakdown_by_month_sum_days,
    extend_dict,
    get_extension,
    to_bin,
    to_days_since_epoch,
    to_json_string,
)


def test_extend_dict():
    """Test extend_dict function."""
    assert extend_dict({"one": 1}, {"two": 2}) == {"one": 1, "two": 2}


def test_get_extension():
    """Test get_extension function."""
    assert get_extension("foo.csv") == ".csv"
    assert get_extension("/path/to/foo.csv") == ".csv"
    assert get_extension("\\path\\to\\foo.csv") == ".csv"
    assert get_extension("foo") == ""
    assert get_extension("foo.CSV") == ".csv"


def test_to_json_string():
    """Test to_json_string functionality."""
    assert to_json_string(1) == "1"
    assert to_json_string("foo") == "foo"
    assert to_json_string(None) == ""
    assert to_json_string(np.nan) == ""
    assert to_json_string(pd.NaT) == ""
    assert to_json_string(pd.Timestamp(2018, 2, 1)) == "2018-02-01"


def test_to_days_since_epoch():
    """Test to_days_since_epoch functionality."""
    assert to_days_since_epoch(datetime.date(1970, 1, 1)) == 0
    assert to_days_since_epoch(datetime.date(1970, 1, 15)) == 14


def test_breakdown_by_month():
    """Test breakdown_by_month functionality."""
    df = pd.DataFrame(
        [
            {
                "key": "ABC-1",
                "priority": "high",
                "start": pd.Timestamp(2018, 1, 1),
                "end": pd.Timestamp(2018, 3, 20),
            },
            {
                "key": "ABC-2",
                "priority": "med",
                "start": pd.Timestamp(2018, 1, 2),
                "end": pd.Timestamp(2018, 1, 20),
            },
            {
                "key": "ABC-3",
                "priority": "high",
                "start": pd.Timestamp(2018, 2, 3),
                "end": pd.Timestamp(2018, 3, 20),
            },
            {
                "key": "ABC-4",
                "priority": "med",
                "start": pd.Timestamp(2018, 1, 4),
                "end": pd.Timestamp(2018, 3, 20),
            },
            {
                "key": "ABC-5",
                "priority": "high",
                "start": pd.Timestamp(2018, 2, 5),
                "end": pd.Timestamp(2018, 2, 20),
            },
            {
                "key": "ABC-6",
                "priority": "med",
                "start": pd.Timestamp(2018, 3, 6),
                "end": pd.Timestamp(2018, 3, 20),
            },
        ],
        columns=["key", "priority", "start", "end"],
    )

    breakdown = breakdown_by_month(
        df,
        {
            "start_column": "start",
            "end_column": "end",
            "key_column": "key",
            "value_column": "priority",
            "output_columns": ["low", "med", "high"],
        },
    )
    assert list(breakdown.columns) == ["med", "high"]

    assert list(breakdown.index) == [
        pd.Timestamp(2018, 1, 1),
        pd.Timestamp(2018, 2, 1),
        pd.Timestamp(2018, 3, 1),
    ]
    assert breakdown.to_dict("records") == [
        {"high": 1, "med": 2},
        {"high": 3, "med": 1},
        {"high": 2, "med": 2},
    ]


def test_breakdown_by_month_open_ended():
    """Test breakdown_by_month with open-ended data."""
    df = pd.DataFrame(
        [
            {
                "key": "ABC-1",
                "priority": "high",
                "start": pd.Timestamp(2018, 1, 1),
                "end": pd.Timestamp(2018, 3, 20),
            },
            {
                "key": "ABC-2",
                "priority": "med",
                "start": pd.Timestamp(2018, 1, 2),
                "end": pd.Timestamp(2018, 1, 20),
            },
            {
                "key": "ABC-3",
                "priority": "high",
                "start": pd.Timestamp(2018, 2, 3),
                "end": pd.Timestamp(2018, 3, 20),
            },
            {
                "key": "ABC-4",
                "priority": "med",
                "start": pd.Timestamp(2018, 1, 4),
                "end": pd.Timestamp(2018, 3, 20),
            },
            {
                "key": "ABC-5",
                "priority": "high",
                "start": pd.Timestamp(2018, 2, 5),
                "end": pd.Timestamp(2018, 2, 20),
            },
            {
                "key": "ABC-6",
                "priority": "med",
                "start": pd.Timestamp(2018, 3, 6),
                "end": None,
            },
        ],
        columns=["key", "priority", "start", "end"],
    )

    breakdown = breakdown_by_month(
        df,
        {
            "start_column": "start",
            "end_column": "end",
            "key_column": "key",
            "value_column": "priority",
            "output_columns": ["low", "med", "high"],
        },
    )
    assert list(breakdown.columns) == ["med", "high"]

    # Note: We will get columns until the current month; assume this test is
    # run from June onwards ;)

    assert list(breakdown.index)[:5] == [
        pd.Timestamp(2018, 1, 1),
        pd.Timestamp(2018, 2, 1),
        pd.Timestamp(2018, 3, 1),
        pd.Timestamp(2018, 4, 1),
        pd.Timestamp(2018, 5, 1),
    ]
    assert breakdown.to_dict("records")[:5] == [
        {"high": 1, "med": 2},
        {"high": 3, "med": 1},
        {"high": 2, "med": 2},
        {"high": 0, "med": 1},
        {"high": 0, "med": 1},
    ]


def test_breakdown_by_month_no_column_spec():
    """Test breakdown_by_month without column specification."""
    df = pd.DataFrame(
        [
            {
                "key": "ABC-1",
                "priority": "high",
                "start": pd.Timestamp(2018, 1, 1),
                "end": pd.Timestamp(2018, 3, 20),
            },
            {
                "key": "ABC-2",
                "priority": "med",
                "start": pd.Timestamp(2018, 1, 2),
                "end": pd.Timestamp(2018, 1, 20),
            },
            {
                "key": "ABC-3",
                "priority": "high",
                "start": pd.Timestamp(2018, 2, 3),
                "end": pd.Timestamp(2018, 3, 20),
            },
            {
                "key": "ABC-4",
                "priority": "med",
                "start": pd.Timestamp(2018, 1, 4),
                "end": pd.Timestamp(2018, 3, 20),
            },
            {
                "key": "ABC-5",
                "priority": "high",
                "start": pd.Timestamp(2018, 2, 5),
                "end": pd.Timestamp(2018, 2, 20),
            },
            {
                "key": "ABC-6",
                "priority": "med",
                "start": pd.Timestamp(2018, 3, 6),
                "end": pd.Timestamp(2018, 3, 20),
            },
        ],
        columns=["key", "priority", "start", "end"],
    )

    breakdown = breakdown_by_month(
        df,
        {
            "start_column": "start",
            "end_column": "end",
            "key_column": "key",
            "value_column": "priority",
        },
    )
    assert list(breakdown.columns) == ["high", "med"]  # alphabetical

    assert list(breakdown.index) == [
        pd.Timestamp(2018, 1, 1),
        pd.Timestamp(2018, 2, 1),
        pd.Timestamp(2018, 3, 1),
    ]
    assert breakdown.to_dict("records") == [
        {"high": 1, "med": 2},
        {"high": 3, "med": 1},
        {"high": 2, "med": 2},
    ]


def test_breakdown_by_month_none_values():
    """Test breakdown_by_month with None values."""
    df = pd.DataFrame(
        [
            {
                "key": "ABC-1",
                "priority": None,
                "start": pd.Timestamp(2018, 1, 1),
                "end": pd.Timestamp(2018, 3, 20),
            },
            {
                "key": "ABC-2",
                "priority": None,
                "start": pd.Timestamp(2018, 1, 2),
                "end": pd.Timestamp(2018, 1, 20),
            },
            {
                "key": "ABC-3",
                "priority": None,
                "start": pd.Timestamp(2018, 2, 3),
                "end": pd.Timestamp(2018, 3, 20),
            },
            {
                "key": "ABC-4",
                "priority": None,
                "start": pd.Timestamp(2018, 1, 4),
                "end": pd.Timestamp(2018, 3, 20),
            },
            {
                "key": "ABC-5",
                "priority": None,
                "start": pd.Timestamp(2018, 2, 5),
                "end": pd.Timestamp(2018, 2, 20),
            },
            {
                "key": "ABC-6",
                "priority": None,
                "start": pd.Timestamp(2018, 3, 6),
                "end": pd.Timestamp(2018, 3, 20),
            },
        ],
        columns=["key", "priority", "start", "end"],
    )

    breakdown = breakdown_by_month(
        df,
        {
            "start_column": "start",
            "end_column": "end",
            "key_column": "key",
            "value_column": "priority",
        },
    )
    assert list(breakdown.columns) == [None]

    assert list(breakdown.index) == [
        pd.Timestamp(2018, 1, 1),
        pd.Timestamp(2018, 2, 1),
        pd.Timestamp(2018, 3, 1),
    ]
    assert breakdown.to_dict("records") == [{None: 3}, {None: 4}, {None: 4}]


def test_breakdown_by_month_sum_days():
    """Test breakdown_by_month_sum_days functionality."""
    df = pd.DataFrame(
        [
            {
                "priority": "high",
                "start": pd.Timestamp(2018, 1, 1),
                "end": pd.Timestamp(2018, 3, 20),
            },  # Jan: 31 Feb: 28 Mar: 20
            {
                "priority": "med",
                "start": pd.Timestamp(2018, 1, 2),
                "end": pd.Timestamp(2018, 1, 20),
            },  # Jan: 19 Feb:  0 Mar:  0
            {
                "priority": "high",
                "start": pd.Timestamp(2018, 2, 3),
                "end": pd.Timestamp(2018, 3, 20),
            },  # Jan:  0 Feb: 26 Mar: 20
            {
                "priority": "med",
                "start": pd.Timestamp(2018, 1, 4),
                "end": pd.Timestamp(2018, 3, 20),
            },  # Jan: 28 Feb: 28 Mar: 20
            {
                "priority": "high",
                "start": pd.Timestamp(2018, 2, 5),
                "end": pd.Timestamp(2018, 2, 20),
            },  # Jan:  0 Feb: 16 Mar:  0
            {
                "priority": "med",
                "start": pd.Timestamp(2018, 3, 6),
                "end": pd.Timestamp(2018, 3, 20),
            },  # Jan:  0 Feb:  0 Mar: 15
        ],
        columns=["key", "priority", "start", "end"],
    )

    breakdown = breakdown_by_month_sum_days(
        df,
        {
            "start_column": "start",
            "end_column": "end",
            "value_column": "priority",
            "output_columns": ["low", "med", "high"],
        },
    )
    assert list(breakdown.columns) == ["med", "high"]

    assert list(breakdown.index) == [
        pd.Timestamp(2018, 1, 1),
        pd.Timestamp(2018, 2, 1),
        pd.Timestamp(2018, 3, 1),
    ]

    assert breakdown.to_dict("records") == [
        {"high": 31.0, "med": 47.0},
        {"high": 70.0, "med": 28.0},
        {"high": 40.0, "med": 35.0},
    ]


def test_breakdown_by_month_sum_days_no_column_spec():
    """Test breakdown_by_month_sum_days without column specification."""
    df = pd.DataFrame(
        [
            {
                "priority": "high",
                "start": pd.Timestamp(2018, 1, 1),
                "end": pd.Timestamp(2018, 3, 20),
            },  # Jan: 31 Feb: 28 Mar: 20
            {
                "priority": "med",
                "start": pd.Timestamp(2018, 1, 2),
                "end": pd.Timestamp(2018, 1, 20),
            },  # Jan: 19 Feb:  0 Mar:  0
            {
                "priority": "high",
                "start": pd.Timestamp(2018, 2, 3),
                "end": pd.Timestamp(2018, 3, 20),
            },  # Jan:  0 Feb: 26 Mar: 20
            {
                "priority": "med",
                "start": pd.Timestamp(2018, 1, 4),
                "end": pd.Timestamp(2018, 3, 20),
            },  # Jan: 28 Feb: 28 Mar: 20
            {
                "priority": "high",
                "start": pd.Timestamp(2018, 2, 5),
                "end": pd.Timestamp(2018, 2, 20),
            },  # Jan:  0 Feb: 16 Mar:  0
            {
                "priority": "med",
                "start": pd.Timestamp(2018, 3, 6),
                "end": pd.Timestamp(2018, 3, 20),
            },  # Jan:  0 Feb:  0 Mar: 15
        ],
        columns=["key", "priority", "start", "end"],
    )

    breakdown = breakdown_by_month_sum_days(
        df, {"start_column": "start", "end_column": "end", "value_column": "priority"}
    )
    assert list(breakdown.columns) == ["high", "med"]  # alphabetical

    assert list(breakdown.index) == [
        pd.Timestamp(2018, 1, 1),
        pd.Timestamp(2018, 2, 1),
        pd.Timestamp(2018, 3, 1),
    ]

    assert breakdown.to_dict("records") == [
        {"high": 31.0, "med": 47.0},
        {"high": 70.0, "med": 28.0},
        {"high": 40.0, "med": 35.0},
    ]


def test_breakdown_by_month_sum_day_open_ended():
    """Test breakdown_by_month_sum_days with open-ended data."""
    df = pd.DataFrame(
        [
            {
                "priority": "high",
                "start": pd.Timestamp(2018, 1, 1),
                "end": pd.Timestamp(2018, 3, 20),
            },  # Jan: 31 Feb: 28 Mar: 20
            {
                "priority": "med",
                "start": pd.Timestamp(2018, 1, 2),
                "end": pd.Timestamp(2018, 1, 20),
            },  # Jan: 19 Feb:  0 Mar:  0
            {
                "priority": "high",
                "start": pd.Timestamp(2018, 2, 3),
                "end": pd.Timestamp(2018, 3, 20),
            },  # Jan:  0 Feb: 26 Mar: 20
            {
                "priority": "med",
                "start": pd.Timestamp(2018, 1, 4),
                "end": pd.Timestamp(2018, 3, 20),
            },  # Jan: 28 Feb: 28 Mar: 20
            {
                "priority": "high",
                "start": pd.Timestamp(2018, 2, 5),
                "end": pd.Timestamp(2018, 2, 20),
            },  # Jan:  0 Feb: 16 Mar:  0
            {
                "priority": "med",
                "start": pd.Timestamp(2018, 3, 6),
                "end": None,
            },  # Jan:  0 Feb:  0 Mar: 26
        ],
        columns=["key", "priority", "start", "end"],
    )

    breakdown = breakdown_by_month_sum_days(
        df,
        {
            "start_column": "start",
            "end_column": "end",
            "value_column": "priority",
            "output_columns": ["low", "med", "high"],
        },
    )
    assert list(breakdown.columns) == ["med", "high"]

    # Note: We will get columns until the current month; assume this test is
    # run from June onwards ;)

    assert list(breakdown.index)[:5] == [
        pd.Timestamp(2018, 1, 1),
        pd.Timestamp(2018, 2, 1),
        pd.Timestamp(2018, 3, 1),
        pd.Timestamp(2018, 4, 1),
        pd.Timestamp(2018, 5, 1),
    ]
    assert breakdown.to_dict("records")[:5] == [
        {"high": 31.0, "med": 47.0},
        {"high": 70.0, "med": 28.0},
        {"high": 40.0, "med": 46.0},
        {"high": 0, "med": 30.0},
        {"high": 0, "med": 31.0},
    ]


def test_breakdown_by_month_sum_days_none_values():
    """Test breakdown_by_month_sum_days with None values."""
    df = pd.DataFrame(
        [
            {
                "priority": None,
                "start": pd.Timestamp(2018, 1, 1),
                "end": pd.Timestamp(2018, 3, 20),
            },  # Jan: 31 Feb: 28 Mar: 20
            {
                "priority": "med",
                "start": pd.Timestamp(2018, 1, 2),
                "end": pd.Timestamp(2018, 1, 20),
            },  # Jan: 19 Feb:  0 Mar:  0
            {
                "priority": None,
                "start": pd.Timestamp(2018, 2, 3),
                "end": pd.Timestamp(2018, 3, 20),
            },  # Jan:  0 Feb: 26 Mar: 20
            {
                "priority": "med",
                "start": pd.Timestamp(2018, 1, 4),
                "end": pd.Timestamp(2018, 3, 20),
            },  # Jan: 28 Feb: 28 Mar: 20
            {
                "priority": None,
                "start": pd.Timestamp(2018, 2, 5),
                "end": pd.Timestamp(2018, 2, 20),
            },  # Jan:  0 Feb: 16 Mar:  0
            {
                "priority": "med",
                "start": pd.Timestamp(2018, 3, 6),
                "end": pd.Timestamp(2018, 3, 20),
            },  # Jan:  0 Feb:  0 Mar: 15
        ],
        columns=["key", "priority", "start", "end"],
    )

    breakdown = breakdown_by_month_sum_days(
        df, {"start_column": "start", "end_column": "end", "value_column": "priority"}
    )
    assert set(breakdown.columns) == set([None, "med"])

    assert list(breakdown.index) == [
        pd.Timestamp(2018, 1, 1),
        pd.Timestamp(2018, 2, 1),
        pd.Timestamp(2018, 3, 1),
    ]

    assert breakdown.to_dict("records") == [
        {None: 31.0, "med": 47.0},
        {None: 70.0, "med": 28.0},
        {None: 40.0, "med": 35.0},
    ]


def test_to_bin():
    """Test to_bin functionality."""
    assert to_bin(0, [10, 20, 30]) == (0, 10)
    assert to_bin(9, [10, 20, 30]) == (0, 10)
    assert to_bin(10, [10, 20, 30]) == (0, 10)

    assert to_bin(11, [10, 20, 30]) == (10, 20)
    assert to_bin(20, [10, 20, 30]) == (10, 20)

    assert to_bin(30, [10, 20, 30]) == (20, 30)

    assert to_bin(31, [10, 20, 30]) == (30, None)
