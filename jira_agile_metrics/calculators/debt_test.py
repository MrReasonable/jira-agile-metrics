"""Tests for debt calculator functionality in Jira Agile Metrics.

This module contains unit tests for the debt calculator.
"""

import datetime

import pandas as pd
import pytest
from pandas import Timedelta

from ..columns import DEBT_COLUMNS
from ..querymanager import QueryManager
from ..test_classes import FauxJIRA as JIRA
from ..test_data import COMMON_TEST_ISSUES
from ..test_utils import (
    assert_common_d1_d2_record_values,
    assert_common_d1_d2_record_values_no_priority,
    run_empty_calculator_test,
    validate_defect_test_data,
)
from ..utils import extend_dict
from .debt import DebtCalculator


@pytest.fixture(name="fields")
def fixture_fields(base_minimal_fields):
    """Provide fields fixture for debt tests."""
    return base_minimal_fields + [
        {"id": "priority", "name": "Priority"},
    ]


@pytest.fixture(name="settings")
def fixture_settings(base_minimal_settings):
    """Provide settings fixture for debt tests."""
    return extend_dict(
        base_minimal_settings,
        {
            "debt_query": 'issueType = "Tech Debt"',
            "debt_priority_field": "Priority",
            "debt_priority_values": ["Low", "Medium", "High"],
            "debt_chart": "debt-chart.png",
            "debt_chart_title": "Debt chart",
            "debt_window": 3,
            "debt_age_chart": "debt-age-chart.png",
            "debt_age_chart_title": "Debt age",
            "debt_age_chart_bins": [10, 20, 30],
        },
    )


@pytest.fixture(name="jira")
def fixture_jira(fields):
    """Provide JIRA fixture for debt tests."""
    return JIRA(
        fields=fields,
        issues=COMMON_TEST_ISSUES,
    )


def test_no_query(jira, settings):
    """Test debt calculator with no query configured."""
    query_manager = QueryManager(jira, settings)
    results = {}
    settings = extend_dict(settings, {"debt_query": None})
    calculator = DebtCalculator(query_manager, settings, results)

    data = calculator.run()
    assert data is None


def test_columns(jira, settings):
    """Test debt calculator column structure."""
    query_manager = QueryManager(jira, settings)
    results = {}
    calculator = DebtCalculator(query_manager, settings, results)

    data = calculator.run()
    expected_columns = DEBT_COLUMNS
    assert list(data.columns) == expected_columns


def test_empty(fields, settings):
    """Test debt calculator with empty data."""

    expected_columns = DEBT_COLUMNS

    data = run_empty_calculator_test(DebtCalculator, fields, settings, expected_columns)
    assert len(data.index) == 0


def test_breakdown(jira, settings):
    """Test debt calculator breakdown functionality."""
    query_manager = QueryManager(jira, settings)
    results = {}
    calculator = DebtCalculator(query_manager, settings, results)

    data = calculator.run(now=datetime.datetime(2018, 3, 21, 2, 2, 2))

    # Check that we have the expected number of rows
    assert len(data) == 12

    # Check that we have the expected columns
    expected_columns = DEBT_COLUMNS
    records, valid_records = validate_defect_test_data(data, expected_columns)

    # Check specific valid records - common assertions
    assert_common_d1_d2_record_values(valid_records)

    # Additional debt-specific checks
    d1_record = next(r for r in valid_records if r["key"] == "D-1")
    assert d1_record["age"] == Timedelta("78 days 01:01:01")
    assert d1_record["type"] is None
    assert d1_record["environment"] is None

    d2_record = next(r for r in valid_records if r["key"] == "D-2")
    assert d2_record["age"] == Timedelta("18 days 01:01:01")
    assert d2_record["type"] is None
    assert d2_record["environment"] is None

    # Check that we have 6 records with nan values
    nan_records = [r for r in records if pd.isna(r["key"])]
    assert len(nan_records) == 6


def test_no_priority_field(jira, settings):
    """Test debt calculator with no priority field configured."""
    settings = extend_dict(settings, {"debt_priority_field": None})

    query_manager = QueryManager(jira, settings)
    results = {}
    calculator = DebtCalculator(query_manager, settings, results)

    data = calculator.run(now=datetime.datetime(2018, 3, 21, 2, 2, 2))

    # Check that we have the expected number of rows
    assert len(data) == 12

    # Check that we have the expected columns
    expected_columns = DEBT_COLUMNS
    records, valid_records = validate_defect_test_data(data, expected_columns)

    # Check specific valid records (no priority check due to no priority field)
    assert_common_d1_d2_record_values_no_priority(valid_records)

    # Additional debt-specific checks
    d1_record = next(r for r in valid_records if r["key"] == "D-1")
    assert d1_record["age"] == Timedelta("78 days 01:01:01")
    assert d1_record["type"] is None
    assert d1_record["environment"] is None

    d2_record = next(r for r in valid_records if r["key"] == "D-2")
    assert d2_record["age"] == Timedelta("18 days 01:01:01")
    assert d2_record["type"] is None
    assert d2_record["environment"] is None

    # Check that we have 6 records with nan values
    nan_records = [r for r in records if pd.isna(r["key"])]
    assert len(nan_records) == 6
