"""Tests for defects calculator functionality in Jira Agile Metrics.

This module contains unit tests for the defects calculator.
"""

import pandas as pd
import pytest

from ..querymanager import QueryManager
from ..test_classes import FauxJIRA as JIRA
from ..test_data import COMMON_DEFECT_TEST_ISSUES
from ..test_utils import (
    assert_common_d1_d2_record_values,
    assert_common_d1_d2_record_values_no_priority,
    create_common_defect_test_settings,
    run_empty_calculator_test,
    validate_defect_test_data,
)
from ..utils import create_common_defects_columns, extend_dict
from .defects import DefectsCalculator


@pytest.fixture(name="fields")
def fixture_fields(base_minimal_fields):
    """Provide fields fixture for defects tests."""
    return base_minimal_fields + [
        {"id": "priority", "name": "Priority"},
        {"id": "customfield_001", "name": "Environment"},
        {"id": "customfield_002", "name": "Defect type"},
    ]


@pytest.fixture(name="settings")
def fixture_settings(base_minimal_settings):
    """Provide settings fixture for defects tests."""
    return extend_dict(
        base_minimal_settings,
        {
            "defects_query": "issueType = Defect",
            "defects_window": 3,
            "defects_priority_field": "Priority",
            "defects_priority_values": ["Low", "Medium", "High"],
            **create_common_defect_test_settings(),
        },
    )


@pytest.fixture(name="jira")
def fixture_jira(fields):
    """Provide JIRA fixture for defects tests."""
    return JIRA(
        fields=fields,
        issues=COMMON_DEFECT_TEST_ISSUES,
    )


def test_no_query(jira, settings):
    """Test defects calculator with no query."""
    query_manager = QueryManager(jira, settings)
    results = {}
    settings = extend_dict(settings, {"defects_query": None})
    calculator = DefectsCalculator(query_manager, settings, results)

    data = calculator.run()
    assert data is None


def test_columns(jira, settings):
    """Test defects calculator column structure."""
    query_manager = QueryManager(jira, settings)
    results = {}
    calculator = DefectsCalculator(query_manager, settings, results)

    data = calculator.run()

    assert list(data.columns) == create_common_defects_columns()


def test_empty(fields, settings):
    """Test defects calculator with empty data."""
    expected_columns = create_common_defects_columns()

    data = run_empty_calculator_test(
        DefectsCalculator, fields, settings, expected_columns
    )
    assert len(data.index) == 0


def test_breakdown(jira, settings):
    """Test defects calculator breakdown functionality."""
    query_manager = QueryManager(jira, settings)
    results = {}
    calculator = DefectsCalculator(query_manager, settings, results)

    data = calculator.run()

    # Check that we have the expected number of rows
    assert len(data) == 12

    # Check that we have the expected columns
    expected_columns = ["key", "priority", "type", "environment", "created", "resolved"]
    records, valid_records = validate_defect_test_data(data, expected_columns)

    # Check specific valid records - common assertions
    assert_common_d1_d2_record_values(valid_records)

    # Additional defects-specific checks
    d1_record = next(r for r in valid_records if r["key"] == "D-1")
    assert d1_record["type"] == "Bug"
    assert d1_record["environment"] == "PROD"

    d2_record = next(r for r in valid_records if r["key"] == "D-2")
    assert d2_record["type"] == "Bug"
    assert d2_record["environment"] == "SIT"

    # Check that we have 6 records with nan values
    nan_records = [r for r in records if pd.isna(r["key"])]
    assert len(nan_records) == 6


def test_no_priority_field(jira, settings):
    """Test defects calculator without priority field."""
    settings = extend_dict(settings, {"defects_priority_field": None})

    query_manager = QueryManager(jira, settings)
    results = {}
    calculator = DefectsCalculator(query_manager, settings, results)

    data = calculator.run()

    # Check that we have the expected number of rows
    assert len(data) == 12

    # Check that we have the expected columns
    expected_columns = ["key", "priority", "type", "environment", "created", "resolved"]
    records, valid_records = validate_defect_test_data(data, expected_columns)

    # Check specific valid records - common assertions (except priority)
    assert_common_d1_d2_record_values_no_priority(valid_records)

    # Additional defects-specific checks
    d1_record = next(r for r in valid_records if r["key"] == "D-1")
    assert d1_record["type"] == "Bug"
    assert d1_record["environment"] == "PROD"

    d2_record = next(r for r in valid_records if r["key"] == "D-2")
    assert d2_record["type"] == "Bug"
    assert d2_record["environment"] == "SIT"

    # Check that we have 6 records with nan values
    nan_records = [r for r in records if pd.isna(r["key"])]
    assert len(nan_records) == 6


def test_no_type_field(jira, settings):
    """Test defects calculator without type field."""
    settings = extend_dict(settings, {"defects_type_field": None})

    query_manager = QueryManager(jira, settings)
    results = {}
    calculator = DefectsCalculator(query_manager, settings, results)

    data = calculator.run()

    # Check that we have the expected number of rows
    assert len(data) == 12

    # Check that we have the expected columns
    expected_columns = ["key", "priority", "type", "environment", "created", "resolved"]
    assert list(data.columns) == expected_columns

    # Check the actual data values (ignoring field order)
    records = data.to_dict("records")

    # Check valid records (first 6)
    valid_records = [
        r for r in records if r["key"] in ["D-1", "D-2", "D-3", "D-4", "D-5", "D-6"]
    ]
    assert len(valid_records) == 6

    # Check specific valid records - common assertions
    assert_common_d1_d2_record_values(valid_records)

    # Additional defects-specific checks
    d1_record = next(r for r in valid_records if r["key"] == "D-1")
    assert d1_record["type"] is None
    assert d1_record["environment"] == "PROD"

    d2_record = next(r for r in valid_records if r["key"] == "D-2")
    assert d2_record["type"] is None
    assert d2_record["environment"] == "SIT"

    # Check that we have 6 records with nan values
    nan_records = [r for r in records if pd.isna(r["key"])]
    assert len(nan_records) == 6


def test_no_environment_field(jira, settings):
    """Test defects calculator without environment field."""
    settings = extend_dict(settings, {"defects_environment_field": None})

    query_manager = QueryManager(jira, settings)
    results = {}
    calculator = DefectsCalculator(query_manager, settings, results)

    data = calculator.run()

    # Check that we have the expected number of rows
    assert len(data) == 12

    # Check that we have the expected columns
    expected_columns = ["key", "priority", "type", "environment", "created", "resolved"]
    assert list(data.columns) == expected_columns

    # Check the actual data values (ignoring field order)
    records = data.to_dict("records")

    # Check valid records (first 6)
    valid_records = [
        r for r in records if r["key"] in ["D-1", "D-2", "D-3", "D-4", "D-5", "D-6"]
    ]
    assert len(valid_records) == 6

    # Check specific valid records - common assertions
    assert_common_d1_d2_record_values(valid_records)

    # Additional defects-specific checks
    d1_record = next(r for r in valid_records if r["key"] == "D-1")
    assert d1_record["type"] == "Bug"
    assert d1_record["environment"] is None

    d2_record = next(r for r in valid_records if r["key"] == "D-2")
    assert d2_record["type"] == "Bug"
    assert d2_record["environment"] is None

    # Check that we have 6 records with nan values
    nan_records = [r for r in records if pd.isna(r["key"])]
    assert len(nan_records) == 6
