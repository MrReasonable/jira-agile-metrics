"""Tests for CFD calculator functionality in Jira Agile Metrics.

This module contains unit tests for the CFD (Cumulative Flow Diagram) calculator.
"""

import os

import pandas as pd
import pytest
from pandas import DataFrame

from ..test_data import COMMON_CFD_DATA
from ..test_utils import (
    EXTENDED_TIMESTAMP_INDEX,
    assert_calculator_wrote_json_file,
    create_common_cycle_status_list,
)
from ..utils import extend_dict
from .cfd import CFDCalculator
from .cycletime import CycleTimeCalculator


def assert_cfd_timestamp_index(data):
    """Assert that data has the CFD timestamp index."""
    # Use first 7 elements of EXTENDED_TIMESTAMP_INDEX
    expected_index = EXTENDED_TIMESTAMP_INDEX[:7]
    assert list(data.index) == expected_index


@pytest.fixture(name="test_query_manager")
def query_manager(minimal_query_manager):
    """Test query manager fixture."""
    return minimal_query_manager


@pytest.fixture(name="test_settings")
def settings(base_minimal_settings):
    """Test settings fixture."""
    return extend_dict(base_minimal_settings, {})


@pytest.fixture(name="test_columns")
def columns(base_minimal_cycle_time_columns):
    """Test columns fixture."""
    return base_minimal_cycle_time_columns


def test_empty(test_query_manager, test_settings, test_columns):
    """Test CFD calculator with empty data."""
    results = {CycleTimeCalculator: DataFrame([], columns=test_columns)}

    calculator = CFDCalculator(test_query_manager, test_settings, results)

    data = calculator.run()
    assert len(data.index) == 0


def test_cfd_columns(
    test_query_manager, test_settings, base_minimal_cycle_time_results
):
    """Test CFD calculator column handling."""
    calculator = CFDCalculator(
        test_query_manager, test_settings, base_minimal_cycle_time_results
    )

    data = calculator.run()

    assert list(data.columns) == create_common_cycle_status_list()


def test_calculate_cfd(
    test_query_manager, test_settings, base_minimal_cycle_time_results
):
    """Test CFD calculation functionality."""
    calculator = CFDCalculator(
        test_query_manager, test_settings, base_minimal_cycle_time_results
    )

    data = calculator.run()

    assert_cfd_timestamp_index(data)

    assert data.to_dict("records") == COMMON_CFD_DATA


def test_write_file_csv(
    test_query_manager, test_settings, base_minimal_cycle_time_results, tmp_path
):
    """Test writing CFD data to CSV file."""
    output_file = str(tmp_path / "cfd.csv")
    test_settings_extended = extend_dict(
        test_settings,
        {
            "cfd_data": [output_file],
            "cfd_chart": None,
        },
    )

    calculator = CFDCalculator(
        test_query_manager, test_settings_extended, base_minimal_cycle_time_results
    )
    result = calculator.run()
    base_minimal_cycle_time_results[CFDCalculator] = result
    calculator.write()

    assert os.path.exists(output_file)
    df = pd.read_csv(output_file)
    assert "Date" in df.columns
    assert len(df) > 0


def test_write_file_json(
    test_query_manager, test_settings, base_minimal_cycle_time_results, tmp_path
):
    """Test writing CFD data to JSON file."""
    output_file = str(tmp_path / "cfd.json")
    test_settings_extended = extend_dict(
        test_settings,
        {
            "cfd_data": [output_file],
            "cfd_chart": None,
        },
    )

    calculator = CFDCalculator(
        test_query_manager, test_settings_extended, base_minimal_cycle_time_results
    )
    assert_calculator_wrote_json_file(
        calculator, output_file, base_minimal_cycle_time_results
    )


def test_write_file_xlsx(
    test_query_manager, test_settings, base_minimal_cycle_time_results, tmp_path
):
    """Test writing CFD data to XLSX file."""
    output_file = str(tmp_path / "cfd.xlsx")
    test_settings_extended = extend_dict(
        test_settings,
        {
            "cfd_data": [output_file],
            "cfd_chart": None,
        },
    )

    calculator = CFDCalculator(
        test_query_manager, test_settings_extended, base_minimal_cycle_time_results
    )
    result = calculator.run()
    base_minimal_cycle_time_results[CFDCalculator] = result
    calculator.write()

    assert os.path.exists(output_file)
    df = pd.read_excel(output_file, sheet_name="CFD")
    assert len(df.columns) > 0


def test_write_chart(
    test_query_manager, test_settings, base_minimal_cycle_time_results, tmp_path
):
    """Test writing CFD chart."""
    output_file = str(tmp_path / "cfd.png")
    test_settings_extended = extend_dict(
        test_settings,
        {
            "cfd_data": None,
            "cfd_chart": output_file,
            "cfd_window": None,
            "cfd_chart_title": "Test CFD Chart",
        },
    )

    calculator = CFDCalculator(
        test_query_manager, test_settings_extended, base_minimal_cycle_time_results
    )
    result = calculator.run()
    base_minimal_cycle_time_results[CFDCalculator] = result
    calculator.write()

    assert os.path.exists(output_file)


def test_write_chart_with_window(
    test_query_manager, test_settings, base_minimal_cycle_time_results, tmp_path
):
    """Test writing CFD chart with window filtering."""
    output_file = str(tmp_path / "cfd.png")
    test_settings_extended = extend_dict(
        test_settings,
        {
            "cfd_data": None,
            "cfd_chart": output_file,
            "cfd_window": 5,
            "cfd_chart_title": "Test CFD Chart",
        },
    )

    calculator = CFDCalculator(
        test_query_manager, test_settings_extended, base_minimal_cycle_time_results
    )
    result = calculator.run()
    base_minimal_cycle_time_results[CFDCalculator] = result
    calculator.write()

    assert os.path.exists(output_file)


def test_write_chart_empty_data(
    test_query_manager, test_settings, test_columns, tmp_path
):
    """Test writing CFD chart with empty data."""
    output_file = str(tmp_path / "cfd.png")
    results = {CycleTimeCalculator: DataFrame([], columns=test_columns)}
    test_settings_extended = extend_dict(
        test_settings,
        {
            "cfd_data": None,
            "cfd_chart": output_file,
            "cfd_window": None,
        },
    )

    calculator = CFDCalculator(test_query_manager, test_settings_extended, results)
    result = calculator.run()
    results[CFDCalculator] = result
    calculator.write()

    # Chart should not be created with empty data
    assert not os.path.exists(output_file)


def test_write_no_output_files(
    test_query_manager, test_settings, base_minimal_cycle_time_results
):
    """Test write() when no output files are specified."""
    test_settings_extended = extend_dict(
        test_settings,
        {
            "cfd_data": None,
            "cfd_chart": None,
        },
    )

    calculator = CFDCalculator(
        test_query_manager, test_settings_extended, base_minimal_cycle_time_results
    )
    calculator.run()
    # Should not raise an error
    calculator.write()


def test_write_chart_missing_backlog_column(
    test_query_manager, test_settings, base_minimal_cycle_time_results, tmp_path
):
    """Test write_chart() when backlog column is missing."""
    output_file = str(tmp_path / "cfd.png")
    test_settings_extended = extend_dict(
        test_settings,
        {
            "cfd_data": None,
            "cfd_chart": output_file,
            "backlog_column": "NonExistentColumn",
            "cfd_window": None,
        },
    )

    calculator = CFDCalculator(
        test_query_manager, test_settings_extended, base_minimal_cycle_time_results
    )
    result = calculator.run()
    base_minimal_cycle_time_results[CFDCalculator] = result
    calculator.write()

    # Chart should not be created when backlog column is missing
    assert not os.path.exists(output_file)
