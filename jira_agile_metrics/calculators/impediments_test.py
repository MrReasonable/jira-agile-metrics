"""Tests for impediments calculator functionality in Jira Agile Metrics.

This module contains unit tests for the impediments calculator.
"""

import os

import pytest
from pandas import DataFrame, NaT, Timestamp

from ..test_data_factory import create_impediments_cycle_time_results
from ..utils import extend_dict
from .cycletime import CycleTimeCalculator
from .impediments import ImpedimentsCalculator


def _ts(datestring, timestring="00:00:00"):
    return Timestamp(f"{datestring} {timestring}")


@pytest.fixture(name="query_manager")
def fixture_query_manager(minimal_query_manager):
    """Provide query manager fixture for impediments tests."""
    return minimal_query_manager


@pytest.fixture(name="settings")
def fixture_settings(base_minimal_settings):
    """Provide settings fixture for impediments tests."""
    return extend_dict(
        base_minimal_settings,
        {
            # Set output paths to None by default to prevent files from being
            # written to project root. Tests that need to write files should
            # override these with tmp_path-based paths.
            "impediments_data": None,
            "impediments_chart": None,
            "impediments_days_chart": None,
            "impediments_status_chart": None,
            "impediments_status_days_chart": None,
        },
    )


@pytest.fixture(name="columns")
def fixture_columns(base_minimal_cycle_time_columns):
    """Provide columns fixture for impediments tests."""
    return base_minimal_cycle_time_columns


@pytest.fixture(name="cycle_time_results")
def fixture_cycle_time_results(base_minimal_cycle_time_columns):
    """A results dict mimicing a minimal result
    from the CycleTimeCalculator."""
    return create_impediments_cycle_time_results(base_minimal_cycle_time_columns)


@pytest.mark.parametrize(
    "chart_config,expected_runs",
    [
        (
            {
                "impediments_data": None,
                "impediments_chart": None,
                "impediments_days_chart": None,
                "impediments_status_chart": None,
                "impediments_status_days_chart": None,
            },
            False,
        ),
        (
            {
                "impediments_data": "impediments.csv",
                "impediments_chart": None,
                "impediments_days_chart": None,
                "impediments_status_chart": None,
                "impediments_status_days_chart": None,
            },
            True,
        ),
        (
            {
                "impediments_data": None,
                "impediments_chart": "impediments.png",
                "impediments_days_chart": None,
                "impediments_status_chart": None,
                "impediments_status_days_chart": None,
            },
            True,
        ),
        (
            {
                "impediments_data": None,
                "impediments_chart": None,
                "impediments_days_chart": "days.png",
                "impediments_status_chart": None,
                "impediments_status_days_chart": None,
            },
            True,
        ),
        (
            {
                "impediments_data": None,
                "impediments_chart": None,
                "impediments_days_chart": None,
                "impediments_status_chart": "status.png",
                "impediments_status_days_chart": None,
            },
            True,
        ),
        (
            {
                "impediments_data": None,
                "impediments_chart": None,
                "impediments_days_chart": None,
                "impediments_status_chart": None,
                "impediments_status_days_chart": "status-days.png",
            },
            True,
        ),
    ],
)
def test_only_runs_if_charts_set(
    query_manager, settings, cycle_time_results, chart_config, expected_runs
):
    """Test that impediments calculator only runs when charts are configured."""
    test_settings = extend_dict(settings, chart_config)

    calculator = ImpedimentsCalculator(query_manager, test_settings, cycle_time_results)
    data = calculator.run()

    assert (data is not None) == expected_runs


def test_empty(query_manager, settings, columns):
    """Test impediments calculator with empty data."""
    results = {CycleTimeCalculator: DataFrame([], columns=columns)}
    # Configure output so calculator runs
    settings_with_output = extend_dict(
        settings, {"impediments_data": "impediments.csv"}
    )

    calculator = ImpedimentsCalculator(query_manager, settings_with_output, results)

    data = calculator.run()
    assert len(data.index) == 0


def test_columns(query_manager, settings, cycle_time_results):
    """Test impediments calculator column structure."""
    # Configure output so calculator runs
    settings_with_output = extend_dict(
        settings, {"impediments_data": "impediments.csv"}
    )
    calculator = ImpedimentsCalculator(
        query_manager, settings_with_output, cycle_time_results
    )

    data = calculator.run()

    assert list(data.columns) == ["key", "status", "flag", "start", "end"]


def test_calculate_impediments(query_manager, settings, cycle_time_results):
    """Test impediments calculator functionality."""
    # Configure output so calculator runs
    settings_with_output = extend_dict(
        settings, {"impediments_data": "impediments.csv"}
    )
    calculator = ImpedimentsCalculator(
        query_manager, settings_with_output, cycle_time_results
    )

    data = calculator.run()

    assert data.to_dict("records") == [
        {
            "key": "A-2",
            "status": "Committed",
            "flag": "Impediment",
            "start": _ts("2018-01-10"),
            "end": _ts("2018-01-12"),
        },
        {
            "key": "A-3",
            "status": "Build",
            "flag": "Impediment",
            "start": _ts("2018-01-04"),
            "end": _ts("2018-01-05"),
        },
        {
            "key": "A-4",
            "status": "Committed",
            "flag": "Awaiting input",
            "start": _ts("2018-01-05"),
            "end": NaT,
        },
    ]


def test_different_backlog_column(query_manager, settings, cycle_time_results):
    """Test impediments calculator with different backlog column."""
    settings = extend_dict(
        settings,
        {
            "backlog_column": "Committed",
            "committed_column": "Build",
            # Configure output so calculator runs
            "impediments_data": "impediments.csv",
        },
    )
    calculator = ImpedimentsCalculator(query_manager, settings, cycle_time_results)

    data = calculator.run()

    assert data.to_dict("records") == [
        {
            "key": "A-3",
            "status": "Build",
            "flag": "Impediment",
            "start": _ts("2018-01-04"),
            "end": _ts("2018-01-05"),
        },
    ]


def test_different_done_column(query_manager, settings, cycle_time_results):
    """Test impediments calculator with different done column."""
    settings = extend_dict(
        settings,
        {
            "done_column": "Build",
            # Configure output so calculator runs
            "impediments_data": "impediments.csv",
        },
    )
    calculator = ImpedimentsCalculator(query_manager, settings, cycle_time_results)

    data = calculator.run()

    assert data.to_dict("records") == [
        {
            "key": "A-2",
            "status": "Committed",
            "flag": "Impediment",
            "start": _ts("2018-01-10"),
            "end": _ts("2018-01-12"),
        },
        {
            "key": "A-4",
            "status": "Committed",
            "flag": "Awaiting input",
            "start": _ts("2018-01-05"),
            "end": NaT,
        },
    ]


class TestImpedimentsWrite:
    """Test cases for ImpedimentsCalculator write methods."""

    def test_write_data_csv(
        self, query_manager, settings, cycle_time_results, tmp_path
    ):
        """Test write_data() with CSV format."""
        output_file = str(tmp_path / "impediments.csv")
        test_settings = extend_dict(settings, {"impediments_data": [output_file]})

        calculator = ImpedimentsCalculator(
            query_manager, test_settings, cycle_time_results
        )
        data = calculator.run()
        assert data is not None and len(data) > 0, "Data should not be empty"
        # Store result so write() can retrieve it
        cycle_time_results[ImpedimentsCalculator] = data
        calculator.write()

        assert os.path.exists(output_file)

    def test_write_data_json(
        self, query_manager, settings, cycle_time_results, tmp_path
    ):
        """Test write_data() with JSON format."""
        output_file = str(tmp_path / "impediments.json")
        test_settings = extend_dict(settings, {"impediments_data": [output_file]})

        calculator = ImpedimentsCalculator(
            query_manager, test_settings, cycle_time_results
        )
        data = calculator.run()
        assert data is not None and len(data) > 0, "Data should not be empty"
        # Store result so write() can retrieve it
        cycle_time_results[ImpedimentsCalculator] = data
        calculator.write()

        assert os.path.exists(output_file)

    def test_write_data_xlsx(
        self, query_manager, settings, cycle_time_results, tmp_path
    ):
        """Test write_data() with XLSX format."""
        output_file = str(tmp_path / "impediments.xlsx")
        test_settings = extend_dict(settings, {"impediments_data": [output_file]})

        calculator = ImpedimentsCalculator(
            query_manager, test_settings, cycle_time_results
        )
        data = calculator.run()
        assert data is not None and len(data) > 0, "Data should not be empty"
        # Store result so write() can retrieve it
        cycle_time_results[ImpedimentsCalculator] = data
        calculator.write()

        assert os.path.exists(output_file)

    def test_write_impediments_chart(
        self, query_manager, settings, cycle_time_results, tmp_path
    ):
        """Test write_impediments_chart()."""
        output_file = str(tmp_path / "impediments.png")
        test_settings = extend_dict(
            settings,
            {
                "impediments_chart": output_file,
                "impediments_chart_title": "Test Impediments Chart",
            },
        )

        calculator = ImpedimentsCalculator(
            query_manager, test_settings, cycle_time_results
        )
        data = calculator.run()
        assert data is not None and len(data) > 0, "Data should not be empty"
        # Store result so write() can retrieve it
        cycle_time_results[ImpedimentsCalculator] = data
        calculator.write()

        assert os.path.exists(output_file)

    def test_write_impediments_chart_empty_data(
        self, query_manager, settings, columns, tmp_path
    ):
        """Test write_impediments_chart() with empty data."""
        results = {CycleTimeCalculator: DataFrame([], columns=columns)}
        output_file = str(tmp_path / "impediments.png")
        test_settings = extend_dict(settings, {"impediments_chart": output_file})

        calculator = ImpedimentsCalculator(query_manager, test_settings, results)
        calculator.run()
        calculator.write()

        # File should not be created when data is empty
        assert not os.path.exists(output_file)

    def test_write_impediments_chart_with_window(
        self, query_manager, settings, cycle_time_results, tmp_path
    ):
        """Test write_impediments_chart() with window filtering."""
        output_file = str(tmp_path / "impediments.png")
        test_settings = extend_dict(
            settings,
            {
                "impediments_chart": output_file,
                "impediments_window": 3,
            },
        )

        calculator = ImpedimentsCalculator(
            query_manager, test_settings, cycle_time_results
        )
        data = calculator.run()
        assert data is not None and len(data) > 0, "Data should not be empty"
        # Store result so write() can retrieve it
        cycle_time_results[ImpedimentsCalculator] = data
        calculator.write()

        assert os.path.exists(output_file)

    def test_write_impediments_days_chart(
        self, query_manager, settings, cycle_time_results, tmp_path
    ):
        """Test write_impediments_days_chart()."""
        output_file = str(tmp_path / "impediments-days.png")
        test_settings = extend_dict(
            settings,
            {
                "impediments_days_chart": output_file,
                "impediments_days_chart_title": "Test Impediments Days Chart",
            },
        )

        calculator = ImpedimentsCalculator(
            query_manager, test_settings, cycle_time_results
        )
        data = calculator.run()
        assert data is not None and len(data) > 0, "Data should not be empty"
        # Store result so write() can retrieve it
        cycle_time_results[ImpedimentsCalculator] = data
        calculator.write()

        assert os.path.exists(output_file)

    def test_write_impediments_days_chart_empty_data(
        self, query_manager, settings, columns, tmp_path
    ):
        """Test write_impediments_days_chart() with empty data."""
        results = {CycleTimeCalculator: DataFrame([], columns=columns)}
        output_file = str(tmp_path / "impediments-days.png")
        test_settings = extend_dict(settings, {"impediments_days_chart": output_file})

        calculator = ImpedimentsCalculator(query_manager, test_settings, results)
        calculator.run()
        calculator.write()

        # File should not be created when data is empty
        assert not os.path.exists(output_file)

    def test_write_impediments_status_chart(
        self, query_manager, settings, cycle_time_results, tmp_path
    ):
        """Test write_impediments_status_chart()."""
        output_file = str(tmp_path / "impediments-status.png")
        test_settings = extend_dict(
            settings,
            {
                "impediments_status_chart": output_file,
                "impediments_status_chart_title": "Test Impediments Status Chart",
            },
        )

        calculator = ImpedimentsCalculator(
            query_manager, test_settings, cycle_time_results
        )
        data = calculator.run()
        assert data is not None and len(data) > 0, "Data should not be empty"
        # Store result so write() can retrieve it
        cycle_time_results[ImpedimentsCalculator] = data
        calculator.write()

        assert os.path.exists(output_file)

    def test_write_impediments_status_chart_empty_data(
        self, query_manager, settings, columns, tmp_path
    ):
        """Test write_impediments_status_chart() with empty data."""
        results = {CycleTimeCalculator: DataFrame([], columns=columns)}
        output_file = str(tmp_path / "impediments-status.png")
        test_settings = extend_dict(settings, {"impediments_status_chart": output_file})

        calculator = ImpedimentsCalculator(query_manager, test_settings, results)
        calculator.run()
        calculator.write()

        # File should not be created when data is empty
        assert not os.path.exists(output_file)

    def test_write_impediments_status_days_chart(
        self, query_manager, settings, cycle_time_results, tmp_path
    ):
        """Test write_impediments_status_days_chart()."""
        output_file = str(tmp_path / "impediments-status-days.png")
        test_settings = extend_dict(
            settings,
            {
                "impediments_status_days_chart": output_file,
                "impediments_status_days_chart_title": (
                    "Test Impediments Status Days Chart"
                ),
            },
        )

        calculator = ImpedimentsCalculator(
            query_manager, test_settings, cycle_time_results
        )
        data = calculator.run()
        assert data is not None and len(data) > 0, "Data should not be empty"
        # Store result so write() can retrieve it
        cycle_time_results[ImpedimentsCalculator] = data
        calculator.write()

        assert os.path.exists(output_file)

    def test_write_impediments_status_days_chart_empty_data(
        self, query_manager, settings, columns, tmp_path
    ):
        """Test write_impediments_status_days_chart() with empty data."""
        results = {CycleTimeCalculator: DataFrame([], columns=columns)}
        output_file = str(tmp_path / "impediments-status-days.png")
        test_settings = extend_dict(
            settings, {"impediments_status_days_chart": output_file}
        )

        calculator = ImpedimentsCalculator(query_manager, test_settings, results)
        calculator.run()
        calculator.write()

        # File should not be created when data is empty
        assert not os.path.exists(output_file)

    def test_write_all_outputs(
        self, query_manager, settings, cycle_time_results, tmp_path
    ):
        """Test writing all output types at once."""
        test_settings = extend_dict(
            settings,
            {
                "impediments_data": [str(tmp_path / "impediments.csv")],
                "impediments_chart": str(tmp_path / "impediments.png"),
                "impediments_days_chart": str(tmp_path / "impediments-days.png"),
                "impediments_status_chart": str(tmp_path / "impediments-status.png"),
                "impediments_status_days_chart": str(
                    tmp_path / "impediments-status-days.png"
                ),
            },
        )

        calculator = ImpedimentsCalculator(
            query_manager, test_settings, cycle_time_results
        )
        data = calculator.run()
        assert data is not None and len(data) > 0, "Data should not be empty"
        # Store result so write() can retrieve it
        cycle_time_results[ImpedimentsCalculator] = data
        calculator.write()

        # All files should be created
        assert os.path.exists(tmp_path / "impediments.csv")
        assert os.path.exists(tmp_path / "impediments.png")
        assert os.path.exists(tmp_path / "impediments-days.png")
        assert os.path.exists(tmp_path / "impediments-status.png")
        assert os.path.exists(tmp_path / "impediments-status-days.png")
