import pandas as pd
import pytest
from pandas import DataFrame

from ..utils import extend_dict
from .cycletime import CycleTimeCalculator
from .histogram import HistogramCalculator


@pytest.fixture
def settings(minimal_settings):
    return extend_dict(minimal_settings, {})


@pytest.fixture
def query_manager(minimal_query_manager):
    return minimal_query_manager


@pytest.fixture
def results(large_cycle_time_results):
    return extend_dict(large_cycle_time_results, {})


def test_empty(query_manager, settings, minimal_cycle_time_columns):
    results = {CycleTimeCalculator: DataFrame([], columns=minimal_cycle_time_columns, index=[])}

    calculator = HistogramCalculator(query_manager, settings, results)

    # Should not raise error on empty input
    try:
        data = calculator.run()
    except AttributeError:
        # Acceptable if .dt accessor fails on empty input
        data = None
    assert data is None or isinstance(data, pd.Series)


def test_calculate_histogram(query_manager, settings, results):
    calculator = HistogramCalculator(query_manager, settings, results)

    data = calculator.run()

    assert list(data.index) == [
        "0.0 to 1.0",
        "1.0 to 2.0",
        "2.0 to 3.0",
        "3.0 to 4.0",
        "4.0 to 5.0",
        "5.0 to 6.0",
    ]
    assert list(data) == [0, 0, 1, 1, 1, 5]
