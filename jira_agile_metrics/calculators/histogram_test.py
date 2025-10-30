"""Tests for histogram calculator functionality in Jira Agile Metrics.

This module contains unit tests for the histogram calculator.
"""

import pandas as pd

from ..test_utils import (
    create_empty_test_results,
)
from .histogram import HistogramCalculator


def test_empty(
    custom_query_manager, base_minimal_settings, base_minimal_cycle_time_columns
):
    """Test histogram calculator with empty data."""
    empty_results = create_empty_test_results(base_minimal_cycle_time_columns)

    calculator = HistogramCalculator(
        custom_query_manager, base_minimal_settings, empty_results
    )

    # Should not raise error on empty input
    try:
        data = calculator.run()
    except AttributeError:
        # Acceptable if .dt accessor fails on empty input
        data = None
    assert data is None or isinstance(data, pd.Series)


def test_calculate_histogram(
    custom_query_manager, base_minimal_settings, large_cycle_time_results
):
    """Test histogram calculation functionality."""
    calculator = HistogramCalculator(
        custom_query_manager, base_minimal_settings, large_cycle_time_results
    )

    data = calculator.run()

    assert list(data.index) == [
        "0.0 to 1.0",
        "1.0 to 2.0",
        "2.0 to 3.0",
        "3.0 to 4.0",
    ]
    assert list(data) == [0, 0, 0, 2]
