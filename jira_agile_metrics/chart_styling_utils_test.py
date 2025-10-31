"""Tests for chart styling utilities."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from .chart_styling_utils import (
    _format_index_labels,
    apply_common_chart_styling,
    save_chart_with_styling,
    set_chart_style,
)


class TestSetChartStyle:
    """Test cases for set_chart_style function."""

    @patch("jira_agile_metrics.chart_styling_utils.sns")
    def test_set_chart_style_default(self, mock_sns):
        """Test setting default chart style."""
        set_chart_style()

        mock_sns.set_style.assert_called_once_with("whitegrid")
        mock_sns.despine.assert_called_once()

    @patch("jira_agile_metrics.chart_styling_utils.sns")
    def test_set_chart_style_custom(self, mock_sns):
        """Test setting custom chart style."""
        set_chart_style("darkgrid", despine=False)

        mock_sns.set_style.assert_called_once_with("darkgrid")
        mock_sns.despine.assert_not_called()


class TestApplyCommonChartStyling:
    """Test cases for apply_common_chart_styling function."""

    @patch("jira_agile_metrics.chart_styling_utils.set_chart_style")
    def test_apply_styling_with_datetime_index(self, _mock_set_style):
        """Test applying styling with datetime index."""
        ax = Mock()
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        breakdown = pd.Series([1, 2, 3], index=dates)

        apply_common_chart_styling(ax, breakdown)

        _mock_set_style.assert_called_once()
        ax.legend.assert_called_once()
        ax.set_xlabel.assert_called_once_with("Month", labelpad=20)
        ax.set_ylabel.assert_called_once_with("Number of items", labelpad=10)
        # Verify ticks are set to match data points
        ax.set_xticks.assert_called_once_with([0, 1, 2])
        ax.set_xticklabels.assert_called_once()

    @patch("jira_agile_metrics.chart_styling_utils.set_chart_style")
    def test_apply_styling_with_string_index(self, _mock_set_style):
        """Test applying styling with string index."""
        ax = Mock()
        breakdown = pd.Series([1, 2, 3], index=["Jan 24", "Feb 24", "Mar 24"])

        apply_common_chart_styling(ax, breakdown)

        _mock_set_style.assert_called_once()
        ax.set_xticks.assert_called_once_with([0, 1, 2])
        ax.set_xticklabels.assert_called_once()

    @patch("jira_agile_metrics.chart_styling_utils.set_chart_style")
    def test_apply_styling_with_numeric_index(self, _mock_set_style):
        """Test applying styling with numeric index."""
        ax = Mock()
        breakdown = pd.Series([1, 2, 3], index=[1, 2, 3])

        apply_common_chart_styling(ax, breakdown)

        _mock_set_style.assert_called_once()
        ax.set_xticks.assert_called_once_with([0, 1, 2])
        ax.set_xticklabels.assert_called_once()

    @patch("jira_agile_metrics.chart_styling_utils.set_chart_style")
    def test_apply_styling_ticks_match_data_points(self, _mock_set_style):
        """Test that ticks are set to match data points exactly."""
        ax = Mock()
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        breakdown = pd.Series([1, 2, 3, 4, 5], index=dates)

        apply_common_chart_styling(ax, breakdown)

        # Verify ticks are set to match all data points (0-indexed)
        ax.set_xticks.assert_called_once_with([0, 1, 2, 3, 4])
        # Verify labels match data points
        call_args = ax.set_xticklabels.call_args[0][0]
        assert len(call_args) == 5

    def test_apply_styling_no_index_attribute(self):
        """Test error when breakdown has no index attribute."""
        ax = Mock()
        breakdown = Mock(spec=[])  # No index attribute

        with pytest.raises(TypeError, match="breakdown must have an 'index' attribute"):
            apply_common_chart_styling(ax, breakdown)

    @patch("jira_agile_metrics.chart_styling_utils.set_chart_style")
    def test_apply_styling_mixed_date_formats(self, _mock_set_style):
        """Test handling of mixed date formats in index."""
        ax = Mock()
        # Mix of datetime-like and string values
        breakdown = pd.Series(
            [1, 2, 3],
            index=[
                pd.Timestamp("2024-01-01"),
                "invalid",
                pd.Timestamp("2024-01-03"),
            ],
        )

        # Should handle gracefully
        apply_common_chart_styling(ax, breakdown)

        # Verify ticks match data points
        ax.set_xticks.assert_called_once_with([0, 1, 2])
        ax.set_xticklabels.assert_called_once()
        # Verify labels include formatted dates and string fallback
        call_args = ax.set_xticklabels.call_args[0][0]
        assert len(call_args) == 3


class TestFormatIndexLabels:
    """Test cases for _format_index_labels helper function."""

    def test_format_datetime_index(self):
        """Test formatting a datetime index."""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        labels = _format_index_labels(dates)

        assert len(labels) == 3
        assert all(isinstance(label, str) for label in labels)
        # Check format is "Mon YY" (e.g., "Jan 24")
        assert all(len(label.split()) == 2 for label in labels)

    def test_format_string_index(self):
        """Test formatting a string index falls back to string representation."""
        index = ["Jan 24", "Feb 24", "Mar 24"]
        labels = _format_index_labels(index)

        assert len(labels) == 3
        assert labels == ["Jan 24", "Feb 24", "Mar 24"]

    def test_format_numeric_index(self):
        """Test formatting a numeric index falls back to string representation."""
        index = [1, 2, 3]
        labels = _format_index_labels(index)

        assert len(labels) == 3
        assert labels == ["1", "2", "3"]

    def test_format_mixed_datetime_types(self):
        """Test formatting with mix of datetime-like objects."""
        index = [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-02-01"),
            "invalid",
        ]
        labels = _format_index_labels(index)

        assert len(labels) == 3
        # First two should be formatted dates
        assert all(isinstance(label, str) for label in labels)
        # Last should be string representation of "invalid"
        assert labels[2] == "invalid"

    def test_format_pandas_series_index(self):
        """Test formatting index from a pandas Series."""
        date_range = pd.date_range("2024-01-01", periods=3, freq="ME")
        series = pd.Series([1, 2, 3], index=date_range)
        labels = _format_index_labels(series.index)

        assert len(labels) == 3
        assert all(isinstance(label, str) for label in labels)


class TestSaveChartWithStyling:
    """Test cases for save_chart_with_styling function."""

    @patch("jira_agile_metrics.chart_styling_utils.plt")
    @patch("jira_agile_metrics.chart_styling_utils.logging.getLogger")
    def test_save_chart_success(self, mock_get_logger, mock_plt):
        """Test successful chart saving."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_fig = Mock()

        save_chart_with_styling(mock_fig, "output.png", "Test Chart")

        mock_fig.savefig.assert_called_once_with(
            "output.png", bbox_inches="tight", dpi=300
        )
        mock_plt.close.assert_called_once_with(mock_fig)
        mock_logger.info.assert_called_once_with(
            "Writing %s chart to %s", "Test Chart", "output.png"
        )

    @patch("jira_agile_metrics.chart_styling_utils.plt")
    @patch("jira_agile_metrics.chart_styling_utils.logging.getLogger")
    def test_save_chart_default_title(self, mock_get_logger, _mock_plt):
        """Test saving chart with default title."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_fig = Mock()

        save_chart_with_styling(mock_fig, "output.png")

        mock_logger.info.assert_called_once_with(
            "Writing %s chart to %s", "Chart", "output.png"
        )

    @patch("jira_agile_metrics.chart_styling_utils.plt")
    @patch("jira_agile_metrics.chart_styling_utils.logging.getLogger")
    def test_save_chart_with_path(self, mock_get_logger, _mock_plt):
        """Test saving chart to a file path."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_fig = Mock()

        save_chart_with_styling(mock_fig, "/path/to/output.png", "Test Chart")

        mock_fig.savefig.assert_called_once_with(
            "/path/to/output.png", bbox_inches="tight", dpi=300
        )
