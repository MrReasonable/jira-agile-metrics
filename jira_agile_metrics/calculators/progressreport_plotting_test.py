"""Tests for progressreport_plotting module."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from .progressreport_plotting import (
    _add_deadline_to_cfd,
    _add_target_value_to_cfd,
    plot_cfd,
    plot_scatterplot,
    plot_throughput,
)


class TestAddDeadlineToCfd:
    """Test cases for _add_deadline_to_cfd function."""

    def test_add_deadline(self):
        """Test adding deadline line to CFD."""
        ax = Mock()
        deadline = pd.Timestamp("2024-12-31")
        target = None

        _add_deadline_to_cfd(ax, deadline, target)

        ax.axvline.assert_called_once()
        call_kwargs = ax.axvline.call_args[1]
        assert call_kwargs["x"] == deadline
        assert call_kwargs["color"] == "red"
        assert call_kwargs["linestyle"] == "--"

    def test_add_target_date(self):
        """Test adding target date line to CFD."""
        ax = Mock()
        deadline = None
        target = pd.Timestamp("2024-12-31")

        _add_deadline_to_cfd(ax, deadline, target)

        ax.axvline.assert_called_once()
        call_kwargs = ax.axvline.call_args[1]
        assert call_kwargs["x"] == target
        assert call_kwargs["color"] == "orange"

    def test_add_both_dates(self):
        """Test adding both deadline and target dates."""
        ax = Mock()
        deadline = pd.Timestamp("2024-12-31")
        target = pd.Timestamp("2024-11-30")

        _add_deadline_to_cfd(ax, deadline, target)

        assert ax.axvline.call_count == 2

    def test_add_no_dates(self):
        """Test adding no dates (both None)."""
        ax = Mock()
        deadline = None
        target = None

        _add_deadline_to_cfd(ax, deadline, target)

        ax.axvline.assert_not_called()


class TestAddTargetValueToCfd:
    """Test cases for _add_target_value_to_cfd function."""

    def test_add_target(self):
        """Test adding target line to CFD."""
        ax = Mock()
        target = 100

        _add_target_value_to_cfd(ax, target)

        ax.axhline.assert_called_once()
        call_kwargs = ax.axhline.call_args[1]
        assert call_kwargs["y"] == 100
        assert call_kwargs["color"] == "orange"
        assert call_kwargs["linestyle"] == ":"

    def test_add_target_none(self):
        """Test adding no target (None)."""
        ax = Mock()
        target = None

        _add_target_value_to_cfd(ax, target)

        ax.axhline.assert_not_called()

    def test_add_target_zero(self):
        """Test adding target with value 0 (should still add line)."""
        ax = Mock()
        target = 0

        _add_target_value_to_cfd(ax, target)

        ax.axhline.assert_called_once()
        call_kwargs = ax.axhline.call_args[1]
        assert call_kwargs["y"] == 0


class TestPlotCfd:
    """Test cases for plot_cfd function."""

    @pytest.fixture
    def cycle_data(self):
        """Create sample cycle data."""
        return pd.DataFrame(
            [
                {
                    "key": "ISSUE-1",
                    "status": "Done",
                    "started": pd.Timestamp("2024-01-01"),
                    "completed": pd.Timestamp("2024-01-10"),
                },
                {
                    "key": "ISSUE-2",
                    "status": "Done",
                    "started": pd.Timestamp("2024-01-05"),
                    "completed": pd.Timestamp("2024-01-15"),
                },
            ]
        )

    @pytest.fixture
    def plot_config(self):
        """Create sample plot configuration."""
        return {
            "cycle_names": {
                "Backlog": ["Backlog"],
                "Committed": ["Committed"],
                "Done": ["Done"],
            },
            "deadline": None,
            "target": None,
        }

    def test_plot_cfd_basic(self, cycle_data, plot_config):
        """Test basic CFD plotting."""
        mock_cfd_data = pd.DataFrame(
            {
                "Backlog": [10, 9, 8],
                "Committed": [5, 6, 7],
                "Done": [0, 1, 2],
            },
            index=pd.date_range("2024-01-01", periods=3),
        )

        with (
            patch(
                "jira_agile_metrics.calculators.progressreport_plotting.set_chart_style"
            ),
            patch(
                "jira_agile_metrics.calculators.progressreport_plotting.calculate_cfd_data",
                return_value=mock_cfd_data,
            ) as mock_calc_cfd,
            patch(
                "jira_agile_metrics.calculators.progressreport_plotting.plt"
            ) as mock_plt,
        ):
            mock_fig = Mock()
            mock_ax = Mock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)

            def _save_png(buf, **_kwargs):
                buf.write(b"fake_png_data")

            mock_fig.savefig = _save_png

            result = plot_cfd(cycle_data, plot_config)

            assert isinstance(result, str)
            assert len(result) > 0  # Base64 string should not be empty
            mock_calc_cfd.assert_called_once()
            mock_ax.plot.assert_called()
            mock_plt.close.assert_called_once()

    @patch("jira_agile_metrics.calculators.progressreport_plotting.plt")
    @patch("jira_agile_metrics.calculators.progressreport_plotting.calculate_cfd_data")
    def test_plot_cfd_with_target(
        self, mock_calc_cfd, mock_plt, cycle_data, plot_config
    ):
        """Test CFD plotting with target value."""
        mock_cfd_data = pd.DataFrame(
            {
                "Backlog": [10, 9, 8],
                "Done": [0, 1, 2],
            },
            index=pd.date_range("2024-01-01", periods=3),
        )
        mock_calc_cfd.return_value = mock_cfd_data
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plot_cfd(cycle_data, plot_config, target=100)

        mock_ax.axhline.assert_called_once()

    @patch("jira_agile_metrics.calculators.progressreport_plotting.plt")
    @patch("jira_agile_metrics.calculators.progressreport_plotting.calculate_cfd_data")
    def test_plot_cfd_with_deadline(self, mock_calc_cfd, mock_plt, cycle_data):
        """Test CFD plotting with deadline."""
        mock_cfd_data = pd.DataFrame(
            {
                "Backlog": [10, 9, 8],
                "Done": [0, 1, 2],
            },
            index=pd.date_range("2024-01-01", periods=3),
        )
        mock_calc_cfd.return_value = mock_cfd_data
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plot_config = {
            "cycle_names": {
                "Backlog": ["Backlog"],
                "Done": ["Done"],
            },
            "deadline": pd.Timestamp("2024-12-31"),
            "target": None,
        }

        plot_cfd(cycle_data, plot_config)

        mock_ax.axvline.assert_called()


class TestPlotThroughput:
    """Test cases for plot_throughput function."""

    @pytest.fixture
    def cycle_data(self):
        """Create sample cycle data."""
        return pd.DataFrame(
            [
                {
                    "key": "ISSUE-1",
                    "status": "Done",
                    "completed": pd.Timestamp("2024-01-01"),
                },
                {
                    "key": "ISSUE-2",
                    "status": "Done",
                    "completed": pd.Timestamp("2024-01-02"),
                },
                {
                    "key": "ISSUE-3",
                    "status": "Done",
                    "completed": pd.Timestamp("2024-01-03"),
                },
            ]
        )

    @patch("jira_agile_metrics.calculators.progressreport_plotting.plt")
    @patch(
        "jira_agile_metrics.calculators.progressreport_plotting.calculate_throughput"
    )
    @patch("jira_agile_metrics.calculators.progressreport_plotting.set_chart_style")
    def test_plot_throughput_basic(
        self, _mock_set_style, mock_calc, mock_plt, cycle_data
    ):
        """Test basic throughput plotting."""
        throughput_data = pd.Series(
            [1, 2, 3], index=pd.date_range("2024-01-01", periods=3)
        )
        mock_calc.return_value = throughput_data
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        def _save_png(buf, **_kwargs):
            buf.write(b"fake_png_data")

        mock_fig.savefig = _save_png

        result = plot_throughput(cycle_data)

        assert isinstance(result, str)
        assert len(result) > 0
        mock_calc.assert_called_once()
        mock_ax.bar.assert_called_once()
        mock_ax.set_title.assert_called_once_with("Throughput Over Time")
        mock_plt.close.assert_called_once()

    @patch("jira_agile_metrics.calculators.progressreport_plotting.plt")
    @patch(
        "jira_agile_metrics.calculators.progressreport_plotting.calculate_throughput"
    )
    def test_plot_throughput_custom_frequency(self, mock_calc, mock_plt, cycle_data):
        """Test throughput plotting with custom frequency."""
        throughput_data = pd.Series(
            [1, 2], index=pd.date_range("2024-01-01", periods=2, freq="W")
        )
        mock_calc.return_value = throughput_data
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plot_throughput(cycle_data, frequency="1W")

        mock_calc.assert_called_once_with(cycle_data, "1W")


class TestPlotScatterplot:
    """Test cases for plot_scatterplot function."""

    @pytest.fixture
    def cycle_data(self):
        """Create sample cycle data."""
        return pd.DataFrame(
            [
                {
                    "key": "ISSUE-1",
                    "cycle_time": pd.Timedelta(days=5),
                    "lead_time": pd.Timedelta(days=10),
                },
                {
                    "key": "ISSUE-2",
                    "cycle_time": pd.Timedelta(days=7),
                    "lead_time": pd.Timedelta(days=12),
                },
                {
                    "key": "ISSUE-3",
                    "cycle_time": pd.Timedelta(days=3),
                    "lead_time": pd.Timedelta(days=8),
                },
            ]
        )

    @patch("jira_agile_metrics.calculators.progressreport_plotting.plt")
    @patch(
        "jira_agile_metrics.calculators.progressreport_plotting."
        "calculate_scatterplot_data"
    )
    @patch("jira_agile_metrics.calculators.progressreport_plotting.set_chart_style")
    def test_plot_scatterplot_basic(
        self, _mock_set_style, mock_calc, mock_plt, cycle_data
    ):
        """Test basic scatterplot plotting."""
        scatterplot_data = pd.DataFrame(
            {
                "cycle_time": pd.TimedeltaIndex(
                    [pd.Timedelta(days=x) for x in [5, 7, 3]]
                ),
                "lead_time": pd.TimedeltaIndex(
                    [pd.Timedelta(days=x) for x in [10, 12, 8]]
                ),
            }
        )
        mock_calc.return_value = scatterplot_data
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        def _save_png(buf, **_kwargs):
            buf.write(b"fake_png_data")

        mock_fig.savefig = _save_png

        result = plot_scatterplot(cycle_data, [0.5, 0.85, 0.95])

        assert isinstance(result, str)
        assert len(result) > 0
        mock_calc.assert_called_once_with(cycle_data)
        mock_ax.scatter.assert_called_once()
        mock_ax.set_title.assert_called_once_with("Cycle Time vs Lead Time")
        mock_plt.close.assert_called_once()

    @patch("jira_agile_metrics.calculators.progressreport_plotting.plt")
    @patch(
        "jira_agile_metrics.calculators.progressreport_plotting."
        "calculate_scatterplot_data"
    )
    def test_plot_scatterplot_with_quantiles(self, mock_calc, mock_plt, cycle_data):
        """Test scatterplot plotting with quantiles."""
        scatterplot_data = pd.DataFrame(
            {
                "cycle_time": pd.TimedeltaIndex(
                    [pd.Timedelta(days=x) for x in [5, 7, 3]]
                ),
                "lead_time": pd.TimedeltaIndex(
                    [pd.Timedelta(days=x) for x in [10, 12, 8]]
                ),
            }
        )
        mock_calc.return_value = scatterplot_data
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        quantiles = [0.5, 0.85, 0.95]
        plot_scatterplot(cycle_data, quantiles)

        # Should call axhline for each quantile
        assert mock_ax.axhline.call_count == len(quantiles)

    @patch("jira_agile_metrics.calculators.progressreport_plotting.plt")
    @patch(
        "jira_agile_metrics.calculators.progressreport_plotting."
        "calculate_scatterplot_data"
    )
    def test_plot_scatterplot_no_lead_time(self, mock_calc, mock_plt, cycle_data):
        """Test scatterplot plotting without lead_time column."""
        scatterplot_data = pd.DataFrame(
            {
                "cycle_time": pd.TimedeltaIndex(
                    [pd.Timedelta(days=x) for x in [5, 7, 3]]
                ),
            }
        )
        mock_calc.return_value = scatterplot_data
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plot_scatterplot(cycle_data, [0.5])

        # Should use cycle_time as fallback
        mock_ax.scatter.assert_called_once()

    @patch("jira_agile_metrics.calculators.progressreport_plotting.plt")
    @patch(
        "jira_agile_metrics.calculators.progressreport_plotting."
        "calculate_scatterplot_data"
    )
    def test_plot_scatterplot_empty_data(self, mock_calc, mock_plt):
        """Test scatterplot plotting with empty data."""
        scatterplot_data = pd.DataFrame(
            {
                "cycle_time": pd.TimedeltaIndex([]),
            }
        )
        mock_calc.return_value = scatterplot_data
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plot_scatterplot(pd.DataFrame(), [0.5])

        # Should not call axhline when no data
        mock_ax.axhline.assert_not_called()
