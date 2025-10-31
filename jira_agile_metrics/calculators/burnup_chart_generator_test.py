"""Tests for burnup_chart_generator module."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from .burnup_chart_generator import BurnupChartGenerator


class TestBurnupChartGeneratorBase:
    """Base class with shared fixtures for BurnupChartGenerator tests."""

    @pytest.fixture
    def generator(self, tmp_path):
        """Create a BurnupChartGenerator instance."""
        output_file = tmp_path / "burnup.png"
        return BurnupChartGenerator(str(output_file))

    @pytest.fixture
    def burnup_data(self):
        """Create sample burnup data."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        return pd.DataFrame(
            {
                "Backlog": [100, 95, 90, 85, 80, 75, 70, 65, 60, 55],
                "Done": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
            },
            index=dates,
        )

    @pytest.fixture
    def chart_data(self):
        """Create sample chart data."""
        dates = pd.date_range("2024-01-11", periods=5, freq="D")
        return {
            "forecast_dates": dates.tolist(),
            "backlog_trials": [[50, 48, 45, 42, 40] for _ in range(10)],
            "done_trials": [[45, 50, 55, 60, 65] for _ in range(10)],
            "trust_metrics": {"confidence": 0.85, "accuracy": 0.92},
            "target": 100,
            "quantile_data": {
                "50%": dates[2],
                "75%": dates[3],
                "95%": dates[4],
            },
        }


class TestBurnupChartGeneratorInit(TestBurnupChartGeneratorBase):
    """Test cases for BurnupChartGenerator initialization and validation."""

    def test_init(self, tmp_path):
        """Test BurnupChartGenerator initialization."""
        output_file = tmp_path / "burnup.png"
        generator = BurnupChartGenerator(str(output_file))

        assert generator.output_file == str(output_file)
        assert generator.figure is None
        assert generator.axis is None

    def test_validate_chart_data_valid(self, generator, chart_data):
        """Test validating valid chart data."""
        assert generator.validate_chart_data(chart_data) is True

    def test_validate_chart_data_missing_forecast_dates(self, generator):
        """Test validating chart data with missing forecast_dates."""
        chart_data = {
            "backlog_trials": [],
            "done_trials": [],
        }
        assert generator.validate_chart_data(chart_data) is False

    def test_validate_chart_data_missing_backlog_trials(self, generator):
        """Test validating chart data with missing backlog_trials."""
        chart_data = {
            "forecast_dates": [],
            "done_trials": [],
        }
        assert generator.validate_chart_data(chart_data) is False

    def test_validate_chart_data_missing_done_trials(self, generator):
        """Test validating chart data with missing done_trials."""
        chart_data = {
            "forecast_dates": [],
            "backlog_trials": [],
        }
        assert generator.validate_chart_data(chart_data) is False

    def test_get_chart_info(self, generator):
        """Test getting chart info."""
        info = generator.get_chart_info()

        assert "output_file" in info
        assert "has_figure" in info
        assert "has_axis" in info
        assert info["has_figure"] is False
        assert info["has_axis"] is False

    def test_get_chart_info_with_figure(self, generator):
        """Test getting chart info when figure exists."""
        generator.figure = Mock()
        generator.axis = Mock()

        info = generator.get_chart_info()

        assert info["has_figure"] is True
        assert info["has_axis"] is True


class TestBurnupChartGeneratorChartGeneration(TestBurnupChartGeneratorBase):
    """Test cases for chart generation methods."""

    @pytest.fixture
    def chart_test_data(self, generator, burnup_data, chart_data):
        """Combine test data for chart generation tests."""
        return {
            "generator": generator,
            "burnup_data": burnup_data,
            "chart_data": chart_data,
        }

    @patch(
        "jira_agile_metrics.calculators.burnup_chart_generator."
        "find_backlog_and_done_columns"
    )
    @patch("jira_agile_metrics.calculators.burnup_chart_generator.set_chart_style")
    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_generate_chart_success(
        self,
        mock_plt,
        _mock_set_style,
        mock_find_columns,
        chart_test_data,
    ):
        """Test successful chart generation."""
        mock_find_columns.return_value = ("Backlog", "Done")
        mock_fig = Mock()
        # savefig will be a mock via mock_fig
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        result = chart_test_data["generator"].generate_chart(
            chart_test_data["burnup_data"], chart_test_data["chart_data"]
        )

        assert result is True
        mock_plt.subplots.assert_called_once()
        mock_ax.plot.assert_called()
        mock_fig.savefig.assert_called_once()

    @patch(
        "jira_agile_metrics.calculators.burnup_chart_generator."
        "find_backlog_and_done_columns"
    )
    def test_generate_chart_no_data(self, _mock_find_columns, generator):
        """Test chart generation with no data."""
        result = generator.generate_chart(pd.DataFrame(), {})

        assert result is False

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_generate_chart_plt_error(self, mock_plt, chart_test_data):
        """Test chart generation when matplotlib raises error."""
        mock_plt.subplots.side_effect = ValueError("Plot error")

        result = chart_test_data["generator"].generate_chart(
            chart_test_data["burnup_data"], chart_test_data["chart_data"]
        )

        assert result is False


class TestBurnupChartGeneratorPlotting(TestBurnupChartGeneratorBase):
    """Test cases for chart plotting methods."""

    @patch(
        "jira_agile_metrics.calculators.burnup_chart_generator."
        "find_backlog_and_done_columns"
    )
    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_plot_historical_data(
        self, mock_plt, mock_find_columns, generator, burnup_data
    ):
        """Test plotting historical data."""
        mock_find_columns.return_value = ("Backlog", "Done")
        mock_fig = Mock()
        mock_fig.savefig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        generator.figure, generator.axis = generator.create_chart_figure()
        generator.plot_historical_data(mock_ax, burnup_data)

        assert mock_ax.plot.call_count >= 2  # At least backlog and done lines

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_plot_backlog_fan(self, mock_plt, generator, chart_data):
        """Test plotting backlog fan."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        _, mock_ax_result = generator.create_chart_figure()
        generator.plot_backlog_fan(
            mock_ax_result, chart_data, chart_data["forecast_dates"]
        )

        # Should have called fill_between and plot for median
        fill_between_calls = [
            call for call in mock_ax_result.method_calls if call[0] == "fill_between"
        ]
        assert len(fill_between_calls) > 0

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_plot_done_fan(self, mock_plt, generator, chart_data):
        """Test plotting done fan."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        _, mock_ax_result = generator.create_chart_figure()
        generator.plot_done_fan(
            mock_ax_result, chart_data, chart_data["forecast_dates"]
        )

        # Should have called fill_between and plot for median
        fill_between_calls = [
            call for call in mock_ax_result.method_calls if call[0] == "fill_between"
        ]
        assert len(fill_between_calls) > 0

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_plot_target_line(self, mock_plt, generator):
        """Test plotting target line."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        _, mock_ax_result = generator.create_chart_figure()
        generator.plot_target_line(mock_ax_result, 100)

        mock_ax_result.axhline.assert_called_once()

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_plot_completion_quantiles(self, mock_plt, generator, chart_data):
        """Test plotting completion quantiles."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        _, mock_ax_result = generator.create_chart_figure()
        generator.plot_completion_quantiles(mock_ax_result, chart_data)

        # Should call axvline for each quantile
        assert mock_ax_result.axvline.call_count > 0

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_plot_completion_quantiles_empty(self, mock_plt, generator):
        """Test plotting completion quantiles with empty data."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        _, mock_ax_result = generator.create_chart_figure()
        generator.plot_completion_quantiles(mock_ax_result, {"quantile_data": {}})

        mock_ax_result.axvline.assert_not_called()

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_add_trust_metrics_annotation(self, mock_plt, generator, chart_data):
        """Test adding trust metrics annotation."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        _, mock_ax_result = generator.create_chart_figure()
        generator.add_trust_metrics_annotation(mock_ax_result, chart_data)

        mock_ax_result.text.assert_called_once()

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_add_trust_metrics_annotation_empty(self, mock_plt, generator):
        """Test adding trust metrics annotation with empty data."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        _, mock_ax_result = generator.create_chart_figure()
        generator.add_trust_metrics_annotation(mock_ax_result, {"trust_metrics": {}})

        mock_ax_result.text.assert_not_called()

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.os.makedirs")
    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_save_chart_creates_directory(self, _mock_plt, mock_makedirs, tmp_path):
        """Test that chart saving creates output directory."""
        output_dir = tmp_path / "subdir"
        output_file = output_dir / "burnup.png"
        generator = BurnupChartGenerator(str(output_file))
        mock_fig = Mock()

        generator.save_chart(mock_fig, str(output_file))

        mock_makedirs.assert_called_once_with(str(output_dir), exist_ok=True)

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_save_chart(self, mock_plt, generator):
        """Test saving chart."""
        mock_fig = Mock()

        generator.save_chart(mock_fig, generator.output_file)

        mock_fig.savefig.assert_called_once_with(
            generator.output_file, bbox_inches="tight", dpi=300
        )
        mock_plt.close.assert_called_once_with(mock_fig)
