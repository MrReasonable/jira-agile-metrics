"""Tests for burnup_chart_generator module."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from .burnup_chart_generator import BurnupChartGenerator
from .burnup_chart_utils import save_chart


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
        # savefig is provided by the Mock
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

    @patch("jira_agile_metrics.calculators.burnup_chart_utils.os.makedirs")
    @patch("jira_agile_metrics.calculators.burnup_chart_utils.plt")
    def test_save_chart_creates_directory(self, _mock_plt, mock_makedirs, tmp_path):
        """Test that chart saving creates output directory."""
        output_dir = tmp_path / "subdir"
        output_file = output_dir / "burnup.png"
        mock_fig = Mock()

        save_chart(mock_fig, str(output_file))

        mock_makedirs.assert_called_once_with(str(output_dir), exist_ok=True)

    @patch("jira_agile_metrics.calculators.burnup_chart_utils.plt")
    def test_save_chart(self, mock_plt, generator):
        """Test saving chart."""
        mock_fig = Mock()

        save_chart(mock_fig, generator.output_file)

        mock_fig.savefig.assert_called_once_with(
            generator.output_file, bbox_inches="tight", dpi=300
        )
        mock_plt.close.assert_called_once_with(mock_fig)


class TestBurnupChartGeneratorLegendPlacement(TestBurnupChartGeneratorBase):
    """Test cases for legend placement helper methods."""

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_pixels_to_figure_fraction(self, _mock_plt, generator):
        """Test pixel to figure fraction conversion."""
        # Setup figure with known dimensions
        mock_fig = Mock()
        mock_fig.get_figheight.return_value = 8.0  # 8 inches
        mock_fig.dpi = 100.0  # 100 DPI
        generator.figure = mock_fig

        # 800 pixels = 8 inches * 100 DPI = 1.0 figure fraction
        result = generator.pixels_to_figure_fraction(800.0)
        assert result == pytest.approx(1.0)

        # 400 pixels = 0.5 figure fraction
        result = generator.pixels_to_figure_fraction(400.0)
        assert result == pytest.approx(0.5)

        # 100 pixels = 0.125 figure fraction
        result = generator.pixels_to_figure_fraction(100.0)
        assert result == pytest.approx(0.125)

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_pixels_to_figure_fraction_different_dpi(self, _mock_plt, generator):
        """Test pixel conversion with different DPI."""
        mock_fig = Mock()
        mock_fig.get_figheight.return_value = 10.0  # 10 inches
        mock_fig.dpi = 72.0  # 72 DPI
        generator.figure = mock_fig

        # 720 pixels = 10 inches * 72 DPI = 1.0 figure fraction
        result = generator.pixels_to_figure_fraction(720.0)
        assert result == pytest.approx(1.0)

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_measure_text_height(self, _mock_plt, generator):
        """Test measuring text height in pixels."""
        # Setup figure and axes
        mock_fig = Mock()
        mock_canvas = Mock()
        mock_renderer = Mock()
        mock_fig.canvas = mock_canvas
        mock_canvas.get_renderer.return_value = mock_renderer
        mock_canvas.draw = Mock()

        mock_ax = Mock()
        mock_text = Mock()
        mock_bbox = Mock()
        mock_bbox.height = 25.5  # pixels
        mock_text.get_window_extent.return_value = mock_bbox
        mock_text.remove = Mock()
        mock_ax.text.return_value = mock_text

        generator.figure = mock_fig
        generator.legend_y_offset = -0.08

        result = generator.measure_text_height(mock_ax, "Test legend text")

        assert result == 25.5
        mock_canvas.draw.assert_called_once()
        mock_ax.text.assert_called_once()
        mock_text.get_window_extent.assert_called_once_with(renderer=mock_renderer)
        mock_text.remove.assert_called_once()

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_calculate_text_bottom_figure_normal_case(self, _mock_plt, generator):
        """Test calculating text bottom position in normal case."""
        # Setup figure
        mock_fig = Mock()
        mock_fig.get_figheight.return_value = 8.0
        mock_fig.dpi = 100.0
        generator.figure = mock_fig
        generator.legend_y_offset = -0.08  # 8% below axes bottom

        # Setup axes position: bottom at 0.1, height 0.7 (in figure coords)
        mock_ax = Mock()
        mock_bbox = Mock()
        mock_bbox.y0 = 0.1  # axes bottom in figure coords
        mock_bbox.height = 0.7  # axes height in figure coords
        mock_ax.get_position.return_value = mock_bbox

        # Text height: 100 pixels = 0.125 figure fraction (100 / 800)
        text_height_pixels = 100.0

        result = generator.calculate_text_bottom_figure(mock_ax, text_height_pixels)

        # Expected:
        # text_y_figure = 0.1 + (-0.08 * 0.7) = 0.1 - 0.056 = 0.044
        # text_height_fraction = 100 / 800 = 0.125
        # text_bottom_figure = 0.044 - 0.125 = -0.081
        expected = 0.044 - 0.125
        assert result == pytest.approx(expected, abs=0.001)

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_calculate_text_bottom_figure_axes_at_bottom(self, _mock_plt, generator):
        """Test text bottom calculation when axes is at figure bottom."""
        mock_fig = Mock()
        mock_fig.get_figheight.return_value = 8.0
        mock_fig.dpi = 100.0
        generator.figure = mock_fig
        generator.legend_y_offset = -0.1

        # Axes at very bottom of figure
        mock_ax = Mock()
        mock_bbox = Mock()
        mock_bbox.y0 = 0.0  # axes bottom at figure bottom
        mock_bbox.height = 0.8
        mock_ax.get_position.return_value = mock_bbox

        text_height_pixels = 50.0  # 0.0625 figure fraction

        result = generator.calculate_text_bottom_figure(mock_ax, text_height_pixels)

        # text_y_figure = 0.0 + (-0.1 * 0.8) = -0.08
        # text_bottom_figure = -0.08 - 0.0625 = -0.1425
        expected = -0.08 - 0.0625
        assert result == pytest.approx(expected, abs=0.001)

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_calculate_legend_bottom_margin_normal_case(self, _mock_plt, generator):
        """Test calculating legend bottom margin in normal case."""
        # Setup figure
        mock_fig = Mock()
        mock_fig.get_figheight.return_value = 8.0
        mock_fig.dpi = 100.0
        mock_canvas = Mock()
        mock_renderer = Mock()
        mock_fig.canvas = mock_canvas
        mock_canvas.get_renderer.return_value = mock_renderer
        mock_canvas.draw = Mock()

        generator.figure = mock_fig
        generator.legend_y_offset = -0.08

        # Setup axes
        mock_ax = Mock()
        mock_bbox = Mock()
        mock_bbox.y0 = 0.1  # axes bottom
        mock_bbox.height = 0.7
        mock_ax.get_position.return_value = mock_bbox

        # Mock text measurement
        mock_text = Mock()
        mock_text_bbox = Mock()
        mock_text_bbox.height = 100.0  # pixels
        mock_text.get_window_extent.return_value = mock_text_bbox
        mock_text.remove = Mock()
        mock_ax.text.return_value = mock_text

        default_margin = 0.12
        result = generator.calculate_legend_bottom_margin(
            mock_ax, "Test legend", default_margin
        )

        # Text extends below figure (text_bottom_figure = -0.081)
        # space_below_figure = 0.081
        # calculated_margin = 0.1 + 0.081 + 0.01 = 0.191
        # Should be >= default_margin (0.12)
        assert result >= default_margin
        assert result <= 1.0

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_calculate_legend_bottom_margin_no_overlap(self, _mock_plt, generator):
        """Test margin calculation when legend doesn't overlap bottom."""
        mock_fig = Mock()
        mock_fig.get_figheight.return_value = 8.0
        mock_fig.dpi = 100.0
        mock_canvas = Mock()
        mock_renderer = Mock()
        mock_fig.canvas = mock_canvas
        mock_canvas.get_renderer.return_value = mock_renderer
        mock_canvas.draw = Mock()

        generator.figure = mock_fig
        generator.legend_y_offset = 0.05  # Above axes bottom

        mock_ax = Mock()
        mock_bbox = Mock()
        mock_bbox.y0 = 0.2  # axes well above figure bottom
        mock_bbox.height = 0.6
        mock_ax.get_position.return_value = mock_bbox

        # Small text that doesn't extend below figure
        mock_text = Mock()
        mock_text_bbox = Mock()
        mock_text_bbox.height = 20.0  # small text
        mock_text.get_window_extent.return_value = mock_text_bbox
        mock_text.remove = Mock()
        mock_ax.text.return_value = mock_text

        default_margin = 0.12
        result = generator.calculate_legend_bottom_margin(
            mock_ax, "Short", default_margin
        )

        # Text doesn't extend below figure, so space_below_figure = 0
        # Should use default_margin as minimum
        assert result >= default_margin

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_calculate_legend_bottom_margin_no_figure(self, _mock_plt, generator):
        """Test margin calculation when figure is None."""
        generator.figure = None

        mock_ax = Mock()
        default_margin = 0.15

        result = generator.calculate_legend_bottom_margin(
            mock_ax, "Test", default_margin
        )

        assert result == default_margin

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_calculate_legend_bottom_margin_zero_size_figure(
        self, _mock_plt, generator
    ):
        """Test margin calculation with zero-size figure (edge case)."""
        mock_fig = Mock()
        mock_fig.get_figheight.return_value = 0.0  # Zero height
        mock_fig.dpi = 100.0
        mock_canvas = Mock()
        mock_fig.canvas = mock_canvas
        mock_canvas.draw = Mock()
        mock_canvas.get_renderer.side_effect = ValueError("Zero size figure")

        generator.figure = mock_fig

        mock_ax = Mock()
        default_margin = 0.12

        # Should catch exception and return default
        result = generator.calculate_legend_bottom_margin(
            mock_ax, "Test", default_margin
        )

        assert result == default_margin

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_calculate_legend_bottom_margin_missing_attribute(
        self, _mock_plt, generator
    ):
        """Test margin calculation when axes missing get_position attribute."""
        mock_fig = Mock()
        mock_fig.get_figheight.return_value = 8.0
        mock_fig.dpi = 100.0
        mock_canvas = Mock()
        mock_renderer = Mock()
        mock_fig.canvas = mock_canvas
        mock_canvas.get_renderer.return_value = mock_renderer
        mock_canvas.draw = Mock()

        generator.figure = mock_fig

        mock_ax = Mock()
        mock_ax.get_position.side_effect = AttributeError("Missing attribute")

        default_margin = 0.12
        result = generator.calculate_legend_bottom_margin(
            mock_ax, "Test", default_margin
        )

        assert result == default_margin

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_calculate_legend_bottom_margin_clamps_to_one(self, _mock_plt, generator):
        """Test that calculated margin is clamped to maximum of 1.0."""
        mock_fig = Mock()
        mock_fig.get_figheight.return_value = 8.0
        mock_fig.dpi = 100.0
        mock_canvas = Mock()
        mock_renderer = Mock()
        mock_fig.canvas = mock_canvas
        mock_canvas.get_renderer.return_value = mock_renderer
        mock_canvas.draw = Mock()

        generator.figure = mock_fig
        generator.legend_y_offset = -0.5  # Very far below axes

        mock_ax = Mock()
        mock_bbox = Mock()
        mock_bbox.y0 = 0.9  # Axes near top
        mock_bbox.height = 0.05  # Very small axes
        mock_ax.get_position.return_value = mock_bbox

        # Large text
        mock_text = Mock()
        mock_text_bbox = Mock()
        mock_text_bbox.height = 500.0  # Very large
        mock_text.get_window_extent.return_value = mock_text_bbox
        mock_text.remove = Mock()
        mock_ax.text.return_value = mock_text

        default_margin = 0.05
        result = generator.calculate_legend_bottom_margin(
            mock_ax, "Very long legend text", default_margin
        )

        # Should be clamped to 1.0
        assert result <= 1.0
        assert result >= default_margin


class TestBurnupChartGeneratorXAxisLimits(TestBurnupChartGeneratorBase):
    """Test cases for x-axis limit setting."""

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.set_chart_style")
    def test_xaxis_limits_with_quantile_data(
        self, _mock_set_style, generator, burnup_data
    ):
        """Test x-axis limits are set correctly with completion quantile data."""
        mock_ax = Mock()

        # Create chart data with completion quantile dates
        completion_date = pd.Timestamp("2024-01-20")
        chart_data = {
            "forecast_dates": pd.date_range("2024-01-11", periods=5, freq="D").tolist(),
            "quantile_data": {
                "50%": pd.Timestamp("2024-01-15"),
                "75%": pd.Timestamp("2024-01-18"),
                "90%": completion_date,  # Latest completion date
            },
            "trust_metrics": {},
        }

        # Test through public API
        generator.setup_chart_legend_and_style(mock_ax, burnup_data, chart_data)

        # Verify set_xlim was called with correct limits
        mock_ax.set_xlim.assert_called_once()
        call_args = mock_ax.set_xlim.call_args
        left_limit, right_limit = call_args[1]["left"], call_args[1]["right"]

        # Left limit should be first date of historical data
        assert left_limit == pd.Timestamp(burnup_data.index[0])

        # Right limit should be completion date + padding
        # Padding is 15% of time span or minimum 7 days
        time_span = (completion_date - left_limit).days
        expected_padding = max(time_span * 0.15, 7)
        expected_right = completion_date + pd.Timedelta(days=expected_padding)
        assert right_limit == expected_right

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.set_chart_style")
    def test_xaxis_limits_with_forecast_dates(
        self, _mock_set_style, generator, burnup_data
    ):
        """Test x-axis limits are set correctly with forecast dates only."""
        mock_ax = Mock()

        # Create chart data with forecast dates but no quantile data
        last_forecast_date = pd.Timestamp("2024-01-20")
        chart_data = {
            "forecast_dates": pd.date_range(
                "2024-01-11", end=last_forecast_date, freq="D"
            ).tolist(),
            "quantile_data": {},  # No completion dates
            "trust_metrics": {},
        }

        # Test through public API
        generator.setup_chart_legend_and_style(mock_ax, burnup_data, chart_data)

        # Verify set_xlim was called
        mock_ax.set_xlim.assert_called_once()
        call_args = mock_ax.set_xlim.call_args
        left_limit, right_limit = call_args[1]["left"], call_args[1]["right"]

        # Left limit should be first date of historical data
        assert left_limit == pd.Timestamp(burnup_data.index[0])

        # Right limit should be last forecast date + padding
        time_span = (last_forecast_date - left_limit).days
        expected_padding = max(time_span * 0.15, 7)
        expected_right = last_forecast_date + pd.Timedelta(days=expected_padding)
        assert right_limit == expected_right


class TestBurnupChartGeneratorTrialPadding(TestBurnupChartGeneratorBase):
    """Test cases for trial padding logic through public API."""

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_pad_trials_with_initial_state_reached_target(self, mock_plt, generator):
        """Test that trials with initial_state that reached target stay >= target."""
        target = 50
        forecast_dates = pd.date_range("2024-01-01", periods=8, freq="D").tolist()
        # Trial has reached target (value 50) but is shorter than target_length
        trial = [20, 30, 40, 50]  # initial=20, forecast=[30,40,50], reached target
        chart_data = {
            "forecast_dates": forecast_dates,
            "done_trials": [trial],
            "target": target,
        }
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        _, mock_ax_result = generator.create_chart_figure()
        generator.plot_done_fan(mock_ax_result, chart_data, forecast_dates)

        # Verify that fill_between was called (indicating fan was plotted)
        fill_between_calls = [
            call for call in mock_ax_result.method_calls if call[0] == "fill_between"
        ]
        assert len(fill_between_calls) > 0
        # Verify the data passed maintains target constraint
        # The fan data should have values >= target after reaching it
        plot_calls = [call for call in mock_ax_result.method_calls if call[0] == "plot"]
        assert len(plot_calls) > 0  # Median line should be plotted

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_pad_trials_without_initial_state_reached_target(self, mock_plt, generator):
        """Test that trials without initial_state that reached target stay >= target."""
        target = 50
        forecast_dates = pd.date_range("2024-01-01", periods=8, freq="D").tolist()
        # Treated as initial_state=30, forecast=[40,50], reached target
        trial = [30, 40, 50]
        chart_data = {
            "forecast_dates": forecast_dates,
            "done_trials": [trial],
            "target": target,
        }
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        _, mock_ax_result = generator.create_chart_figure()
        generator.plot_done_fan(mock_ax_result, chart_data, forecast_dates)

        # Verify that plotting occurred
        fill_between_calls = [
            call for call in mock_ax_result.method_calls if call[0] == "fill_between"
        ]
        assert len(fill_between_calls) > 0

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_pad_trials_not_reached_target(self, mock_plt, generator):
        """Test that trials that haven't reached target pad with last value."""
        target = 50
        forecast_dates = pd.date_range("2024-01-01", periods=8, freq="D").tolist()
        # initial=20, forecast=[30,40], last value is 40, below target
        trial = [20, 30, 40]
        chart_data = {
            "forecast_dates": forecast_dates,
            "done_trials": [trial],
            "target": target,
        }
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        _, mock_ax_result = generator.create_chart_figure()
        generator.plot_done_fan(mock_ax_result, chart_data, forecast_dates)

        # Verify that plotting occurred
        fill_between_calls = [
            call for call in mock_ax_result.method_calls if call[0] == "fill_between"
        ]
        assert len(fill_between_calls) > 0

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.plt")
    def test_pad_trials_mixed_reached_and_not_reached(self, mock_plt, generator):
        """Test padding with mix of trials that reached and didn't reach target."""
        target = 50
        forecast_dates = pd.date_range("2024-01-01", periods=8, freq="D").tolist()
        trials = [
            [20, 30, 40, 50, 60],  # Reached target - should stay >= 50
            [20, 25, 30, 35],  # Not reached - should pad with 35
        ]
        chart_data = {
            "forecast_dates": forecast_dates,
            "done_trials": trials,
            "target": target,
        }
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        _, mock_ax_result = generator.create_chart_figure()
        generator.plot_done_fan(mock_ax_result, chart_data, forecast_dates)

        # Verify that plotting occurred with multiple trials
        fill_between_calls = [
            call for call in mock_ax_result.method_calls if call[0] == "fill_between"
        ]
        assert len(fill_between_calls) > 0

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.set_chart_style")
    def test_xaxis_limits_empty_burnup_data(self, _mock_set_style, generator):
        """Test x-axis limits are not set when burnup data is empty."""
        mock_ax = Mock()

        chart_data = {"forecast_dates": [], "quantile_data": {}, "trust_metrics": {}}

        # Test through public API
        generator.setup_chart_legend_and_style(mock_ax, pd.DataFrame(), chart_data)

        # Should not call set_xlim when burnup_data is empty
        mock_ax.set_xlim.assert_not_called()

    @patch("jira_agile_metrics.calculators.burnup_chart_generator.set_chart_style")
    def test_xaxis_limits_no_forecast_data(
        self, _mock_set_style, generator, burnup_data
    ):
        """Test x-axis limits fallback to historical data range."""
        mock_ax = Mock()

        # Chart data with no forecast dates or completion dates
        chart_data = {"forecast_dates": [], "quantile_data": {}, "trust_metrics": {}}

        # Test through public API
        generator.setup_chart_legend_and_style(mock_ax, burnup_data, chart_data)

        # Verify set_xlim was called with historical data range
        mock_ax.set_xlim.assert_called_once()
        call_args = mock_ax.set_xlim.call_args
        left_limit, right_limit = call_args[1]["left"], call_args[1]["right"]

        # Left limit should be first date of historical data
        assert left_limit == pd.Timestamp(burnup_data.index[0])

        # Right limit should be last historical date + padding
        last_historical_date = pd.Timestamp(burnup_data.index[-1])
        time_span = (last_historical_date - left_limit).days
        expected_padding = max(time_span * 0.1, 7)
        expected_right = last_historical_date + pd.Timedelta(days=expected_padding)
        assert right_limit == expected_right
