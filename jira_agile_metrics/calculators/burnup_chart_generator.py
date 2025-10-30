"""Chart generation service for burnup forecast visualization."""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..chart_styling_utils import set_chart_style
from ..config.exceptions import ChartGenerationError
from ..utils import find_backlog_and_done_columns

logger = logging.getLogger(__name__)


class BurnupChartGenerator:
    """Handles chart generation for burnup forecast visualization."""

    def __init__(
        self, output_file: str, figure_size: Optional[Tuple[float, float]] = None
    ):
        self.output_file = output_file
        self.figure = None
        self.axis = None
        # Validate and set figure_size with default (12, 8)
        self.figure_size = self._validate_figure_size(figure_size)

    def _validate_figure_size(
        self,
        figure_size: Optional[Tuple[float, float]],
        default: Tuple[float, float] = (12, 8),
    ) -> Tuple[float, float]:
        """Validate and normalize a matplotlib figure size.

        Returns a normalized (width, height) tuple of floats if valid, otherwise
        returns the provided default while logging an error.
        """
        if figure_size is None:
            return default

        try:
            if len(figure_size) != 2:
                raise ValueError("figure_size must be a tuple of length 2")
            if not all(isinstance(x, (int, float)) for x in figure_size):
                raise ValueError("figure_size must contain numeric values")
            if not all(x > 0 for x in figure_size):
                raise ValueError("figure_size values must be positive")
            return tuple(float(x) for x in figure_size)  # type: ignore[return-value]
        except (ValueError, TypeError) as e:
            logger.error("Invalid figure_size: %s. Using default %s", e, default)
            return default

    def generate_chart(
        self, burnup_data: pd.DataFrame, chart_data: Dict[str, Any]
    ) -> bool:
        """Generate the complete burnup forecast chart."""
        try:
            # Override figure_size from chart_data if provided
            if "figure_size" in chart_data:
                self.figure_size = self._validate_figure_size(
                    chart_data["figure_size"], default=self.figure_size
                )

            # Prepare chart data
            prepared_data = self._prepare_chart_data(burnup_data, chart_data)
            if prepared_data is None:
                return False

            # Create figure and axis
            self.figure, self.axis = self.create_chart_figure()
            if self.figure is None or self.axis is None:
                return False

            # Plot historical data
            self.plot_historical_data(self.axis, burnup_data)

            # Plot forecast fans
            self.plot_forecast_fans(self.axis, burnup_data, prepared_data)

            # Plot target and completion dates
            self.plot_target_and_completion_dates(self.axis, prepared_data)

            # Setup legend and styling
            self.setup_chart_legend_and_style(self.axis, burnup_data, prepared_data)

            # Save chart
            self.save_chart(self.figure, self.output_file)

            return True

        except ChartGenerationError:
            # Re-raise chart generation errors (already logged and wrapped)
            return False
        except (ValueError, TypeError) as e:
            # Catch ValueError/TypeError from matplotlib/pandas/numpy operations
            logger.error("Error generating chart: %s", e)
            raise ChartGenerationError(f"Chart generation failed: {e}") from e
        except OSError as e:
            # Catch file I/O errors
            logger.error("Error saving chart file: %s", e)
            raise ChartGenerationError(f"Failed to save chart: {e}") from e

    def validate_chart_data(self, chart_data: Dict[str, Any]) -> bool:
        """Validate that chart data contains required fields."""
        required_fields = ["forecast_dates", "backlog_trials", "done_trials"]
        return all(field in chart_data for field in required_fields)

    def get_chart_info(self) -> Dict[str, Any]:
        """Get information about the chart generator."""
        return {
            "output_file": self.output_file,
            "has_figure": self.figure is not None,
            "has_axis": self.axis is not None,
        }

    def _prepare_chart_data(
        self, _burnup_data: pd.DataFrame, chart_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Prepare chart data from stored results."""
        try:
            if not chart_data:
                return None

            # Extract relevant data for charting
            prepared_data = {
                "forecast_dates": chart_data.get("forecast_dates", []),
                "backlog_trials": chart_data.get("backlog_trials", []),
                "done_trials": chart_data.get("done_trials", []),
                "trust_metrics": chart_data.get("trust_metrics", {}),
                "target": chart_data.get("target", 0),
                "quantile_data": chart_data.get("quantile_data", {}),
            }

            return prepared_data

        except (ValueError, TypeError) as e:
            # Catch ValueError/TypeError from data processing operations
            logger.error("Error preparing chart data: %s", e)
            raise ChartGenerationError(f"Failed to prepare chart data: {e}") from e

    def create_chart_figure(self) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
        """Create the matplotlib figure and axis."""
        try:
            fig, ax = plt.subplots(figsize=self.figure_size)
            return fig, ax

        except (ValueError, TypeError) as e:
            # Catch ValueError/TypeError from matplotlib operations
            logger.error("Error creating chart figure: %s", e)
            raise ChartGenerationError(f"Failed to create chart figure: {e}") from e

    def plot_historical_data(self, ax: plt.Axes, burnup_data: pd.DataFrame) -> None:
        """Plot historical backlog and done data."""
        try:
            # Find backlog and done columns
            backlog_column, done_column = find_backlog_and_done_columns(burnup_data)

            if backlog_column:
                ax.plot(
                    burnup_data.index,
                    burnup_data[backlog_column],
                    "b-",
                    label="Backlog",
                    linewidth=2,
                )

            if done_column:
                ax.plot(
                    burnup_data.index,
                    burnup_data[done_column],
                    "g-",
                    label="Done",
                    linewidth=2,
                )

        except (ValueError, TypeError) as e:
            # Catch ValueError/TypeError from matplotlib/pandas operations
            logger.error("Error plotting historical data: %s", e)
            raise ChartGenerationError(f"Failed to plot historical data: {e}") from e

    def plot_forecast_fans(
        self, ax: plt.Axes, _burnup_data: pd.DataFrame, chart_data: Dict[str, Any]
    ) -> None:
        """Plot forecast fans for backlog and done data."""
        try:
            forecast_dates = chart_data.get("forecast_dates", [])
            backlog_trials = chart_data.get("backlog_trials", [])
            done_trials = chart_data.get("done_trials", [])

            if not forecast_dates or not backlog_trials or not done_trials:
                return

            # Plot backlog fan
            self.plot_backlog_fan(ax, chart_data, forecast_dates)

            # Plot done fan
            self.plot_done_fan(ax, chart_data, forecast_dates)

        except (ValueError, TypeError) as e:
            # Catch ValueError/TypeError from matplotlib/numpy operations
            logger.error("Error plotting forecast fans: %s", e)
            raise ChartGenerationError(f"Failed to plot forecast fans: {e}") from e

    def _plot_fan(
        self,
        ax: plt.Axes,
        forecast_dates: list,
        trials: list,
        style: Dict[str, Any],
    ) -> None:
        """Plot a fan chart with percentile bands and median line.

        Args:
            ax: Matplotlib axes to plot on
            forecast_dates: List of dates for x-axis
            trials: List of trial data arrays
            style: Dictionary with keys 'colors', 'line_style', 'line_color'
        """
        try:
            if not trials:
                return

            colors = style.get("colors", ["lightblue", "blue"])
            line_style = style.get("line_style", "--")
            line_color = style.get("line_color", "b")

            # Convert trials to numpy array for easier manipulation
            trials_array = np.array(trials)

            # Calculate percentiles for fan
            percentiles = [10, 25, 50, 75, 90]
            fan_data = np.percentile(trials_array, percentiles, axis=0)

            # Plot fan with transparency
            for i, (p_low, p_high) in enumerate([(10, 90), (25, 75)]):
                ax.fill_between(
                    forecast_dates,
                    fan_data[percentiles.index(p_low)],
                    fan_data[percentiles.index(p_high)],
                    alpha=0.3,
                    color=colors[i],
                )

            # Plot median line
            ax.plot(
                forecast_dates,
                fan_data[percentiles.index(50)],
                f"{line_color}{line_style}",
                alpha=0.7,
                linewidth=1,
            )

        except (ValueError, TypeError) as e:
            # Catch ValueError/TypeError from matplotlib/numpy operations
            logger.error("Error plotting fan: %s", e)
            raise ChartGenerationError(f"Failed to plot fan: {e}") from e

    def plot_backlog_fan(
        self, ax: plt.Axes, chart_data: Dict[str, Any], forecast_dates: list
    ) -> None:
        """Plot backlog growth fan."""
        backlog_trials = chart_data.get("backlog_trials", [])
        style = {"colors": ["lightblue", "blue"], "line_style": "--", "line_color": "b"}
        self._plot_fan(ax, forecast_dates, backlog_trials, style)

    def plot_done_fan(
        self, ax: plt.Axes, chart_data: Dict[str, Any], forecast_dates: list
    ) -> None:
        """Plot done fan."""
        done_trials = chart_data.get("done_trials", [])
        style = {
            "colors": ["lightgreen", "green"],
            "line_style": "--",
            "line_color": "g",
        }
        self._plot_fan(ax, forecast_dates, done_trials, style)

    def plot_target_and_completion_dates(
        self, ax: plt.Axes, chart_data: Dict[str, Any]
    ) -> None:
        """Plot target line and quantile completion dates."""
        try:
            target = chart_data.get("target", 0)
            quantile_data = chart_data.get("quantile_data", {})

            # Plot target line
            if target > 0:
                self.plot_target_line(ax, target)

            # Plot completion quantiles
            if quantile_data:
                self.plot_completion_quantiles(ax, chart_data)

        except (ValueError, TypeError) as e:
            # Catch ValueError/TypeError from matplotlib operations
            logger.error("Error plotting target and completion dates: %s", e)
            raise ChartGenerationError(
                f"Failed to plot target and completion dates: {e}"
            ) from e

    def plot_target_line(self, ax: plt.Axes, target: float) -> None:
        """Plot the target line."""
        try:
            ax.axhline(
                y=target,
                color="red",
                linestyle=":",
                linewidth=2,
                label=f"Target ({target})",
            )

        except (ValueError, TypeError) as e:
            # Catch ValueError/TypeError from matplotlib operations
            logger.error("Error plotting target line: %s", e)
            raise ChartGenerationError(f"Failed to plot target line: {e}") from e

    def plot_completion_quantiles(
        self, ax: plt.Axes, chart_data: Dict[str, Any]
    ) -> None:
        """Plot completion date quantiles."""
        try:
            quantile_data = chart_data.get("quantile_data", {})
            if not quantile_data:
                return

            # Plot vertical lines for completion dates
            colors = ["orange", "red", "darkred"]

            for i, (quantile, date) in enumerate(quantile_data.items()):
                if date:
                    ax.axvline(
                        x=date,
                        color=colors[i],
                        linestyle="--",
                        alpha=0.7,
                        label=f"{quantile} completion",
                    )

        except (ValueError, TypeError) as e:
            # Catch ValueError/TypeError from matplotlib operations
            logger.error("Error plotting completion quantiles: %s", e)
            raise ChartGenerationError(
                f"Failed to plot completion quantiles: {e}"
            ) from e

    def setup_chart_legend_and_style(
        self, ax: plt.Axes, _burnup_data: pd.DataFrame, chart_data: Dict[str, Any]
    ) -> None:
        """Setup chart legend and styling."""
        try:
            # Set chart style
            set_chart_style()

            # Setup legend
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            ax.set_xlabel("Date", labelpad=20)
            ax.set_ylabel("Number of items", labelpad=10)
            title = chart_data.get("title", "Burnup Forecast Chart")
            ax.set_title(title)

            # Add trust metrics annotation
            self.add_trust_metrics_annotation(ax, chart_data)

        except (ValueError, TypeError) as e:
            # Catch ValueError/TypeError from matplotlib operations
            logger.error("Error setting up chart legend and style: %s", e)
            raise ChartGenerationError(
                f"Failed to setup chart legend and style: {e}"
            ) from e

    def add_trust_metrics_annotation(
        self, ax: plt.Axes, chart_data: Dict[str, Any]
    ) -> None:
        """Add trustworthiness metrics annotation to the chart."""
        try:
            trust_metrics = chart_data.get("trust_metrics", {})
            if not trust_metrics:
                return

            # Create annotation text
            annotation_text = "Trust Metrics:\n"
            for metric, value in trust_metrics.items():
                if isinstance(value, (int, float)):
                    annotation_text += f"{metric}: {value:.2f}\n"

            # Add annotation to chart
            ax.text(
                0.02,
                0.98,
                annotation_text,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
            )

        except (ValueError, TypeError) as e:
            # Catch ValueError/TypeError from matplotlib operations
            logger.error("Error adding trust metrics annotation: %s", e)
            raise ChartGenerationError(
                f"Failed to add trust metrics annotation: {e}"
            ) from e

    def save_chart(self, fig: plt.Figure, output_file: str) -> None:
        """Save the chart to file."""
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # Save figure
            fig.savefig(output_file, bbox_inches="tight", dpi=300)
            plt.close(fig)

        except OSError as e:
            # Catch file I/O errors from matplotlib savefig
            logger.error("Error saving chart file: %s", e)
            raise ChartGenerationError(f"Failed to save chart file: {e}") from e
        except (ValueError, TypeError) as e:
            # Catch ValueError/TypeError from matplotlib operations
            logger.error("Error saving chart: %s", e)
            raise ChartGenerationError(f"Failed to save chart: {e}") from e
