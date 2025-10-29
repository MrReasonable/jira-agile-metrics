"""Chart generation service for burnup forecast visualization."""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..chart_styling_utils import set_chart_style
from ..utils import find_backlog_and_done_columns

logger = logging.getLogger(__name__)


class BurnupChartGenerator:
    """Handles chart generation for burnup forecast visualization."""

    def __init__(self, output_file: str):
        self.output_file = output_file
        self.figure = None
        self.axis = None

    def generate_chart(
        self, burnup_data: pd.DataFrame, chart_data: Dict[str, Any]
    ) -> bool:
        """Generate the complete burnup forecast chart."""
        try:
            # Prepare chart data
            prepared_data = self._prepare_chart_data(burnup_data, chart_data)
            if prepared_data is None:
                return False

            # Create figure and axis
            self.figure, self.axis = self._create_chart_figure()
            if self.figure is None or self.axis is None:
                return False

            # Plot historical data
            self._plot_historical_data(self.axis, burnup_data)

            # Plot forecast fans
            self._plot_forecast_fans(self.axis, burnup_data, prepared_data)

            # Plot target and completion dates
            self._plot_target_and_completion_dates(self.axis, prepared_data)

            # Setup legend and styling
            self._setup_chart_legend_and_style(self.axis, burnup_data, prepared_data)

            # Save chart
            self._save_chart(self.figure, self.output_file)

            return True

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error generating chart: %s", e)
            return False

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

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error preparing chart data: %s", e)
            return None

    def _create_chart_figure(self) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
        """Create the matplotlib figure and axis."""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            return fig, ax

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error creating chart figure: %s", e)
            return None, None

    def _plot_historical_data(self, ax: plt.Axes, burnup_data: pd.DataFrame) -> None:
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

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error plotting historical data: %s", e)

    def _plot_forecast_fans(
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
            self._plot_backlog_fan(ax, chart_data, forecast_dates)

            # Plot done fan
            self._plot_done_fan(ax, chart_data, forecast_dates)

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error plotting forecast fans: %s", e)

    def _plot_backlog_fan(
        self, ax: plt.Axes, chart_data: Dict[str, Any], forecast_dates: list
    ) -> None:
        """Plot backlog growth fan."""
        try:
            backlog_trials = chart_data.get("backlog_trials", [])
            if not backlog_trials:
                return

            # Convert trials to numpy array for easier manipulation
            backlog_array = np.array(backlog_trials)

            # Calculate percentiles for fan
            percentiles = [10, 25, 50, 75, 90]
            fan_data = np.percentile(backlog_array, percentiles, axis=0)

            # Plot fan with transparency
            colors = ["lightblue", "blue", "darkblue"]
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
                "b--",
                alpha=0.7,
                linewidth=1,
            )

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error plotting backlog fan: %s", e)

    def _plot_done_fan(
        self, ax: plt.Axes, chart_data: Dict[str, Any], forecast_dates: list
    ) -> None:
        """Plot done fan."""
        try:
            done_trials = chart_data.get("done_trials", [])
            if not done_trials:
                return

            # Convert trials to numpy array for easier manipulation
            done_array = np.array(done_trials)

            # Calculate percentiles for fan
            percentiles = [10, 25, 50, 75, 90]
            fan_data = np.percentile(done_array, percentiles, axis=0)

            # Plot fan with transparency
            colors = ["lightgreen", "green", "darkgreen"]
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
                "g--",
                alpha=0.7,
                linewidth=1,
            )

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error plotting done fan: %s", e)

    def _plot_target_and_completion_dates(
        self, ax: plt.Axes, chart_data: Dict[str, Any]
    ) -> None:
        """Plot target line and quantile completion dates."""
        try:
            target = chart_data.get("target", 0)
            quantile_data = chart_data.get("quantile_data", {})

            # Plot target line
            if target > 0:
                self._plot_target_line(ax, target)

            # Plot completion quantiles
            if quantile_data:
                self._plot_completion_quantiles(ax, chart_data)

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error plotting target and completion dates: %s", e)

    def _plot_target_line(self, ax: plt.Axes, target: float) -> None:
        """Plot the target line."""
        try:
            ax.axhline(
                y=target,
                color="red",
                linestyle=":",
                linewidth=2,
                label=f"Target ({target})",
            )

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error plotting target line: %s", e)

    def _plot_completion_quantiles(
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

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error plotting completion quantiles: %s", e)

    def _setup_chart_legend_and_style(
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
            ax.set_title("Burnup Forecast Chart")

            # Add trust metrics annotation
            self._add_trust_metrics_annotation(ax, chart_data)

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error setting up chart legend and style: %s", e)

    def _add_trust_metrics_annotation(
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

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error adding trust metrics annotation: %s", e)

    def _save_chart(self, fig: plt.Figure, output_file: str) -> None:
        """Save the chart to file."""
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save figure
            fig.savefig(output_file, bbox_inches="tight", dpi=300)
            plt.close(fig)

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Error saving chart: %s", e)
