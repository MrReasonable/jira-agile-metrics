"""Chart generation service for burnup forecast visualization."""

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..chart_styling_utils import set_chart_style
from ..config.exceptions import ChartGenerationError
from ..utils import find_backlog_and_done_columns
from .burnup_chart_types import FanPlotParams
from .burnup_chart_utils import (
    extend_forecast_dates_to_completion,
    extract_forecast_trials,
    find_latest_completion_date,
    format_date_for_legend,
    save_chart,
    validate_figure_size,
    validate_initial_state,
)

logger = logging.getLogger(__name__)


class BurnupChartGenerator:
    """Handles chart generation for burnup forecast visualization."""

    def __init__(
        self,
        output_file: str,
        figure_size: Optional[Tuple[float, float]] = None,
        legend_bottom_margin: Optional[float] = None,
        legend_y_offset: Optional[float] = None,
    ):
        self.output_file = output_file
        self.figure = None
        self.axis = None
        # Validate and set figure_size with default (12, 8)
        self.figure_size = validate_figure_size(figure_size)
        # Legend placement configuration
        # Default bottom margin (as fraction of figure height)
        self.legend_bottom_margin = legend_bottom_margin or 0.12
        # Default y-offset in axes coordinates (negative = below axes)
        self.legend_y_offset = legend_y_offset or -0.08

    def generate_chart(
        self, burnup_data: pd.DataFrame, chart_data: Dict[str, Any]
    ) -> bool:
        """Generate the complete burnup forecast chart."""
        try:
            # Override figure_size from chart_data if provided
            if "figure_size" in chart_data:
                self.figure_size = validate_figure_size(
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
            self.plot_forecast_fans(self.axis, prepared_data)

            # Plot target and completion dates
            self.plot_target_and_completion_dates(self.axis, prepared_data, burnup_data)

            # Setup legend and styling
            self.setup_chart_legend_and_style(self.axis, burnup_data, prepared_data)

            # Save chart
            save_chart(self.figure, self.output_file)

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

            # Get base forecast dates
            forecast_dates = chart_data.get("forecast_dates", [])
            quantile_data = chart_data.get("quantile_data", {})
            target = chart_data.get("target", 0)

            # Debug logging before extension
            if forecast_dates:
                logger.debug(
                    "Before extension: forecast_dates=%d, first_date=%s, last_date=%s",
                    len(forecast_dates),
                    forecast_dates[0] if forecast_dates else "N/A",
                    forecast_dates[-1] if forecast_dates else "N/A",
                )

            # Extend forecast_dates to include latest completion date if needed
            # This ensures the forecast fan shows crossing the target line
            # We extend until ALL trials reach the target, not just percentiles
            done_trials = chart_data.get("done_trials", [])
            if forecast_dates and target > 0:
                original_length = len(forecast_dates)
                forecast_dates = extend_forecast_dates_to_completion(
                    forecast_dates, quantile_data, done_trials, target
                )
                if len(forecast_dates) > original_length:
                    logger.debug(
                        "Extended forecast_dates from %d to %d dates "
                        "(last_date: %s -> %s)",
                        original_length,
                        len(forecast_dates),
                        (
                            forecast_dates[original_length - 1]
                            if original_length > 0
                            else "N/A"
                        ),
                        forecast_dates[-1] if forecast_dates else "N/A",
                    )

            # Extract relevant data for charting
            prepared_data = {
                "forecast_dates": forecast_dates,
                "backlog_trials": chart_data.get("backlog_trials", []),
                "done_trials": chart_data.get("done_trials", []),
                "trust_metrics": chart_data.get("trust_metrics", {}),
                "target": target,
                "quantile_data": quantile_data,
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

    def plot_forecast_fans(self, ax: plt.Axes, chart_data: Dict[str, Any]) -> None:
        """Plot forecast fans for backlog and done data."""
        try:
            forecast_dates = chart_data.get("forecast_dates", [])
            backlog_trials = chart_data.get("backlog_trials", [])
            done_trials = chart_data.get("done_trials", [])

            if not forecast_dates:
                logger.warning(
                    "No forecast dates available. Cannot plot forecast fans."
                )
                return

            # forecast_dates should already include last_date (from _get_forecast_dates)
            # to connect forecast to history. No need to prepend here.

            if not backlog_trials:
                logger.warning(
                    "No backlog trials available. Skipping backlog forecast fan."
                )
            else:
                # Plot backlog fan
                self.plot_backlog_fan(ax, chart_data, forecast_dates)

            if not done_trials:
                logger.warning("No done trials available. Skipping done forecast fan.")
            else:
                # Plot done fan
                self.plot_done_fan(ax, chart_data, forecast_dates)

        except (ValueError, TypeError) as e:
            # Catch ValueError/TypeError from matplotlib/numpy operations
            logger.error("Error plotting forecast fans: %s", e)
            raise ChartGenerationError(f"Failed to plot forecast fans: {e}") from e

    def _plot_fan(
        self, ax: plt.Axes, forecast_dates: list, trials: list, params: Dict[str, Any]
    ) -> None:
        """Plot a fan chart with percentile bands and median line.

        Args:
            ax: Matplotlib axes to plot on
            forecast_dates: List of dates for x-axis
            trials: List of trial data arrays (each array has initial state
                + forecast values)
            params: Dictionary with keys 'style' (dict with 'colors', 'line_style',
                'line_color'), 'target' (int, default 0)
        """
        try:
            if not trials or not forecast_dates:
                return

            expected_length = len(forecast_dates)
            # Pad trials to expected length before extraction
            # Pass target to ensure trials that have reached it stay >= target
            target_value = params.get("target", 0)
            padded_trials = self._pad_trials_to_length(
                trials, expected_length, target_value
            )
            forecast_trials = extract_forecast_trials(padded_trials, expected_length)
            if not forecast_trials:
                return

            # Calculate percentiles for fan
            # Include 99th percentile to show the longest timescale crossing target
            trials_array = np.array(forecast_trials)
            percentiles = [10, 25, 50, 75, 90, 99]
            fan_data = np.percentile(trials_array, percentiles, axis=0)

            # Debug logging to verify we're plotting all data points
            logger.debug(
                "Plotting fan: forecast_dates=%d, fan_data shape=%s, "
                "last_date=%s, last_p99=%.1f, last_p90=%.1f, last_p50=%.1f",
                len(forecast_dates),
                fan_data.shape,
                forecast_dates[-1] if forecast_dates else "N/A",
                fan_data[percentiles.index(99), -1] if fan_data.shape[1] > 0 else 0,
                fan_data[percentiles.index(90), -1] if fan_data.shape[1] > 0 else 0,
                fan_data[percentiles.index(50), -1] if fan_data.shape[1] > 0 else 0,
            )

            # Plot fan with transparency and median line
            style = params.get("style", {})
            plot_params = FanPlotParams(
                forecast_dates=forecast_dates,
                fan_data=fan_data,
                percentiles=percentiles,
                style=style,
            )
            self._plot_fan_bands(ax, plot_params)

        except (ValueError, TypeError) as e:
            # Catch ValueError/TypeError from matplotlib/numpy operations
            logger.error("Error plotting fan: %s", e)
            raise ChartGenerationError(f"Failed to plot fan: {e}") from e

    def _pad_trial_with_initial_state(
        self, trial: list, target_length: int, target_value: int
    ) -> list:
        """Pad a trial that has initial_state to target length."""
        forecast_values = trial[1:]
        forecast_len = len(forecast_values)
        if forecast_len < target_length:
            last_value = forecast_values[-1] if forecast_len > 0 else trial[0]
            pad_value = self._calculate_pad_value(
                forecast_values, last_value, target_value
            )
            padded_forecast = forecast_values + [pad_value] * (
                target_length - forecast_len
            )
            return [trial[0]] + padded_forecast
        if forecast_len == target_length:
            return trial
        # Longer than expected - truncate forecast values
        return [trial[0]] + forecast_values[:target_length]

    def _pad_trial_without_initial_state(
        self, trial: list, target_length: int, target_value: int
    ) -> list:
        """Pad a trial without initial_state to target length."""
        trial_len = len(trial)
        if trial_len < target_length:
            last_value = trial[-1]
            pad_value = self._calculate_pad_value(trial, last_value, target_value)
            return trial + [pad_value] * (target_length - trial_len)
        if trial_len == target_length:
            return trial
        # Longer than expected - truncate
        return trial[:target_length]

    def _calculate_pad_value(
        self, values: list, last_value: float, target_value: int
    ) -> float:
        """Calculate pad value, ensuring trials that reached target stay >= target."""
        if target_value > 0 and any(v >= target_value for v in values):
            return max(last_value, target_value)
        return last_value

    def _pad_trials_to_length(
        self, trials: list, target_length: int, target_value: int = 0
    ) -> list:
        """Pad short trials to target length by repeating the last value.

        Trials that complete early should be padded to the full forecast horizon
        so they can be plotted correctly. Handles both trials with and without
        initial_state. Ensures trials that have reached the target stay at or
        above the target when extended.

        Args:
            trials: List of trial data (each trial is a list of values)
            target_length: Target length for forecast values
                (not including initial_state)
            target_value: Target value to ensure trials that have reached it
                stay at or above it when extended (default: 0, no constraint)

        Returns:
            List of padded trials. Trials with initial_state will have length
            target_length+1, trials without will have length target_length.
        """
        padded = []
        for idx, trial in enumerate(trials):
            if not isinstance(trial, list):
                padded.append(trial)
                continue

            trial_len = len(trial)
            if trial_len == 0:
                logger.warning("Trial %d is empty, padding with zeros", idx)
                padded.append([0] * target_length)
                continue

            has_initial_state = trial_len > 1 and validate_initial_state(trial[0], idx)
            if has_initial_state:
                padded_trial = self._pad_trial_with_initial_state(
                    trial, target_length, target_value
                )
            else:
                padded_trial = self._pad_trial_without_initial_state(
                    trial, target_length, target_value
                )
            padded.append(padded_trial)

        return padded

    def _plot_fan_bands(self, ax: plt.Axes, params: FanPlotParams) -> None:
        """Plot fan bands and median line."""
        colors = params.style.get("colors", ["lightblue", "blue"])
        line_style = params.style.get("line_style", "--")
        line_color = params.style.get("line_color", "b")

        # Plot fan with transparency
        # Include (90, 99) band to show longest timescale crossing target
        band_pairs = [(10, 90), (25, 75), (90, 99)]
        for i, (p_low, p_high) in enumerate(band_pairs):
            if p_low in params.percentiles and p_high in params.percentiles:
                ax.fill_between(
                    params.forecast_dates,
                    params.fan_data[params.percentiles.index(p_low)],
                    params.fan_data[params.percentiles.index(p_high)],
                    alpha=0.3,
                    color=colors[i % len(colors)],
                )

        # Plot median line
        ax.plot(
            params.forecast_dates,
            params.fan_data[params.percentiles.index(50)],
            f"{line_color}{line_style}",
            alpha=0.7,
            linewidth=1,
        )

    def plot_backlog_fan(
        self, ax: plt.Axes, chart_data: Dict[str, Any], forecast_dates: list
    ) -> None:
        """Plot backlog growth fan."""
        backlog_trials = chart_data.get("backlog_trials", [])
        style = {"colors": ["lightblue", "blue"], "line_style": "--", "line_color": "b"}
        # Backlog doesn't have a target, so pass 0
        params = {"style": style, "target": 0}
        self._plot_fan(ax, forecast_dates, backlog_trials, params)

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
        # Pass target to ensure trials that have reached it stay >= target
        target = chart_data.get("target", 0)
        params = {"style": style, "target": target}
        self._plot_fan(ax, forecast_dates, done_trials, params)

    def plot_target_and_completion_dates(
        self, ax: plt.Axes, chart_data: Dict[str, Any], burnup_data: pd.DataFrame
    ) -> None:
        """Plot target line and quantile completion dates."""
        try:
            target = chart_data.get("target", 0)
            quantile_data = chart_data.get("quantile_data", {})

            # Plot target line with improved label
            if target > 0:
                target_label = self._get_target_label(burnup_data, target)
                ax.axhline(
                    y=target,
                    color="red",
                    linestyle=":",
                    linewidth=2,
                    label=target_label,
                )

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

    def _get_target_label(self, burnup_data: pd.DataFrame, target: float) -> str:
        """Generate a descriptive label for the target line.

        Args:
            burnup_data: Historical burnup data
            target: Target value

        Returns:
            Label string showing target with breakdown if available
        """
        try:
            # Try to get current backlog value
            backlog_column, done_column = find_backlog_and_done_columns(burnup_data)
            if backlog_column and done_column and not burnup_data.empty:
                current_backlog = burnup_data[backlog_column].iloc[-1]
                # Show breakdown if target equals backlog (default case)
                if abs(target - current_backlog) < 0.01:
                    return f"Target: {target:.0f} (Backlog: {current_backlog:.0f})"

            # Fallback to simple target
            return f"Target: {target:.0f}"
        except (ValueError, TypeError, KeyError, AttributeError):
            # If anything fails, return simple target
            return f"Target: {target:.0f}"

    def plot_completion_quantiles(
        self, ax: plt.Axes, chart_data: Dict[str, Any]
    ) -> None:
        """Plot completion date quantiles."""
        try:
            quantile_data = chart_data.get("quantile_data", {})
            if not quantile_data:
                logger.debug("No quantile data available for completion dates")
                return

            # Plot vertical lines for completion dates (trial completion levels)
            colors = ["yellow", "orange", "red", "darkred", "maroon"]
            quantile_order = [
                "50%",
                "75%",
                "85%",
                "90%",
                "99%",
            ]  # Include 85% and 99% trials

            plotted_count = 0
            for i, quantile in enumerate(quantile_order):
                if quantile in quantile_data:
                    date = quantile_data[quantile]
                    if date:
                        try:
                            ax.axvline(
                                x=date,
                                color=colors[i],
                                linestyle="--",
                                alpha=0.7,
                                linewidth=1.5,
                                label=f"≤{quantile} trials",
                            )
                            plotted_count += 1
                        except (ValueError, TypeError) as e:
                            logger.warning(
                                "Error plotting ≤%s trials date %s: %s",
                                quantile,
                                date,
                                e,
                            )

            if plotted_count == 0:
                logger.warning(
                    "No valid completion dates found in quantile_data: %s",
                    quantile_data,
                )

        except (ValueError, TypeError) as e:
            # Catch ValueError/TypeError from matplotlib operations
            logger.error("Error plotting completion quantiles: %s", e)
            raise ChartGenerationError(
                f"Failed to plot completion quantiles: {e}"
            ) from e

    def setup_chart_legend_and_style(
        self, ax: plt.Axes, burnup_data: pd.DataFrame, chart_data: Dict[str, Any]
    ) -> None:
        """Setup chart legend and styling."""
        try:
            # Set chart style
            set_chart_style()

            # Set x-axis limits based on completion dates or forecast horizon
            # This ensures the chart focuses on the relevant time period
            self._extend_xaxis_for_completion_dates(ax, burnup_data, chart_data)

            # Setup legend
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            ax.set_xlabel("Date", labelpad=20)
            ax.set_ylabel("Number of items", labelpad=10)
            title = chart_data.get("title", "Burnup Forecast Chart")
            ax.set_title(title)

            # Add trust metrics annotation
            self.add_trust_metrics_annotation(ax, chart_data)

            # Add forecast summary legend beneath the chart
            self.add_forecast_summary_legend(ax, chart_data)

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
                # Skip the use_dates flag
                if metric == "use_dates":
                    continue
                # Handle datetime objects (from date-based percentiles)
                if isinstance(value, datetime):
                    annotation_text += f"{metric}: {value.strftime('%Y-%m-%d')}\n"
                elif isinstance(value, (int, float)):
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

    def _build_legend_lines(
        self, target: int, quantile_data: Dict[str, Any], trust_metrics: Dict[str, Any]
    ) -> list:
        """Build legend lines from target and percentile data."""
        legend_lines = []

        # Add target number
        if target > 0:
            legend_lines.append(f"Target: {target}")

        # Get percentile dates from quantile_data or trust_metrics
        # Prefer quantile_data as it's more complete
        percentiles = {
            "50%": quantile_data.get("50%") or trust_metrics.get("p50"),
            "85%": quantile_data.get("85%"),
            "90%": quantile_data.get("90%") or trust_metrics.get("p90"),
            "99%": quantile_data.get("99%"),
        }

        # Format and add percentile dates with ≤ prefix to indicate trial completion
        for label, value in percentiles.items():
            if value:
                date_str = format_date_for_legend(value)
                legend_lines.append(f"≤{label} trials: {date_str}")

        return legend_lines

    def add_forecast_summary_legend(
        self, ax: plt.Axes, chart_data: Dict[str, Any]
    ) -> None:
        """Add forecast summary legend beneath the chart.

        Shows target number and percentile trial completion dates
        (≤p50, ≤p75, ≤p90, ≤p99).

        The legend placement is automatically calculated based on the rendered text
        height to avoid cutoff on different figure sizes. The margin and y-offset
        can be overridden via chart_data['legend_bottom_margin'] and
        chart_data['legend_y_offset'].
        """
        try:
            target = chart_data.get("target", 0)
            quantile_data = chart_data.get("quantile_data", {})
            trust_metrics = chart_data.get("trust_metrics", {})

            # Build legend text
            legend_lines = self._build_legend_lines(
                target, quantile_data, trust_metrics
            )

            # Only add legend if we have data
            if not legend_lines:
                return

            legend_text = " | ".join(legend_lines)

            # Get margin/y-offset from chart_data or use instance defaults
            bottom_margin = chart_data.get(
                "legend_bottom_margin", self.legend_bottom_margin
            )
            y_offset = chart_data.get("legend_y_offset", self.legend_y_offset)

            # Adjust figure to make room for legend beneath chart
            if self.figure is not None:
                self.figure.subplots_adjust(bottom=bottom_margin)

            # Add text box beneath the chart
            ax.text(
                0.5,
                y_offset,
                legend_text,
                transform=ax.transAxes,
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.8},
            )

        except (ValueError, TypeError) as e:
            # Catch ValueError/TypeError from matplotlib operations
            logger.error("Error adding forecast summary legend: %s", e)
            # Don't raise - this is a nice-to-have feature

    def pixels_to_figure_fraction(self, pixels: float) -> float:
        """Convert pixel measurement to figure coordinate fraction.

        Args:
            pixels: Measurement in pixels

        Returns:
            Equivalent fraction of figure height (0.0 to 1.0)
        """
        # Get figure height in pixels: figure height (inches) * DPI
        figure_height_pixels = self.figure.get_figheight() * self.figure.dpi

        # Convert pixels to fraction of figure height
        figure_fraction = pixels / figure_height_pixels

        return figure_fraction

    def measure_text_height(self, ax: plt.Axes, legend_text: str) -> float:
        """Measure text height in pixels.

        Creates a temporary text object with the same styling as the legend,
        measures its bounding box, then removes it.

        Args:
            ax: The axes object where the legend will be placed
            legend_text: The text that will be displayed in the legend

        Returns:
            Text height in pixels
        """
        # Draw the canvas to ensure accurate text measurements
        # (matplotlib needs a rendered canvas to calculate text extents)
        self.figure.canvas.draw()

        # Create a temporary text object with identical styling to the legend
        # Position: center horizontally (0.5), at legend_y_offset vertically
        # Transform: axes coordinates (0.0-1.0 relative to axes)
        temp_text = ax.text(
            0.5,
            self.legend_y_offset,
            legend_text,
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.8},
        )

        # Get bounding box in display coordinates (pixels)
        # get_window_extent() returns the bounding box after rendering
        renderer = self.figure.canvas.get_renderer()
        bbox_display = temp_text.get_window_extent(renderer=renderer)

        # Extract height from bounding box (in pixels)
        text_height_pixels = bbox_display.height

        # Clean up: remove the temporary text object
        temp_text.remove()

        return text_height_pixels

    def calculate_text_bottom_figure(
        self, ax: plt.Axes, text_height_pixels: float
    ) -> float:
        """Calculate text bottom edge position in figure coordinates.

        Converts the text position from axes coordinates to figure coordinates,
        then subtracts the text height to find the bottom edge.

        Args:
            ax: The axes object where the legend will be placed
            text_height_pixels: Height of the text in pixels

        Returns:
            Bottom edge of text in figure coordinates (fraction of figure height)
        """
        # Get axes position in figure coordinates
        # ax_pos is a Bbox object with x0, y0, width, height in figure coords
        ax_pos = ax.get_position()

        # STEP 1: Convert text top position from axes coords to figure coords
        # text_y_axes: y-position in axes coordinates (0.0-1.0 relative to axes)
        text_y_axes = self.legend_y_offset

        # ax_pos.y0: bottom edge of axes in figure coordinates
        # ax_pos.height: height of axes in figure coordinates
        # text_y_figure: top edge of text in figure coordinates
        text_y_figure = ax_pos.y0 + (text_y_axes * ax_pos.height)

        # STEP 2: Convert text height from pixels to figure coordinates
        # Use helper function to convert pixels to fraction of figure height
        text_height_fraction = self.pixels_to_figure_fraction(text_height_pixels)

        # STEP 3: Calculate bottom edge of text in figure coordinates
        # Bottom edge = top edge - height
        text_bottom_figure = text_y_figure - text_height_fraction

        return text_bottom_figure

    def calculate_legend_bottom_margin(
        self, ax: plt.Axes, legend_text: str, default_margin: float
    ) -> float:
        """Calculate adaptive bottom margin based on rendered text height.

        Measures the actual rendered text height and calculates the minimum
        bottom margin needed to ensure the legend is fully visible. Falls back
        to default_margin if calculation fails or if text doesn't extend below
        the figure.

        Args:
            ax: The axes object where the legend will be placed
            legend_text: The text that will be displayed in the legend
            default_margin: Default bottom margin (fraction of figure height)
                to use as minimum

        Returns:
            Calculated bottom margin (fraction of figure height) that ensures
            the legend text is fully visible
        """
        try:
            # Early return if figure not initialized
            if self.figure is None:
                return default_margin

            # STEP 1: Measure text height in pixels
            text_height_pixels = self.measure_text_height(ax, legend_text)

            # STEP 2: Calculate text bottom edge in figure coordinates
            text_bottom_figure = self.calculate_text_bottom_figure(
                ax, text_height_pixels
            )

            # STEP 3: Calculate required bottom margin
            # Get axes position in figure coordinates
            ax_pos = ax.get_position()
            # ax_pos.y0: bottom edge of axes in figure coordinates
            ax_y0 = ax_pos.y0

            # Calculate space needed below figure (negative = extends below)
            # If text_bottom_figure is negative, text extends below figure bottom
            space_below_figure = max(0, -text_bottom_figure)

            # Add small padding for visual spacing
            padding = 0.01

            # Calculate margin: axes bottom + space below figure + padding
            calculated_margin = ax_y0 + space_below_figure + padding

            # Clamp to valid range [0.0, 1.0]
            calculated_margin = min(calculated_margin, 1.0)

            # Use the larger of calculated margin or default (ensure minimum)
            final_margin = max(calculated_margin, default_margin)

            logger.debug(
                "Calculated legend bottom margin: %.4f (text bottom: %.4f, "
                "axes bottom: %.4f, space below figure: %.4f, default: %.4f)",
                final_margin,
                text_bottom_figure,
                ax_y0,
                space_below_figure,
                default_margin,
            )

            return final_margin

        except (ValueError, TypeError, AttributeError) as e:
            # If calculation fails, fall back to default margin
            # This handles edge cases like zero-size figures, missing attributes, etc.
            logger.warning(
                "Failed to calculate adaptive legend margin: %s. Using default: %.4f",
                e,
                default_margin,
            )
            return default_margin

    def _extend_xaxis_for_completion_dates(
        self, ax: plt.Axes, burnup_data: pd.DataFrame, chart_data: Dict[str, Any]
    ) -> None:
        """Set x-axis limits based on completion dates or forecast horizon.

        Sets both left and right limits based on the actual data range to show
        (historical + forecast), ensuring the chart focuses on the relevant time
        period rather than including all historical data which may span years.

        Args:
            ax: Matplotlib axes
            burnup_data: Historical burnup data
            chart_data: Chart data dictionary with quantile_data and forecast_dates
        """
        try:
            if burnup_data.empty:
                return

            # Left limit: Start from the first date of historical data
            left_limit = pd.Timestamp(burnup_data.index[0])

            # Get forecast dates to determine the right limit
            forecast_dates = chart_data.get("forecast_dates", [])

            # Check quantile completion dates for right limit
            quantile_data = chart_data.get("quantile_data", {})
            latest_completion_date = None
            if quantile_data:
                latest_completion_date = find_latest_completion_date(quantile_data)

            # Right limit: End shortly after the last forecasted completed item
            if latest_completion_date:
                # Use latest completion date as the basis
                # Add padding: 15% of the time span from left_limit to completion date,
                # or minimum 7 days, whichever is larger
                time_span = (latest_completion_date - left_limit).days
                padding_days = max(time_span * 0.15, 7)
                right_limit = latest_completion_date + pd.Timedelta(days=padding_days)
                logger.debug(
                    "Setting x-axis limits: left=%s, right=%s based on latest "
                    "completion date %s (padding: %.1f days)",
                    left_limit,
                    right_limit,
                    latest_completion_date,
                    padding_days,
                )
            elif forecast_dates:
                # Use the last forecast date plus padding
                last_forecast_date = pd.Timestamp(forecast_dates[-1])
                time_span = (last_forecast_date - left_limit).days
                padding_days = max(time_span * 0.15, 7)
                right_limit = last_forecast_date + pd.Timedelta(days=padding_days)
                logger.debug(
                    "Setting x-axis limits: left=%s, right=%s based on forecast "
                    "horizon (last forecast date: %s, padding: %.1f days)",
                    left_limit,
                    right_limit,
                    last_forecast_date,
                    padding_days,
                )
            else:
                # No completion dates or forecast dates - use historical data range
                # with some padding
                right_limit = pd.Timestamp(burnup_data.index[-1])
                time_span = (right_limit - left_limit).days
                padding_days = max(time_span * 0.1, 7)
                right_limit = right_limit + pd.Timedelta(days=padding_days)
                logger.debug(
                    "Setting x-axis limits: left=%s, right=%s based on historical "
                    "data range (padding: %.1f days)",
                    left_limit,
                    right_limit,
                    padding_days,
                )

            # Set both left and right limits to focus on the relevant time period
            ax.set_xlim(left=left_limit, right=right_limit)

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            logger.debug("Error setting x-axis limits for completion dates: %s", e)
            # Don't fail if this doesn't work - it's a nice-to-have
