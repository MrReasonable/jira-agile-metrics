"""Chart plotting functions for progress reports.

This module contains functions for generating charts including CFD,
throughput, and scatterplot visualizations.
"""

import base64
import io

import matplotlib.pyplot as plt
import pandas as pd

from ..chart_styling_utils import set_chart_style
from .cfd import calculate_cfd_data
from .scatterplot import calculate_scatterplot_data
from .throughput import calculate_throughput


def _add_deadline_to_cfd(ax, deadline, target):
    """Add deadline line to CFD plot.

    Args:
        ax: Matplotlib axes
        deadline: Deadline date
        target: Target date
    """
    if deadline:
        ax.axvline(
            x=deadline,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Deadline",
        )
    if target:
        ax.axvline(
            x=target,
            color="orange",
            linestyle=":",
            alpha=0.7,
            label="Target",
        )


def _add_target_value_to_cfd(ax, target):
    """Add target value (horizontal) to CFD plot.

    Args:
        ax: Matplotlib axes
        target: Target value
    """
    if target is not None:
        ax.axhline(
            y=target,
            color="orange",
            linestyle=":",
            alpha=0.7,
            label="Target",
        )


def plot_cfd(cycle_data, plot_config, target=None):
    """Plot cumulative flow diagram.

    Args:
        cycle_data: Cycle data
        plot_config: Plot configuration
        target: Optional target value (number of items) to display as a
            horizontal line on the chart. Only drawn if provided and truthy.

    Returns:
        Base64 encoded plot image
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare CFD data
    cfd_data = calculate_cfd_data(
        cycle_data,
        plot_config["cycle_names"],
    )

    # Plot CFD
    for column in cfd_data.columns:
        ax.plot(cfd_data.index, cfd_data[column], label=column, linewidth=2)

    # Add horizontal target line (value threshold)
    _add_target_value_to_cfd(ax, target)

    # Add deadline/target date lines (vertical lines)
    _add_deadline_to_cfd(
        ax,
        plot_config.get("deadline"),
        plot_config.get("target"),
    )

    # Format plot
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of items")
    ax.legend()
    ax.grid(True, alpha=0.3)
    set_chart_style()

    # Convert to base64
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return image_base64


def plot_throughput(cycle_data, frequency="1W"):
    """Plot throughput chart.

    Args:
        cycle_data: Cycle data
        frequency: Frequency for resampling

    Returns:
        Base64 encoded plot image
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate throughput
    throughput_data = calculate_throughput(cycle_data, frequency)

    # Plot throughput
    ax.bar(throughput_data.index, throughput_data.values, alpha=0.7)

    # Format plot
    ax.set_xlabel("Date")
    ax.set_ylabel("Throughput")
    ax.set_title("Throughput Over Time")
    ax.grid(True, alpha=0.3)
    set_chart_style()

    # Convert to base64
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return image_base64


def plot_scatterplot(cycle_data, quantiles):
    """Plot scatterplot chart.

    Args:
        cycle_data: Cycle data
        quantiles: List of quantiles

    Returns:
        Base64 encoded plot image
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate scatterplot data
    scatterplot_data = calculate_scatterplot_data(cycle_data)

    # Prepare lead_time column for plotting and quantile calculation
    # Use lead_time if available, otherwise fall back to cycle_time
    if "lead_time" in scatterplot_data.columns:
        lead_time_col = scatterplot_data["lead_time"]
    else:
        lead_time_col = scatterplot_data["cycle_time"]

    # Convert lead_time to numeric (days) for both plotting and quantile calculation
    if pd.api.types.is_timedelta64_dtype(lead_time_col):
        lead_time_numeric = lead_time_col.dt.days
    else:
        lead_time_numeric = pd.to_numeric(lead_time_col, errors="coerce")

    # Drop NaN values for quantile calculation
    lead_time_clean = lead_time_numeric.dropna()

    # Plot scatterplot
    ax.scatter(
        scatterplot_data["cycle_time"],
        lead_time_numeric,
        alpha=0.6,
        s=20,
    )

    # Add quantile lines (only if we have valid data)
    if len(lead_time_clean) > 0:
        for quantile in quantiles:
            quantile_value = lead_time_clean.quantile(quantile)
            if pd.notna(quantile_value):
                ax.axhline(
                    y=quantile_value,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label=f"{quantile * 100}% quantile",
                )

    # Format plot
    ax.set_xlabel("Cycle Time (days)")
    ax.set_ylabel("Lead Time (days)")
    ax.set_title("Cycle Time vs Lead Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    set_chart_style()

    # Convert to base64
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return image_base64
