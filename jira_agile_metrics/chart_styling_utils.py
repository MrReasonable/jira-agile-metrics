"""Common chart styling utilities to eliminate code duplication."""

import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_chart_style(style="whitegrid", despine=True):
    """Set seaborn chart style."""
    sns.set_style(style)
    if despine:
        sns.despine()


def _format_index_labels(index):
    """Extract datetime formatting logic into a helper.

    Converts index values to formatted labels, handling datetime objects
    and falling back to string representation for non-datetime values.

    Args:
        index: Index object (pandas Index, list, etc.) to format

    Returns:
        list: Formatted label strings
    """
    # Check if all items are purely numeric (int/float) without datetime-like attributes
    # This prevents interpreting numeric indices as Unix timestamps
    if all(
        isinstance(item, (int, float))
        and not hasattr(item, "strftime")
        and not isinstance(item, pd.Timestamp)
        for item in index
    ):
        return [str(item) for item in index]

    try:
        # Explicitly opt into mixed-format parsing to avoid infer-format warnings
        datetime_index = pd.to_datetime(index, errors="raise", format="mixed")
        return [d.strftime("%b %y") for d in datetime_index]
    except (ValueError, TypeError):
        labels = []
        for item in index:
            try:
                if hasattr(item, "strftime"):
                    labels.append(item.strftime("%b %y"))
                else:
                    # Explicitly opt into mixed-format parsing for individual items
                    datetime_item = pd.to_datetime(item, errors="raise", format="mixed")
                    labels.append(datetime_item.strftime("%b %y"))
            except (ValueError, TypeError, AttributeError):
                labels.append(str(item))
        return labels


def apply_common_chart_styling(ax, breakdown):
    """Apply common chart styling to eliminate code duplication.

    This function consolidates the duplicate chart styling code that was
    present in both test_utils.py and utils.py.

    Args:
        ax: Matplotlib axis object
        breakdown: Data breakdown for labels
    """
    set_chart_style()
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("Month", labelpad=20)
    ax.set_ylabel("Number of items", labelpad=10)

    # Validate that breakdown has an index attribute
    if not hasattr(breakdown, "index"):
        raise TypeError(
            f"breakdown must have an 'index' attribute, got {type(breakdown).__name__}"
        )

    # Format labels from the data index
    labels = _format_index_labels(breakdown.index)

    # IMPORTANT: Set ticks to match data positions
    # (0-indexed positions for categorical data)
    # This ensures labels align with actual data points rather than adjusting labels
    # to match arbitrary matplotlib tick positions. This prevents misalignment issues
    # where tick positions don't align with data points, and ensures labels remain
    # correctly synchronized if used elsewhere after this function.
    num_data_points = len(breakdown.index)
    tick_positions = list(range(num_data_points))
    ax.set_xticks(tick_positions)

    # Set labels to match the data points (length already guaranteed to match)
    ax.set_xticklabels(labels, rotation=90, size="small")


def save_chart_with_styling(fig, output_file, title="Chart"):
    """Save chart with common styling and logging.

    Args:
        fig: Matplotlib figure object
        output_file: Output file path
        title: Chart title for logging
    """
    logger = logging.getLogger(__name__)

    logger.info("Writing %s chart to %s", title, output_file)
    fig.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close(fig)
