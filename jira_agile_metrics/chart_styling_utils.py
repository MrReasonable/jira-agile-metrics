"""Common chart styling utilities to eliminate code duplication."""

import logging

import matplotlib.pyplot as plt
import seaborn as sns


def set_chart_style(style="whitegrid", despine=True):
    """Set seaborn chart style."""
    sns.set_style(style)
    if despine:
        sns.despine()


def apply_common_chart_styling(ax, breakdown):
    """Apply common chart styling to eliminate code duplication.

    This function consolidates the duplicate chart styling code that was
    present in both test_utils.py and utils.py.

    Args:
        ax: Matplotlib axis object
        breakdown: Data breakdown for labels
    """
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("Month", labelpad=20)
    ax.set_ylabel("Number of items", labelpad=10)

    labels = [d.strftime("%b %y") for d in breakdown.index]
    ax.set_xticklabels(labels, rotation=90, size="small")
    set_chart_style()


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
