"""Bottleneck charts calculator for Jira Agile Metrics.

This module provides functionality to generate bottleneck visualizations from
cycle time data, including per-issue stacked bars, aggregate stacked bars,
and distribution charts.
"""

import logging
import pprint
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..calculator import Calculator
from ..common_constants import get_bottleneck_chart_settings
from .cycletime import CycleTimeCalculator

logger = logging.getLogger(__name__)


def calculate_column_durations(cycle_data, cycle_names):
    """Calculate duration spent in each cycle column for each issue.

    Args:
        cycle_data: DataFrame with cycle time data
        cycle_names: List of cycle column names

    Returns:
        DataFrame with durations in days for each issue and column
    """
    # Convert all cycle_name columns to datetime with error handling
    # Only select columns that exist in cycle_data (replicates row.get() behavior)
    existing_cols = [col for col in cycle_names if col in cycle_data.columns]
    if not existing_cols:
        # No matching columns, return empty DataFrame with expected structure
        duration_cols = [
            f"{cycle_names[i]}→{cycle_names[i + 1]}"
            for i in range(len(cycle_names) - 1)
        ]
        return pd.DataFrame(
            columns=duration_cols, index=cycle_data.get("key", cycle_data.index)
        )

    cycle_df = cycle_data[existing_cols].copy()
    for col in existing_cols:
        cycle_df[col] = pd.to_datetime(cycle_df[col], errors="coerce")

    # Compute pairwise differences using vectorized operations
    # Only compute durations for pairs where both columns exist
    duration_dict = {}
    for i in range(len(cycle_names) - 1):
        curr_col = cycle_names[i]
        next_col = cycle_names[i + 1]

        # Only create duration if both columns exist
        if curr_col in existing_cols and next_col in existing_cols:
            duration_col_name = f"{curr_col}→{next_col}"
            # Compute duration as days between adjacent timestamps
            # NaT values will produce NaN in the resulting duration
            duration_dict[duration_col_name] = (
                cycle_df[next_col] - cycle_df[curr_col]
            ).dt.days
        elif curr_col not in existing_cols or next_col not in existing_cols:
            # Create column with NaN values when one or both columns are missing
            duration_col_name = f"{curr_col}→{next_col}"
            duration_dict[duration_col_name] = np.full(len(cycle_data), np.nan)

    # Assemble into DataFrame with index=cycle_data["key"]
    result_df = pd.DataFrame(
        duration_dict, index=cycle_data.get("key", cycle_data.index)
    )
    return result_df


def calculate_column_durations_per_column(
    cycle_data, cycle_names, negative_duration_handling="zero"
):
    """Calculate column durations with per-column handling.

    Args:
        cycle_data: DataFrame with cycle time data
        cycle_names: List of cycle column names
        negative_duration_handling: How to handle negative durations
            ("zero", "nan", "abs")

    Returns:
        DataFrame with column durations

    Raises:
        ValueError: If negative_duration_handling is not one of {"zero", "nan", "abs"}
    """
    # Validate input parameter
    if negative_duration_handling not in {"zero", "nan", "abs"}:
        raise ValueError(
            f"negative_duration_handling must be one of {{'zero', 'nan', 'abs'}}, "
            f"got '{negative_duration_handling}'"
        )

    # Returns a DataFrame: rows=issues, columns=cycle columns (except last),
    # values=duration in days spent in each column
    # Check which columns exist in cycle_data
    existing_cols = [col for col in cycle_names if col in cycle_data.columns]
    if not existing_cols:
        # No matching columns, return empty DataFrame with expected structure
        column_names = cycle_names[:-1]
        return pd.DataFrame(
            columns=column_names, index=cycle_data.get("key", cycle_data.index)
        )

    # Convert timestamp columns to datetime with error handling
    cycle_df = cycle_data[existing_cols].copy()
    for col in existing_cols:
        cycle_df[col] = pd.to_datetime(cycle_df[col], errors="coerce")

    # Get the proper index for the result DataFrame
    # Try to get "key" column if it exists, otherwise use the original index
    if "key" in cycle_data.columns:
        result_index = cycle_data["key"].values
    else:
        result_index = cycle_data.index

    # Compute durations using vectorized operations
    durations_dict = {}
    for i in range(len(cycle_names) - 1):
        curr_col = cycle_names[i]
        next_col = cycle_names[i + 1]

        # Only create duration if both columns exist
        if curr_col in existing_cols and next_col in existing_cols:
            # Compute vectorized differences using .dt.days
            # Note: cycle_df has the same index as cycle_data after copy
            diff = (cycle_df[next_col] - cycle_df[curr_col]).dt.days

            # Handle negative durations with vectorized boolean masks
            if negative_duration_handling == "zero":
                diff = diff.where(diff >= 0, 0)
            elif negative_duration_handling == "nan":
                diff = diff.where(diff >= 0, np.nan)
            elif negative_duration_handling == "abs":
                diff = diff.abs()

            # Convert Series to array to avoid index alignment issues
            durations_dict[curr_col] = diff.values
        else:
            # One or both columns missing, create array of NaN values
            durations_dict[curr_col] = np.full(len(cycle_df), np.nan, dtype=float)

    # Construct DataFrame with original index
    result_df = pd.DataFrame(durations_dict, index=result_index)
    return result_df


class BottleneckChartsCalculator(Calculator):
    """
    Generates bottleneck visualizations: per-issue stacked bar, aggregate stacked bar,
    and box/violin plots.
    """

    def run(self):
        cycle_data = self.get_result(CycleTimeCalculator)
        cycle_names = [s["name"] for s in self.settings["cycle"]]
        # Use "zero" when setting is missing or explicitly None
        negative_duration_handling = (
            self.settings.get("negative_duration_handling") or "zero"
        )
        # Return both transition durations and per-column durations
        return {
            "transitions": calculate_column_durations(cycle_data, cycle_names),
            "columns": calculate_column_durations_per_column(
                cycle_data, cycle_names, negative_duration_handling
            ),
        }

    def write(self):
        results = self.get_result()
        durations_columns = results["columns"]
        output_settings = self.settings
        logger.debug(
            "[BottleneckChartsCalculator] output_settings: %s",
            pprint.pformat(output_settings),
        )
        for key in get_bottleneck_chart_settings():
            logger.debug(
                "[BottleneckChartsCalculator] %s: %s",
                key,
                output_settings.get(key),
            )
        # 1. Stacked bar per issue (now uses per-column durations)
        if output_settings.get("bottleneck_stacked_per_issue_chart"):
            # Allow overriding the number of issues via settings; default remains 30
            configured_max_issues = output_settings.get("bottleneck_max_issues", 30)
            try:
                # Validate positive integer; coerce from str if provided
                max_issues = int(configured_max_issues)
                if max_issues <= 0:
                    raise ValueError("max_issues must be > 0")
            except (ValueError, TypeError):
                logger.warning(
                    "Invalid bottleneck_max_issues=%r; falling back to 30",
                    configured_max_issues,
                )
                max_issues = 30
            self.write_stacked_per_issue(
                durations_columns,
                output_settings["bottleneck_stacked_per_issue_chart"],
                max_issues=max_issues,
            )
        # 2. Aggregated stacked bar (mean, by column)
        if output_settings.get("bottleneck_stacked_aggregate_mean_chart"):
            self.write_stacked_aggregate(
                durations_columns,
                output_settings["bottleneck_stacked_aggregate_mean_chart"],
                aggfunc="mean",
            )
        # 3. Aggregated stacked bar (median, by column)
        if output_settings.get("bottleneck_stacked_aggregate_median_chart"):
            self.write_stacked_aggregate(
                durations_columns,
                output_settings["bottleneck_stacked_aggregate_median_chart"],
                aggfunc="median",
            )
        # 4. Boxplot (by column)
        if output_settings.get("bottleneck_boxplot_chart"):
            self.write_boxplot(
                durations_columns, output_settings["bottleneck_boxplot_chart"]
            )
        # 5. Violin plot (by column)
        if output_settings.get("bottleneck_violin_chart"):
            self.write_violin(
                durations_columns, output_settings["bottleneck_violin_chart"]
            )

    def write_stacked_per_issue(self, durations, output_file, max_issues=30):
        """Write stacked per issue bottleneck chart to file.

        Args:
            durations: DataFrame of per-column durations per issue.
            output_file: Path to write the generated chart image.
            max_issues: Maximum number of issues to include in the chart. Must be
                a validated positive integer. Callers must ensure this parameter
                has been validated before calling this method.
        """
        try:
            logger.info("Writing bottleneck stacked per issue chart to %s", output_file)
            plot_data = durations.dropna(how="all").iloc[:max_issues]
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_data.plot(kind="bar", stacked=True, ax=ax)
            ax.set_xlabel("Issue key")
            ax.set_ylabel("Days in column")
            ax.set_title(
                f"Time spent in each column (per issue, first {max_issues} issues)"
            )
            plt.xticks(rotation=90)
            plt.tight_layout()
            fig.savefig(output_file, bbox_inches="tight", dpi=300)
            plt.close(fig)
            logger.info("Successfully wrote %s", output_file)
        except (IOError, OSError, ValueError, TypeError) as e:
            logger.error(
                "Error writing bottleneck stacked per issue chart to %s: %s\n%s",
                output_file,
                e,
                traceback.format_exc(),
            )

    def write_stacked_aggregate(self, durations, output_file, aggfunc="mean"):
        """Write stacked aggregate bottleneck chart to file."""
        try:
            logger.info("Writing bottleneck stacked aggregate chart to %s", output_file)
            if aggfunc == "mean":
                agg = durations.mean(skipna=True)
                title = "Average time spent in each column (all issues)"
            else:
                agg = durations.median(skipna=True)
                title = "Median time spent in each column (all issues)"
            fig, ax = plt.subplots(figsize=(10, 5))
            agg.plot(
                kind="bar",
                stacked=False,
                ax=ax,
                color=sns.color_palette("tab10"),
            )
            ax.set_xlabel("Column")
            ax.set_ylabel("Days")
            ax.set_title(title)
            plt.xticks(rotation=45)
            plt.tight_layout()
            fig.savefig(output_file, bbox_inches="tight", dpi=300)
            plt.close(fig)
            logger.info("Successfully wrote %s", output_file)
        except (IOError, OSError, ValueError, TypeError) as e:
            logger.error(
                "Error writing bottleneck stacked aggregate chart to %s: %s\n%s",
                output_file,
                e,
                traceback.format_exc(),
            )

    def write_boxplot(self, durations, output_file):
        """Write bottleneck boxplot chart to file."""
        try:
            logger.info("Writing bottleneck boxplot chart to %s", output_file)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=durations, ax=ax)
            ax.set_xlabel("Column")
            ax.set_ylabel("Days")
            ax.set_title("Distribution of time spent in each column (boxplot)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            fig.savefig(output_file, bbox_inches="tight", dpi=300)
            plt.close(fig)
            logger.info("Successfully wrote %s", output_file)
        except (IOError, OSError, ValueError, TypeError) as e:
            logger.error(
                "Error writing bottleneck boxplot chart to %s: %s\n%s",
                output_file,
                e,
                traceback.format_exc(),
            )

    def write_violin(self, durations, output_file):
        """Write violin plot chart for bottleneck analysis."""
        try:
            logger.info("Writing bottleneck violin chart to %s", output_file)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.violinplot(data=durations, ax=ax, cut=0)
            ax.set_xlabel("Column")
            ax.set_ylabel("Days")
            ax.set_title("Distribution of time spent in each column (violin plot)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            fig.savefig(output_file, bbox_inches="tight", dpi=300)
            plt.close(fig)
            logger.info("Successfully wrote %s", output_file)
        except (IOError, OSError, ValueError, TypeError) as e:
            logger.error(
                "Error writing bottleneck violin chart to %s: %s\n%s",
                output_file,
                e,
                traceback.format_exc(),
            )
