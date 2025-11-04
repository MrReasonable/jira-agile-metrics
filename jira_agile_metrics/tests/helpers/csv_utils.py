"""CSV test utilities shared across e2e tests.

Provides helpers to read CSVs consistently for comparisons and to print
human-readable differences between DataFrames when assertions fail.
"""

from pathlib import Path

import pandas as pd

__all__ = [
    "read_csv_for_comparison",
    "print_dataframe_differences",
]


def read_csv_for_comparison(file_path: Path) -> pd.DataFrame:
    """Read CSV file with appropriate parsing based on file type.

    Applies consistent parsing rules for known CSV outputs so that equality
    comparisons in tests are stable and deterministic.
    """
    filename = file_path.name
    datetime_index_files = ["cfd.csv", "throughput.csv"]

    if filename in datetime_index_files:
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
    if filename == "scatterplot.csv":
        df = pd.read_csv(file_path, parse_dates=["completed_date"])
        return df.sort_values("completed_date").reset_index(drop=True)
    return pd.read_csv(file_path)


def print_dataframe_differences(gen_df: pd.DataFrame, exp_df: pd.DataFrame) -> None:
    """Print detailed differences between DataFrames.

    Shows per-column counts and up to five example row differences per column
    when shapes match; otherwise reports a shape mismatch.
    """
    if gen_df.shape == exp_df.shape:
        diff_mask = gen_df != exp_df
        print("\n  Differences by column:")
        for col in gen_df.columns:
            if diff_mask[col].any():
                diff_count = diff_mask[col].sum()
                print(f"    '{col}': {diff_count} different value(s)")
                diff_indices = gen_df[diff_mask[col]].index[:5]
                for idx in diff_indices:
                    gen_val = gen_df.loc[idx, col]
                    exp_val = exp_df.loc[idx, col]
                    print(f"      Row {idx}: generated={gen_val}, expected={exp_val}")
    else:
        print("  Shape mismatch detected")
