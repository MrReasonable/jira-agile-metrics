"""Test helpers for normalizing pandas DataFrames for stable comparisons.

These utilities trim string whitespace, impose deterministic column and row
ordering, and reset indexes so that CSV/fixture comparisons are consistent
across environments.
"""

from __future__ import annotations

import pandas as pd


def normalize_dataframe_for_csv_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy of a DataFrame for robust CSV comparisons.

    Normalization steps:
    - Trim whitespace in all object (string) columns
    - Sort columns by name for a stable column order
    - Sort rows deterministically by all columns
    - Reset index
    """
    # Work on a copy to avoid mutating caller's DataFrame
    normalized = df.copy()

    # Trim whitespace on object columns
    object_columns = normalized.select_dtypes(include=["object"]).columns
    for column_name in object_columns:
        normalized[column_name] = normalized[column_name].astype(str).str.strip()

    # Stable column order
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)

    # Stable row order across all columns
    if len(normalized.columns) > 0:
        normalized = normalized.sort_values(by=list(normalized.columns)).reset_index(
            drop=True
        )
    else:
        normalized = normalized.reset_index(drop=True)

    return normalized


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a DataFrame for deterministic comparisons.

    This is a thin wrapper to keep a concise, shared helper name in tests.
    """
    return normalize_dataframe_for_csv_comparison(df)
