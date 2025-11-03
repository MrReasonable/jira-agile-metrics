#!/usr/bin/env python3
"""Script to validate e2e test outputs against expected fixtures.

This script runs the e2e test logic and compares the generated CSV files
with expected values, providing detailed output about matches and mismatches.
It uses the EXACT same configuration as test_cli_e2e.py to ensure consistency.

Usage:
    python -m jira_agile_metrics.tests.e2e.validate_e2e_outputs
"""

import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path

import pandas as pd

# Setup path and imports using shared module
from jira_agile_metrics.tests.e2e.e2e_config import get_e2e_config_yaml
from jira_agile_metrics.tests.e2e.e2e_helpers import write_config_and_get_parser_args
from jira_agile_metrics.tests.e2e.e2e_imports import setup_path_and_import_cli

_modules = setup_path_and_import_cli()
configure_argument_parser = _modules["configure_argument_parser"]
run_command_line = _modules["run_command_line"]
FileJiraClient = _modules["FileJiraClient"]
jira_agile_metrics_cli = _modules["cli_module"]


def _read_csv_for_comparison(file_path):
    """Read CSV file with appropriate parsing based on file type."""
    filename = file_path.name
    datetime_index_files = ["cfd.csv", "throughput.csv"]

    if filename in datetime_index_files:
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
    if filename == "scatterplot.csv":
        df = pd.read_csv(file_path, parse_dates=["completed_date"])
        return df.sort_values("completed_date").reset_index(drop=True)
    return pd.read_csv(file_path)


def _compare_single_file(generated, expected):
    """Compare a single CSV file pair and return match status and details."""
    if not generated.exists():
        print(f"  ✗ ERROR: Generated file {generated} does not exist!")
        return False

    if not expected.exists():
        print(f"  ✗ ERROR: Expected file {expected} does not exist!")
        return False

    gen_df = _read_csv_for_comparison(generated)
    exp_df = _read_csv_for_comparison(expected)

    print(f"  Generated shape: {gen_df.shape}")
    print(f"  Expected shape: {exp_df.shape}")

    try:
        pd.testing.assert_frame_equal(gen_df, exp_df, check_dtype=False)
        print("  ✓ MATCH: Files are identical")
        print(f"  ✓ Rows: {len(gen_df)}")
        print(f"  ✓ Columns: {list(gen_df.columns)}")
        return True
    except AssertionError:
        print("  ✗ MISMATCH: Files differ")
        _print_dataframe_differences(gen_df, exp_df)
        return False


def _print_dataframe_differences(gen_df, exp_df):
    """Print detailed differences between DataFrames."""
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
                    print(
                        f"      Row {idx}: "
                        f"generated={gen_val}, "
                        f"expected={exp_val}"
                    )
    else:
        print("  Shape mismatch detected")


def _get_e2e_config_yaml(tmp_path):
    """Generate the e2e test configuration YAML."""
    return get_e2e_config_yaml(tmp_path)


def _run_cli_with_config(tmp_path):
    """Run the CLI with the e2e test configuration."""
    config_yaml = _get_e2e_config_yaml(tmp_path)
    parser, args = write_config_and_get_parser_args(
        tmp_path, config_yaml, configure_argument_parser
    )
    original_cwd = os.getcwd()
    try:
        run_command_line(parser, args)
    finally:
        os.chdir(original_cwd)


def _compare_all_csv_files(tmp_path, expected_dir):
    """Compare all CSV files and return match status."""
    files_to_check = {
        "cycletime.csv": "cycletime.csv",
        "cfd.csv": "cfd.csv",
        "histogram.csv": "histogram.csv",
        "scatterplot.csv": "scatterplot.csv",
        "throughput.csv": "throughput.csv",
    }

    print("=" * 80)
    print("CSV File Validation Report")
    print("=" * 80)
    print()

    all_match = True
    for gen_file, exp_file in files_to_check.items():
        generated = tmp_path / gen_file
        expected = expected_dir / exp_file

        print(f"Checking {gen_file}...")
        print("-" * 80)

        if not _compare_single_file(generated, expected):
            all_match = False

        print()

    print("=" * 80)
    if all_match:
        print("✓ All CSV files match expected values!")
        return 0

    print("✗ Some CSV files do not match expected values!")
    print(f"\nGenerated files are in: {tmp_path}")
    print("(Files preserved for inspection)")
    return 1


def _validate_csv_files_internal(tmp_path, fixtures_dir, expected_dir):
    """Internal validation logic extracted from main function."""

    def _fake_get_jira_client(_connection):
        return FileJiraClient(str(fixtures_dir))

    original_get_jira = jira_agile_metrics_cli.get_jira_client
    jira_agile_metrics_cli.get_jira_client = _fake_get_jira_client

    try:
        _run_cli_with_config(tmp_path)
        return _compare_all_csv_files(tmp_path, expected_dir)
    finally:
        jira_agile_metrics_cli.get_jira_client = original_get_jira


def validate_csv_files():
    """Run e2e test and validate CSV outputs."""
    with tempfile.TemporaryDirectory(prefix="e2e_validation_") as tmpdir:
        tmp_path = Path(tmpdir)
        # Setup paths
        test_dir = Path(__file__).resolve().parent.parent
        fixtures_dir = test_dir / "fixtures" / "jira"
        expected_dir = test_dir / "fixtures" / "expected"

        result = _validate_csv_files_internal(tmp_path, fixtures_dir, expected_dir)

        # Preserve files on mismatch - move directory before context cleanup
        if result != 0:
            preserved_path = (
                tmp_path.parent / f"e2e_validation_preserved_{tmp_path.name}"
            )
            tmp_path.rename(preserved_path)
            print(f"\nFiles preserved at: {preserved_path}")
            # Directory moved, so TemporaryDirectory cleanup will be safe

        return result


if __name__ == "__main__":
    try:
        sys.exit(validate_csv_files())
    except (FileNotFoundError, PermissionError, OSError) as e:
        # Handle runtime I/O errors: log, print traceback, and return 1
        logging.error("I/O error during validation: %s", e)
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        # Development-time catch-all: log and re-raise so unexpected exceptions surface
        logging.error("Unexpected error during validation: %s", e)
        traceback.print_exc()
        raise
