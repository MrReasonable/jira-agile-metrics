"""End-to-end test that runs the CLI and verifies produced outputs.

This test uses the file-backed JIRA client with JSON fixtures to avoid
network calls. It invokes the CLI flow, writes outputs to a temporary
directory, compares them to expected fixtures, and cleans up the
generated files if assertions succeed.
"""

import os
from pathlib import Path

import pandas as pd
import pytest

from jira_agile_metrics.cli import configure_argument_parser, run_command_line
from jira_agile_metrics.test_file_jira_client import FileJiraClient
from jira_agile_metrics.tests.e2e.e2e_config import get_e2e_config_yaml
from jira_agile_metrics.tests.e2e.e2e_helpers import write_config_and_get_parser_args


def _read_csv_for_comparison(file_path):
    """Read CSV file with appropriate parsing based on file type.

    Args:
        file_path: Path to CSV file to read

    Returns:
        DataFrame with appropriate index and date parsing
    """
    filename = file_path.name
    datetime_index_files = ["cfd.csv", "throughput.csv"]

    if filename in datetime_index_files:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    elif filename == "scatterplot.csv":
        df = pd.read_csv(file_path, parse_dates=["completed_date"])
        df = df.sort_values("completed_date").reset_index(drop=True)
    else:
        df = pd.read_csv(file_path)

    return df


def _compare_dataframes(actual_df, expected_df, filename):
    """Compare two DataFrames and raise with detailed error message on mismatch.

    Args:
        actual_df: Generated DataFrame
        expected_df: Expected DataFrame
        filename: Name of file being compared (for error messages)

    Raises:
        AssertionError: If DataFrames don't match, with detailed diff info
    """
    try:
        pd.testing.assert_frame_equal(actual_df, expected_df, check_dtype=False)
    except AssertionError:
        print(f"\nMismatch in {filename}:")
        print(f"  Generated shape: {actual_df.shape}")
        print(f"  Expected shape: {expected_df.shape}")
        if actual_df.shape == expected_df.shape:
            _print_column_differences(actual_df, expected_df)
        else:
            print("  Shape mismatch!")
            print(f"  Generated columns: {list(actual_df.columns)}")
            print(f"  Expected columns: {list(expected_df.columns)}")
        raise


def _print_column_differences(actual_df, expected_df):
    """Print detailed column-wise differences between DataFrames.

    Args:
        actual_df: Generated DataFrame
        expected_df: Expected DataFrame
    """
    diff_mask = actual_df != expected_df
    for col in actual_df.columns:
        if diff_mask[col].any():
            diff_count = diff_mask[col].sum()
            print(f"  Column '{col}': {diff_count} differences")
            diff_indices = actual_df[diff_mask[col]].index[:5]
            for idx in diff_indices:
                print(
                    f"    Row {idx}: "
                    f"got {actual_df.loc[idx, col]}, "
                    f"expected {expected_df.loc[idx, col]}"
                )


def _cleanup_output_files(output_files):
    """Remove generated output files.

    Args:
        output_files: List of file paths to remove
    """
    for file_path in output_files:
        try:
            file_path.unlink()
        except FileNotFoundError:
            # Already removed or not present; ignore
            pass


@pytest.mark.e2e
def test_cli_e2e_generates_expected_outputs_and_cleans_up(tmp_path, monkeypatch):
    """Run the CLI end-to-end and validate all CSV outputs match expected fixtures.

    This test exercises the full application flow:
    1. Config file parsing
    2. JIRA client creation (mocked with file-backed fixtures)
    3. QueryManager initialization
    4. All calculators running in sequence
    5. Output file generation
    6. Validation that all CSV outputs match expected fixtures exactly

    Validates the following CSV files:
    - cycletime.csv - cycle time data with workflow status dates
    - cfd.csv - cumulative flow diagram data
    - histogram.csv - cycle time histogram bins and counts
    - scatterplot.csv - completed items with cycle times
    - throughput.csv - daily throughput counts

    On success, remove the generated files to keep the workspace clean.
    """
    # Save original working directory to restore after test
    original_cwd = os.getcwd()

    try:
        # Arrange: point the JIRA client at test fixtures and patch the CLI to use it
        fixtures_dir = Path(__file__).resolve().parents[1] / "fixtures" / "jira"

        def _fake_get_jira_client(_connection):
            return FileJiraClient(str(fixtures_dir))

        # Patch the CLI module-level JIRA client getter used by run_command_line
        monkeypatch.setattr(
            "jira_agile_metrics.cli.get_jira_client", _fake_get_jira_client
        )

        # Use the same workflow configuration as functional tests
        # to match expected fixtures
        # Include all necessary settings to match functional test outputs exactly
        config_yaml = get_e2e_config_yaml(tmp_path)

        # Act: run the CLI pipeline using the same parser as production
        # Provide config via arg list to build a Namespace consistent with production
        parser, args = write_config_and_get_parser_args(
            tmp_path, config_yaml, configure_argument_parser
        )
        run_command_line(parser, args)

        # Assert: verify that all output files match expected fixtures exactly
        expected_fixtures_dir = (
            Path(__file__).resolve().parents[1] / "fixtures" / "expected"
        )

        # All CSV outputs should match expected fixtures exactly
        files_to_validate = {
            tmp_path / "cycletime.csv": expected_fixtures_dir / "cycletime.csv",
            tmp_path / "cfd.csv": expected_fixtures_dir / "cfd.csv",
            tmp_path / "histogram.csv": expected_fixtures_dir / "histogram.csv",
            tmp_path / "scatterplot.csv": expected_fixtures_dir / "scatterplot.csv",
            tmp_path / "throughput.csv": expected_fixtures_dir / "throughput.csv",
        }

        # Verify all files match expected fixtures
        for produced, expected in files_to_validate.items():
            assert produced.exists(), f"Missing expected output: {produced}"
            assert expected.exists(), f"Missing expected fixture: {expected}"

            actual_df = _read_csv_for_comparison(produced)
            expected_df = _read_csv_for_comparison(expected)
            _compare_dataframes(actual_df, expected_df, produced.name)

        # If we reached here, all assertions passed — cleanup generated outputs
        _cleanup_output_files(list(files_to_validate.keys()))
    finally:
        # Always restore the original working directory
        os.chdir(original_cwd)
