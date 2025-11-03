"""Shared configuration for e2e tests."""

from pathlib import Path


def _get_standard_cycle_config():
    """Get standard cycle configuration used across tests.

    Returns:
        List of cycle stage dictionaries
    """
    return [
        {"name": "Backlog", "statuses": ["Backlog"]},
        {"name": "Committed", "statuses": ["Next"]},
        {"name": "Build", "statuses": ["Build"]},
        {"name": "Test", "statuses": ["QA"]},
        {"name": "Done", "statuses": ["Done"]},
    ]


def get_e2e_config_yaml(output_directory):
    """Generate the standard e2e test configuration YAML.

    Args:
        output_directory: Path where output files should be written

    Returns:
        YAML configuration string for e2e tests
    """
    return f"""
Connection:
  Type: jira
  Domain: https://example.org

Query: project = TEST

Workflow:
  Backlog: Backlog
  Committed: Next
  Build: Build
  Test: QA
  Done: Done

Output:
  Cycle time data:
    - cycletime.csv
  CFD data: cfd.csv
  Scatterplot data: scatterplot.csv
  Histogram data: histogram.csv
  Throughput data: throughput.csv
  Throughput frequency: D
  Throughput window: 0
  Histogram window: 0
  Scatterplot window: 0
  Date format: "%Y-%m-%d"
  Output directory: {output_directory}
"""


def get_e2e_settings_dict(output_directory):
    """Generate the standard e2e test settings dictionary.

    Args:
        output_directory: Path where output files should be written

    Returns:
        Settings dictionary for e2e tests
    """
    output_path = Path(output_directory)
    return {
        "cycle": _get_standard_cycle_config(),
        "committed_column": "Committed",
        "done_column": "Done",
        "backlog_column": "Backlog",
        "attributes": {},
        "queries": [{"jql": "project=TEST"}],
        "query_attribute": None,
        "cycle_time_data": [str(output_path / "cycletime.csv")],
        "cfd_data": [str(output_path / "cfd.csv")],
        "cfd_chart": None,
        "cfd_window": 0,
        "scatterplot_data": [str(output_path / "scatterplot.csv")],
        "scatterplot_chart": None,
        "scatterplot_window": 0,
        "histogram_data": [str(output_path / "histogram.csv")],
        "histogram_chart": None,
        "histogram_window": 0,
        "throughput_data": [str(output_path / "throughput.csv")],
        "throughput_chart": None,
        "throughput_frequency": "D",
        "throughput_window": 0,
        "date_format": "%Y-%m-%d",
        "quantiles": [0.5, 0.85, 0.95],
    }
