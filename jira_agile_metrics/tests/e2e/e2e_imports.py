"""Shared import setup for e2e test scripts.

This module handles the common pattern of setting up sys.path
and importing modules needed by standalone e2e test scripts.
"""

import importlib
import sys
from pathlib import Path


def setup_path_and_import_cli():
    """Setup Python path and import CLI modules.

    Returns:
        Dictionary with 'configure_argument_parser', 'run_command_line',
        'FileJiraClient', and 'cli_module' keys

    Raises:
        SystemExit: If imports fail
    """
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))

    try:
        cli_module = importlib.import_module("jira_agile_metrics.cli")
        test_file_jira_client_module = importlib.import_module(
            "jira_agile_metrics.test_file_jira_client"
        )

        return {
            "configure_argument_parser": cli_module.configure_argument_parser,
            "run_command_line": cli_module.run_command_line,
            "FileJiraClient": test_file_jira_client_module.FileJiraClient,
            "cli_module": cli_module,
        }
    except ImportError as import_error:
        print(f"Error importing modules: {import_error}")
        print(f"Project root: {project_root}")
        sys.exit(1)


def setup_path_and_import_calculators():
    """Setup Python path and import calculator modules.

    Returns:
        Dictionary with calculator classes and QueryManager, FileJiraClient

    Raises:
        SystemExit: If imports fail
    """
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))

    try:
        calculator_module = importlib.import_module("jira_agile_metrics.calculator")
        cfd_module = importlib.import_module("jira_agile_metrics.calculators.cfd")
        cycletime_module = importlib.import_module(
            "jira_agile_metrics.calculators.cycletime"
        )
        histogram_module = importlib.import_module(
            "jira_agile_metrics.calculators.histogram"
        )
        scatterplot_module = importlib.import_module(
            "jira_agile_metrics.calculators.scatterplot"
        )
        throughput_module = importlib.import_module(
            "jira_agile_metrics.calculators.throughput"
        )
        querymanager_module = importlib.import_module("jira_agile_metrics.querymanager")
        test_file_jira_client_module = importlib.import_module(
            "jira_agile_metrics.test_file_jira_client"
        )

        return {
            "run_calculators": calculator_module.run_calculators,
            "CFDCalculator": cfd_module.CFDCalculator,
            "CycleTimeCalculator": cycletime_module.CycleTimeCalculator,
            "HistogramCalculator": histogram_module.HistogramCalculator,
            "ScatterplotCalculator": scatterplot_module.ScatterplotCalculator,
            "ThroughputCalculator": throughput_module.ThroughputCalculator,
            "QueryManager": querymanager_module.QueryManager,
            "FileJiraClient": test_file_jira_client_module.FileJiraClient,
        }
    except ImportError as import_error:
        print(f"Error importing modules: {import_error}")
        print(f"Project root: {project_root}")
        sys.exit(1)
