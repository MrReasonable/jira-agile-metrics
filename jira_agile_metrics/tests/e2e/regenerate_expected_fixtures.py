#!/usr/bin/env python3
"""Script to regenerate expected fixtures for e2e tests.

This script runs the CLI with the same configuration as the e2e test
to regenerate the expected CSV fixture files. It must use the exact
same configuration parameters as test_cli_e2e.py to ensure consistency.

Usage:
    python -m jira_agile_metrics.tests.e2e.regenerate_expected_fixtures
"""

import os
from pathlib import Path

# Setup path and imports using shared module
from jira_agile_metrics.tests.e2e.e2e_config import get_e2e_settings_dict
from jira_agile_metrics.tests.e2e.e2e_imports import (
    setup_path_and_import_calculators,
)

_modules = setup_path_and_import_calculators()
run_calculators = _modules["run_calculators"]
CFDCalculator = _modules["CFDCalculator"]
CycleTimeCalculator = _modules["CycleTimeCalculator"]
HistogramCalculator = _modules["HistogramCalculator"]
ScatterplotCalculator = _modules["ScatterplotCalculator"]
ThroughputCalculator = _modules["ThroughputCalculator"]
QueryManager = _modules["QueryManager"]
FileJiraClient = _modules["FileJiraClient"]


def main():
    """Regenerate expected fixtures using the same configuration as e2e test."""
    # Setup paths
    test_dir = Path(__file__).resolve().parent.parent
    fixtures_dir = test_dir / "fixtures" / "jira"
    expected_dir = test_dir / "fixtures" / "expected"
    expected_dir.mkdir(parents=True, exist_ok=True)

    # Create JIRA client and query manager
    jira_client = FileJiraClient(str(fixtures_dir))

    # Use the EXACT same configuration as test_cli_e2e.py
    # This includes all critical output configuration parameters:
    # - Throughput frequency: D
    # - Throughput window: 0
    # - Histogram window: 0
    # - Scatterplot window: 0
    # - Date format: "%Y-%m-%d"
    settings = get_e2e_settings_dict(expected_dir)

    query_manager = QueryManager(jira_client, settings)

    # Run calculators to generate expected files
    calculators = [
        CycleTimeCalculator,
        CFDCalculator,
        ScatterplotCalculator,
        HistogramCalculator,
        ThroughputCalculator,
    ]

    print("Regenerating expected fixtures with e2e test configuration...")
    print(f"Output directory: {expected_dir}")
    original_cwd = os.getcwd()
    try:
        os.chdir(expected_dir.parent)
        run_calculators(calculators, query_manager, settings)
        print(f"\nExpected fixtures regenerated in: {expected_dir}")
        print("Files created:")
        for f in sorted(expected_dir.glob("*.csv")):
            size = f.stat().st_size
            lines = len(f.read_text().splitlines())
            print(f"  - {f.name} ({size} bytes, {lines} lines)")
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
