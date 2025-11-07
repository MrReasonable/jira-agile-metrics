"""Generate expected forecast CSV for functional testing.

This script generates a reproducible forecast CSV file with a fixed random seed
for use in functional tests. The expected output should match exactly when
run with the same seed.

Usage:
    python -m jira_agile_metrics.tests.functional.generate_expected_forecast_csv
"""

import os
import random
from pathlib import Path

import numpy as np

from jira_agile_metrics.calculator import run_calculators
from jira_agile_metrics.calculators.burnup import BurnupCalculator
from jira_agile_metrics.calculators.cfd import CFDCalculator
from jira_agile_metrics.calculators.cycletime import CycleTimeCalculator
from jira_agile_metrics.calculators.forecast import BurnupForecastCalculator
from jira_agile_metrics.querymanager import QueryManager
from jira_agile_metrics.test_file_jira_client import FileJiraClient
from jira_agile_metrics.tests.functional.conftest import (
    _create_base_settings,
    _create_query_manager_settings,
    fixtures_path,
    get_burnup_base_settings,
    get_default_forecast_settings,
)


def main():
    """Generate expected forecast CSV file."""
    # Set fixed random seed for reproducible results
    random.seed(42)
    np.random.seed(42)

    # Setup test environment
    client = FileJiraClient(fixtures_path("jira"))
    query_manager = QueryManager(client, settings=_create_query_manager_settings())

    # Configure settings matching test_burnup_forecast_generates_csv_output
    # Use helper function from conftest to avoid code duplication
    settings = _create_base_settings()
    settings = get_burnup_base_settings(settings)
    settings.update(get_default_forecast_settings())
    settings.update({
        "burnup_forecast_chart": os.devnull,
        "burnup_forecast_chart_data": None,
        "burnup_forecast_chart_trials": 20,
    })

    # Determine output path
    fixtures_dir = Path(__file__).resolve().parent.parent / "fixtures" / "expected"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    output_csv = fixtures_dir / "burnup-forecast.csv"

    # Run calculators
    print("Running forecast calculators with seed 42...")
    results = run_calculators(
        [
            CycleTimeCalculator,
            CFDCalculator,
            BurnupCalculator,
            BurnupForecastCalculator,
        ],
        query_manager,
        settings,
    )

    forecast_result = results[BurnupForecastCalculator]
    if forecast_result is None:
        print("ERROR: Forecast could not be generated")
        return 1

    # Save to CSV
    print(f"Writing expected forecast CSV to {output_csv}...")
    forecast_result.to_csv(output_csv, index=True, index_label="Date")
    print(f"Successfully generated expected CSV with {len(forecast_result)} rows")
    print(f"Date range: {forecast_result.index.min()} to {forecast_result.index.max()}")
    print(f"Number of trials: {len(forecast_result.columns)}")

    # Verify it starts at the last burnup date
    burnup_data = results[BurnupCalculator]
    if not burnup_data.empty:
        last_burnup_date = burnup_data.index[-1]
        first_forecast_date = forecast_result.index[0]
        print(f"Last burnup date: {last_burnup_date}")
        print(f"First forecast date: {first_forecast_date}")
        if first_forecast_date == last_burnup_date:
            print("✓ Forecast correctly starts at last burnup date")
        else:
            print(
                f"⚠ WARNING: Forecast starts at {first_forecast_date}, "
                f"expected {last_burnup_date}"
            )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
