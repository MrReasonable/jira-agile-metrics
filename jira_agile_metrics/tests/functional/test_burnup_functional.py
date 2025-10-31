"""Functional tests for burnup calculator."""

from jira_agile_metrics.calculator import run_calculators
from jira_agile_metrics.calculators.burnup import BurnupCalculator
from jira_agile_metrics.calculators.cfd import CFDCalculator
from jira_agile_metrics.calculators.cycletime import CycleTimeCalculator
from jira_agile_metrics.tests.functional.conftest import get_burnup_base_settings


def test_burnup_data_has_expected_values(query_manager, simple_cycle_settings):
    """Test that burnup calculator generates expected data values."""
    settings, _ = simple_cycle_settings
    settings = get_burnup_base_settings(settings)

    results = run_calculators(
        [CycleTimeCalculator, CFDCalculator, BurnupCalculator],
        query_manager,
        settings,
    )
    burnup_df = results[BurnupCalculator]

    # Expect burnup dataframe to include backlog and done columns
    assert list(burnup_df.columns) == ["Backlog", "Done"]
    # Done reaches 1 on 2021-01-20 and 2 on 2021-01-25
    assert int(burnup_df.loc["2021-01-20"].get("Done", 0)) == 1
    assert int(burnup_df.loc["2021-01-25"].get("Done", 0)) == 2
