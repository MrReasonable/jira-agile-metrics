"""Tests for calculator functionality in Jira Agile Metrics.

This module contains unit tests for the base Calculator class and calculator utilities.
"""

from .calculator import Calculator, run_calculators


def test_run_calculator():
    """Test run_calculator functionality."""
    written = []

    class Enabled(Calculator):
        """Test calculator that is enabled."""

        def run(self):
            return "Enabled"

        def write(self):
            written.append("Enabled")

    class Disabled(Calculator):
        """Test calculator that is disabled."""

        def run(self):
            return "Disabled"

        def write(self):
            pass

    class GetPreviousResult(Calculator):
        """Test calculator that gets previous results."""

        def run(self):
            return self.get_result(Enabled) + " " + self.settings["foo"]

        def write(self):
            written.append(self.get_result())

    calculators = [Enabled, Disabled, GetPreviousResult]
    query_manager = object()
    settings = {"foo": "bar"}

    results = run_calculators(calculators, query_manager, settings)

    assert results == {
        Enabled: "Enabled",
        Disabled: "Disabled",
        GetPreviousResult: "Enabled bar",
    }

    assert written == ["Enabled", "Enabled bar"]
