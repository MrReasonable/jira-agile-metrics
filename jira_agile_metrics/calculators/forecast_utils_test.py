"""Tests for forecast utilities.

Covers calculation helpers, sampler wrappers, and Monte Carlo paths.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from .forecast_utils import (
    backlog_growth_sampler,
    burnup_monte_carlo_horizon,
    calculate_daily_backlog_growth,
    calculate_daily_throughput,
    run_single_trial,
    throughput_sampler,
)


def test_calculate_daily_throughput_empty_inputs():
    """Return empty DataFrame when throughput input is empty."""
    df = pd.DataFrame()
    out = calculate_daily_throughput(df, "done")
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_calculate_daily_throughput_groups_and_counts():
    """Group by date and count completed items correctly."""
    dates = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]).date
    df = pd.DataFrame({"done": dates, "key": ["A", "B", "C"]})
    out = calculate_daily_throughput(df, "done")
    # Expect counts per day with date index
    assert list(out.index.astype(str)) == ["2024-01-01", "2024-01-02"]
    assert list(out["throughput"]) == [2, 1]


def test_calculate_daily_backlog_growth_empty_inputs():
    """Return empty DataFrame when backlog input is empty."""
    df = pd.DataFrame()
    out = calculate_daily_backlog_growth(df, "backlog")
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_calculate_daily_backlog_growth_positive_diffs_only():
    """Keep only positive diffs when computing backlog growth."""
    series = pd.Series([10, 12, 11, 15], index=pd.date_range("2024-01-01", periods=4))
    df = pd.DataFrame({"b": series})
    out = calculate_daily_backlog_growth(df, "b")
    # diffs: [NaN, +2, -1, +4] -> keep positives [2, 4]
    assert list(out["backlog_growth"]) == [2, 4]


def test_sampler_wrappers_return_callables():
    """Sampler wrappers return callables producing numeric scalars."""
    # Throughput sampler: any DataFrame accepted by underlying function
    t_df = pd.DataFrame({"throughput": [1, 2, 3]})
    sampler_t = throughput_sampler(t_df, sample_buffer_size=5)
    assert callable(sampler_t)
    # Invoke and assert integer-like scalar output
    val_t = sampler_t()
    assert np.isscalar(val_t)
    assert isinstance(val_t, (int, np.integer))
    # Backlog growth sampler expects column named backlog_growth
    bg_df = pd.DataFrame({"backlog_growth": [0, 2, 1]})
    sampler_b = backlog_growth_sampler(bg_df, sample_buffer_size=5)
    assert callable(sampler_b)
    # Invoke and assert numeric scalar output
    val_b = sampler_b()
    assert np.isscalar(val_b)
    assert isinstance(val_b, (int, float, np.number))


def test_burnup_monte_carlo_horizon_missing_samplers_returns_empty():
    """Return empty result when samplers are not provided."""
    params: Dict[str, Any] = {"trials": 3}
    out = burnup_monte_carlo_horizon(params)
    assert not out


def test_run_single_trial_accumulates_values():
    """Single trial updates backlog and done cumulatively per period."""

    # Provide deterministic samplers
    def t_sampler() -> int:
        return 2

    def b_sampler() -> int:
        return 5

    params: Dict[str, Any] = {
        "forecast_dates": list(range(3)),
        "throughput_sampler": t_sampler,
        "backlog_growth_sampler": b_sampler,
        "initial_backlog": 10,
        "initial_done": 1,
    }

    res = run_single_trial(params, trial_num=7)
    # Stepwise updates per period: done += 2, backlog += 5 - 2
    # initial: backlog=10, done=1
    # after 3 periods: backlog=10 + 3*(3) = 19, done=1 + 3*2 = 7
    assert res["trial_num"] == 7
    assert res["final_backlog"] == 19
    assert res["final_done"] == 7
    assert len(res["backlog_trial"]) == 4  # initial + 3 periods
    assert len(res["done_trial"]) == 4


def test_burnup_monte_carlo_horizon_runs_trials_with_samplers():
    """Run multiple trials and ensure expected structure is returned."""
    rng = np.random.default_rng(0)

    def t_sampler() -> int:
        return int(rng.integers(1, 4))

    def b_sampler() -> int:
        return int(rng.integers(0, 6))

    params: Dict[str, Any] = {
        "trials": 5,
        "forecast_dates": list(range(2)),
        "throughput_sampler": t_sampler,
        "backlog_growth_sampler": b_sampler,
        "initial_backlog": 0,
        "initial_done": 0,
    }

    out = burnup_monte_carlo_horizon(params)
    assert out["num_trials"] == 5
    assert len(out["trials"]) == 5
    # Each trial should contain the expected keys
    for tr in out["trials"]:
        for key in (
            "trial_num",
            "backlog_trial",
            "done_trial",
            "final_backlog",
            "final_done",
        ):
            assert key in tr
