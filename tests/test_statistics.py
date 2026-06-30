import numpy as np
import pandas as pd

from forex.backtest.statistics import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    per_period_sharpe,
    probabilistic_sharpe_ratio,
)


def _returns(mean, std, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, std, size=n))


def test_psr_high_for_strong_positive_sharpe():
    r = _returns(0.001, 0.005, n=2000)  # per-period SR ~0.2, long sample
    psr = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
    assert 0.99 < psr <= 1.0


def test_psr_is_half_for_exactly_zero_sharpe():
    # Demean so the observed Sharpe is exactly 0; PSR must then be exactly 0.5
    # (the numerator (SR - 0) is 0 regardless of skew/kurtosis).
    r = _returns(0.0, 0.01, n=2000)
    r = r - r.mean()
    psr = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
    assert abs(psr - 0.5) < 1e-9


def test_expected_max_sharpe_increases_with_trials():
    low = expected_max_sharpe(sr_trials_std=0.1, n_trials=5)
    high = expected_max_sharpe(sr_trials_std=0.1, n_trials=500)
    assert high > low > 0


def test_expected_max_sharpe_zero_for_single_trial():
    assert expected_max_sharpe(sr_trials_std=0.1, n_trials=1) == 0.0


def test_dsr_below_psr_when_many_trials():
    r = _returns(0.0005, 0.005, n=2000)
    psr = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
    dsr_few = deflated_sharpe_ratio(r, n_trials=2, sr_trials_std=0.05)
    dsr_many = deflated_sharpe_ratio(r, n_trials=1000, sr_trials_std=0.05)
    # Deflation only lowers confidence, and more trials lower it further.
    assert dsr_few <= psr + 1e-9
    assert dsr_many < dsr_few


def test_per_period_sharpe_matches_definition():
    r = _returns(0.001, 0.01, n=500)
    assert abs(per_period_sharpe(r) - r.mean() / r.std(ddof=1)) < 1e-12


def test_degenerate_inputs_return_nan():
    assert np.isnan(per_period_sharpe(pd.Series([0.01])))
    assert np.isnan(probabilistic_sharpe_ratio(pd.Series([0.01])))
