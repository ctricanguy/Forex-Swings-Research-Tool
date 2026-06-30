"""Probabilistic and Deflated Sharpe Ratio.

Implements Bailey & Lopez de Prado's machinery for the multiple-testing problem:
after trying many configurations, a high backtest Sharpe is trivially achievable
by luck, so the reported Sharpe must be *deflated* by how many trials were run.

* **PSR** -- Probabilistic Sharpe Ratio: the probability the true Sharpe exceeds
  a benchmark, given the observed Sharpe, sample length, skew and kurtosis.
* **DSR** -- Deflated Sharpe Ratio: PSR where the benchmark is the *expected
  maximum* Sharpe achievable from ``N`` independent zero-skill trials.

All Sharpe ratios here are in **per-period** units (mean / std of the per-bar
returns), not annualised -- the ``sqrt(T-1)`` term carries the sample length.
Uses only the standard library (``statistics.NormalDist``) + pandas.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Optional, Sequence, Tuple

import pandas as pd

_NORM = NormalDist()
_EULER_MASCHERONI = 0.5772156649015329


def per_period_sharpe(returns: pd.Series) -> float:
    """Mean / std of returns (per-period, non-annualised)."""
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else float("nan")


def _moments(returns: pd.Series) -> Tuple[int, float, float, float]:
    r = pd.Series(returns).dropna()
    T = len(r)
    sr = per_period_sharpe(r)
    skew = float(r.skew()) if T > 2 else 0.0
    # pandas .kurt() is excess (Fisher) kurtosis; the PSR formula uses the
    # Pearson 4th standardised moment (Gaussian == 3).
    kurt = float(r.kurt()) + 3.0 if T > 3 else 3.0
    return T, sr, skew, kurt


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sr: float = 0.0) -> float:
    """P(true per-period Sharpe > ``benchmark_sr``) given the sample.

    ``benchmark_sr`` is in per-period units (use 0 for "better than nothing").
    """
    T, sr, skew, kurt = _moments(returns)
    if T < 2 or math.isnan(sr):
        return float("nan")
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr))
    z = (sr - benchmark_sr) * math.sqrt(T - 1) / denom
    return _NORM.cdf(z)


def expected_max_sharpe(sr_trials_std: float, n_trials: int) -> float:
    """Expected maximum per-period Sharpe from ``n_trials`` zero-skill trials.

    ``sr_trials_std`` is the cross-trial standard deviation of the (per-period)
    Sharpe estimates produced by the search. With one trial there is nothing to
    deflate, so this returns 0.
    """
    if n_trials is None or n_trials < 2 or not (sr_trials_std > 0):
        return 0.0
    a = _NORM.inv_cdf(1.0 - 1.0 / n_trials)
    b = _NORM.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return sr_trials_std * ((1.0 - _EULER_MASCHERONI) * a + _EULER_MASCHERONI * b)


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    sr_trials_std: float,
) -> float:
    """Deflated Sharpe Ratio: PSR against the expected-maximum-Sharpe benchmark.

    Parameters
    ----------
    returns:
        The selected strategy's per-bar return series.
    n_trials:
        How many configurations were tried to obtain this strategy.
    sr_trials_std:
        Cross-trial std of the per-period Sharpe estimates over those trials.
    """
    sr_star = expected_max_sharpe(sr_trials_std, n_trials)
    return probabilistic_sharpe_ratio(returns, benchmark_sr=sr_star)


def trial_sharpe_std(trial_returns: Sequence[pd.Series]) -> float:
    """Cross-trial std of per-period Sharpes, for feeding the DSR deflation."""
    sharpes = pd.Series([per_period_sharpe(r) for r in trial_returns])
    return float(sharpes.std(ddof=1)) if sharpes.notna().sum() > 1 else float("nan")
