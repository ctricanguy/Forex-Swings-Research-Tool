"""The honest backtest engine.

Three things make a backtest honest, and all three live here so no strategy can
forget them:

1. **Act on completed bars only.** The incomplete tail bar is dropped.
2. **One-bar execution shift.** You decide on bar ``t``'s close and are filled
   over ``t -> t+1``. We apply ``position.shift(1)`` exactly once, centrally.
   The live runner uses the same convention, so backtest == live.
3. **Explicit costs.** Every change in position pays half the bid/ask spread per
   side; held positions accrue swap/rollover. These costs routinely halve or
   erase a swing strategy's gross edge, so they are never optional here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from forex.backtest.metrics import compute_metrics
from forex.strategies.base import Strategy


@dataclass(slots=True)
class CostModel:
    """Transaction-cost assumptions.

    Parameters
    ----------
    use_quoted_spread:
        If True (default), the per-bar spread is read from the ``bid``/``ask``
        columns. If False, ``fixed_spread_frac`` is used for every bar.
    fixed_spread_frac:
        Fallback full spread as a fraction of price (e.g. ``0.0001`` ~= 1 pip
        on a 1.0 quote) when quoted spreads are unavailable.
    spread_fill:
        Fraction of the quoted spread you actually pay. Menkhoff et al. (2012)
        note effective spreads are often 50-75% of quoted; ``1.0`` is the
        conservative full-spread assumption.
    swap_per_bar:
        Signed financing accrued per bar per unit of (long) exposure. Positive
        means a long earns carry; a short pays it (sign follows the position).
        Default 0.0 -- set it from the pair's real swap to model overnight cost.
    """

    use_quoted_spread: bool = True
    fixed_spread_frac: float = 0.0001
    spread_fill: float = 1.0
    swap_per_bar: float = 0.0


@dataclass(slots=True)
class BacktestResult:
    returns: pd.Series  # net per-bar returns
    equity: pd.Series  # cumulative equity curve (starts near 1.0)
    position: pd.Series  # position actually held each bar (post-shift)
    metrics: Dict[str, float] = field(default_factory=dict)
    granularity: str = "D"

    def summary(self) -> str:
        m = self.metrics
        return (
            "Backtest summary\n"
            f"  bars            : {int(m.get('bars', 0))}\n"
            f"  total return    : {m.get('total_return', 0):+.2%}\n"
            f"  ann. return     : {m.get('ann_return', 0):+.2%}\n"
            f"  ann. volatility : {m.get('ann_vol', 0):.2%}\n"
            f"  Sharpe          : {m.get('sharpe', 0):.2f}\n"
            f"  max drawdown    : {m.get('max_drawdown', 0):.2%}\n"
            f"  trades          : {int(m.get('num_trades', 0))}\n"
            f"  exposure        : {m.get('exposure', 0):.1%}\n"
            f"  win rate        : {m.get('win_rate', 0):.1%}\n"
            f"  profit factor   : {m.get('profit_factor', 0):.2f}"
        )


def _spread_frac(data: pd.DataFrame, costs: CostModel) -> pd.Series:
    if costs.use_quoted_spread and {"bid", "ask"}.issubset(data.columns):
        mid = data["close"].replace(0, np.nan)
        spread = (data["ask"] - data["bid"]) / mid
        spread = spread.fillna(costs.fixed_spread_frac)
    else:
        spread = pd.Series(costs.fixed_spread_frac, index=data.index)
    return spread.clip(lower=0.0) * costs.spread_fill


def run_backtest(
    strategy: Strategy,
    data: pd.DataFrame,
    *,
    costs: CostModel | None = None,
    granularity: str = "D",
) -> BacktestResult:
    """Run ``strategy`` over ``data`` and return a :class:`BacktestResult`."""
    costs = costs or CostModel()

    # (1) Completed bars only.
    if "complete" in data.columns:
        data = data[data["complete"].astype(bool)]
    data = data.sort_index()

    # Desired position from the shared strategy code (point-in-time).
    desired = strategy.generate_signals(data)["position"].astype(float)

    # (2) One-bar execution shift -- the single, central lookahead guard.
    held = desired.shift(1).fillna(0.0)

    # Market move of the mid price.
    bar_return = data["close"].pct_change(fill_method=None).fillna(0.0)
    gross = held * bar_return

    # (3a) Spread cost: pay half the spread per side whenever exposure changes.
    spread = _spread_frac(data, costs)
    turnover = held.diff().abs().fillna(held.abs())
    trade_cost = turnover * (spread / 2.0)

    # (3b) Swap/rollover on the held position.
    swap = held * costs.swap_per_bar

    net = gross - trade_cost + swap
    equity = (1.0 + net).cumprod()

    metrics = compute_metrics(net, held, granularity=granularity)
    return BacktestResult(
        returns=net,
        equity=equity,
        position=held,
        metrics=metrics,
        granularity=granularity,
    )


def detect_lookahead(
    strategy: Strategy,
    data: pd.DataFrame,
    *,
    probe: int = 25,
) -> Dict[str, float]:
    """Diagnostic for hidden lookahead bias (the truncation test).

    The defining property of a point-in-time strategy: the position computed for
    bar ``k`` must be identical whether or not the strategy can see bars after
    ``k``. We verify it directly -- for the last ``probe`` bars, recompute the
    position using only data up to and including bar ``k`` and compare it to the
    position computed with the full history.

    * Honest strategy  -> the two always match (``leaking_bars == 0``).
    * Future-peeking    -> they differ wherever the strategy read ahead.

    This is more robust than delaying features, because delaying the price column
    shifts the signal and the return series together and would hide a ``shift(-1)``
    leak. It is the same idea behind Freqtrade's lookahead analysis: a signal must
    not change when future candles are withheld.
    """
    if "complete" in data.columns:
        data = data[data["complete"].astype(bool)]
    data = data.sort_index()

    full = strategy.generate_signals(data)["position"].astype(float)
    n = len(data)
    probe = min(probe, n - 1)

    leaking = 0
    max_diff = 0.0
    for k in range(n - probe, n):
        partial = strategy.generate_signals(data.iloc[: k + 1])["position"].astype(float)
        diff = abs(float(partial.iloc[-1]) - float(full.iloc[k]))
        max_diff = max(max_diff, diff)
        if diff > 1e-9:
            leaking += 1

    return {
        "leaking_bars": float(leaking),
        "max_position_diff": max_diff,
        "probe": float(probe),
    }
