"""Cross-sectional portfolio backtest.

Turns a :class:`~forex.strategies.cross_sectional.CrossSectionalSignal` into a
long/short, dollar-neutral book and evaluates it as a :class:`Backtestable` --
so it plugs into the *same* walk-forward / holdout harness as a single-pair
strategy, with no harness changes.

Construction: each bar, rank pairs by score; long the top ``quantile`` fraction,
short the bottom ``quantile`` fraction, equal-weight within each leg, scaled so
gross exposure equals ``gross_leverage`` (default 1.0) and net exposure is ~0.

The same honesty rules as the per-pair engine apply here, in one place:
completed-bar input, a single central one-bar execution shift on the weights,
and explicit per-pair spread costs on weight changes.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Type

import numpy as np
import pandas as pd

from forex.backtest.engine import BacktestResult, CostModel
from forex.backtest.evaluation import Backtestable, _reslice
from forex.backtest.metrics import compute_metrics
from forex.data.panel import Panel
from forex.strategies.cross_sectional import CrossSectionalSignal


def rank_weights(
    scores: pd.DataFrame,
    quantile: float = 1.0 / 3.0,
    gross_leverage: float = 1.0,
) -> pd.DataFrame:
    """Convert a wide score frame into dollar-neutral long/short weights.

    For each row: long the top ``quantile`` of scored pairs, short the bottom
    ``quantile``, equal weight per name, scaled to ``gross_leverage`` total gross
    exposure (sum of absolute weights). Rows with fewer than two scored pairs are
    all-flat. ``NaN`` scores never receive weight.
    """
    if not 0 < quantile <= 0.5:
        raise ValueError("quantile must be in (0, 0.5]")

    ranks = scores.rank(axis=1, method="first", ascending=True)
    count = scores.notna().sum(axis=1).to_numpy(dtype=float)
    k = np.floor(count * quantile)
    k = np.where(count >= 2, np.maximum(k, 1.0), 0.0)

    rv = ranks.to_numpy(dtype=float)  # NaN preserved -> never selected
    count_b = count[:, None]
    k_b = k[:, None]
    per_leg = np.divide(
        0.5 * gross_leverage, k_b, out=np.zeros_like(k_b), where=k_b > 0
    )
    per_leg = np.broadcast_to(per_leg, rv.shape)

    with np.errstate(invalid="ignore"):
        long_mask = rv > (count_b - k_b)
        short_mask = rv <= k_b

    weights = np.zeros_like(rv)
    weights[long_mask] = per_leg[long_mask]
    weights[short_mask] = -per_leg[short_mask]
    weights[np.isnan(rv)] = 0.0
    return pd.DataFrame(weights, index=scores.index, columns=scores.columns)


class PortfolioBacktest(Backtestable):
    def __init__(
        self,
        signal: CrossSectionalSignal,
        panel: Panel,
        *,
        costs: Optional[CostModel] = None,
        granularity: str = "D",
        quantile: float = 1.0 / 3.0,
        gross_leverage: float = 1.0,
    ) -> None:
        self.signal = signal
        self.panel = panel
        self.costs = costs or CostModel()
        self.granularity = granularity
        self.quantile = quantile
        self.gross_leverage = gross_leverage
        self.name = getattr(signal, "name", "portfolio")

    def _spread(self, held: pd.DataFrame) -> pd.DataFrame:
        if self.costs.use_quoted_spread:
            spread = self.panel.spread_frac().reindex_like(held)
            spread = spread.fillna(self.costs.fixed_spread_frac)
        else:
            spread = pd.DataFrame(
                self.costs.fixed_spread_frac, index=held.index, columns=held.columns
            )
        return spread.clip(lower=0.0) * self.costs.spread_fill

    def run(self, start=None, end=None) -> BacktestResult:
        panel = self.panel.upto(end)  # never see beyond the window end
        scores = self.signal.score(panel)
        weights = rank_weights(scores, self.quantile, self.gross_leverage)

        # Single central execution shift: decide on bar t, hold over t -> t+1.
        held = weights.shift(1).fillna(0.0)

        bar_return = panel.close.pct_change(fill_method=None).fillna(0.0)
        gross = (held * bar_return).sum(axis=1)

        spread = self._spread(held)
        turnover = held.diff().abs().fillna(held.abs())
        trade_cost = (turnover * (spread / 2.0)).sum(axis=1)

        swap = held.sum(axis=1) * self.costs.swap_per_bar  # ~0 for a neutral book

        net = gross - trade_cost + swap
        equity = (1.0 + net).cumprod()
        exposure = held.abs().sum(axis=1)  # gross exposure proxy for metrics

        metrics = compute_metrics(net, exposure, granularity=self.granularity)
        result = BacktestResult(
            returns=net,
            equity=equity,
            position=exposure,
            metrics=metrics,
            granularity=self.granularity,
        )
        return _reslice(result, start, self.granularity)


Factory = Callable[[Dict], Backtestable]


def portfolio_factory(
    signal_cls: Type[CrossSectionalSignal],
    panel: Panel,
    costs: Optional[CostModel] = None,
    granularity: str = "D",
    **construction,
) -> Factory:
    """Build a factory ``config -> PortfolioBacktest(signal_cls(**config), ...)``.

    ``construction`` (e.g. ``quantile``, ``gross_leverage``) is fixed across the
    sweep; ``config`` varies the signal's own parameters (e.g. ``lookback``).
    This is the cross-sectional analogue of
    :func:`~forex.backtest.evaluation.strategy_factory`, and it plugs straight
    into :func:`~forex.backtest.validation.walk_forward`.
    """

    def factory(config: Dict) -> Backtestable:
        return PortfolioBacktest(
            signal_cls(**config),
            panel,
            costs=costs,
            granularity=granularity,
            **construction,
        )

    return factory
