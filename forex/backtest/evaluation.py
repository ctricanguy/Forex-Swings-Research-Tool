"""The evaluation interface the validation harness depends on.

The harness must not care whether it is validating a single-pair strategy or a
cross-sectional multi-pair portfolio. Both produce the same thing: **a dated
net-return / position stream over a date window**. That contract is
:class:`Backtestable`.

* :class:`StrategyBacktest` adapts the existing per-pair ``Strategy`` + bars to
  the interface (implemented now).
* A ``PortfolioBacktest`` for the cross-sectional engine will implement the same
  interface later (step B) and plug into the harness with zero changes here.

Window semantics (important for honest walk-forward): ``run(start, end)`` scores
performance only on ``[start, end]``, but the strategy still sees **all history
up to each bar** for warm-up. We compute over the full data up to ``end`` and
then slice the result to ``[start, end]`` -- so a momentum lookback is not
artificially flattened at the start of every test window, and no data after
``end`` ever leaks in.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Type

import pandas as pd

from forex.backtest.engine import BacktestResult, CostModel, run_backtest
from forex.backtest.metrics import compute_metrics
from forex.strategies.base import Strategy


class Backtestable(ABC):
    """Anything that yields a dated return/position stream over a window.

    This is the unifying contract for the validation harness. Implementations
    bind together whatever they need (a strategy + data + costs, or a universe
    of pairs + a portfolio rule) and expose a single :meth:`run`.
    """

    #: Bar granularity, used for annualising metrics.
    granularity: str = "D"
    #: Stable identifier for reports.
    name: str = "backtestable"

    @abstractmethod
    def run(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> BacktestResult:
        """Return a :class:`BacktestResult` measured over ``[start, end]``.

        ``start``/``end`` are inclusive bounds on the evaluation window. History
        before ``start`` may be used for warm-up; data after ``end`` must not be
        used at all.
        """
        raise NotImplementedError


def _reslice(result: BacktestResult, start: Optional[datetime], granularity: str) -> BacktestResult:
    """Slice a full-period result to the evaluation window, rebasing equity."""
    if start is None:
        return result
    mask = result.returns.index >= start
    ret = result.returns[mask]
    held = result.position[mask]
    equity = (1.0 + ret).cumprod()
    metrics = compute_metrics(ret, held, granularity=granularity)
    return BacktestResult(
        returns=ret,
        equity=equity,
        position=held,
        metrics=metrics,
        granularity=granularity,
    )


class StrategyBacktest(Backtestable):
    """Per-pair adapter: one :class:`~forex.strategies.base.Strategy` over bars."""

    def __init__(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        costs: Optional[CostModel] = None,
        granularity: str = "D",
    ) -> None:
        self.strategy = strategy
        self.data = data.sort_index()
        self.costs = costs or CostModel()
        self.granularity = granularity
        self.name = getattr(strategy, "name", "strategy")

    def run(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> BacktestResult:
        data = self.data
        if end is not None:
            data = data[data.index <= end]  # never see beyond the window end
        result = run_backtest(
            self.strategy, data, costs=self.costs, granularity=self.granularity
        )
        return _reslice(result, start, self.granularity)


# -- helpers for parameter sweeps / walk-forward ---------------------------

#: A factory maps a parameter config (dict) to a Backtestable.
Factory = Callable[[Dict], Backtestable]


def expand_grid(grid: Dict[str, Sequence]) -> List[Dict]:
    """Cartesian product of a ``{param: [values]}`` grid into a list of configs."""
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def strategy_factory(
    strategy_cls: Type[Strategy],
    data: pd.DataFrame,
    costs: Optional[CostModel] = None,
    granularity: str = "D",
) -> Factory:
    """Build a factory ``config -> StrategyBacktest(strategy_cls(**config), ...)``."""

    def factory(config: Dict) -> Backtestable:
        return StrategyBacktest(strategy_cls(**config), data, costs, granularity)

    return factory
