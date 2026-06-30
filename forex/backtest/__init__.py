"""Backtesting & validation layer.

A deliberately small, auditable, vectorised engine plus a validation harness.
Both import strategies from :mod:`forex.strategies` -- the exact same code the
live runner uses.

The harness depends only on the :class:`~forex.backtest.evaluation.Backtestable`
interface (a dated return/position stream over a window), so a single-pair
strategy and a future cross-sectional portfolio are validated by identical code.
"""

from forex.backtest.engine import (
    BacktestResult,
    CostModel,
    detect_lookahead,
    run_backtest,
)
from forex.backtest.evaluation import (
    Backtestable,
    Factory,
    StrategyBacktest,
    expand_grid,
    strategy_factory,
)
from forex.backtest.metrics import compute_metrics, periods_per_year
from forex.backtest.portfolio import (
    PortfolioBacktest,
    portfolio_factory,
    rank_weights,
)
from forex.backtest.statistics import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    per_period_sharpe,
    probabilistic_sharpe_ratio,
)
from forex.backtest.validation import (
    Fold,
    HoldoutResult,
    WalkForwardResult,
    graduation_verdict,
    holdout_split,
    validation_report,
    walk_forward,
)

__all__ = [
    # engine
    "BacktestResult",
    "CostModel",
    "detect_lookahead",
    "run_backtest",
    # evaluation interface
    "Backtestable",
    "Factory",
    "StrategyBacktest",
    "expand_grid",
    "strategy_factory",
    # cross-sectional portfolio engine
    "PortfolioBacktest",
    "portfolio_factory",
    "rank_weights",
    # metrics
    "compute_metrics",
    "periods_per_year",
    # statistics
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "per_period_sharpe",
    "probabilistic_sharpe_ratio",
    # validation harness
    "Fold",
    "HoldoutResult",
    "WalkForwardResult",
    "graduation_verdict",
    "holdout_split",
    "validation_report",
    "walk_forward",
]
