"""Backtesting layer.

A deliberately small, auditable, vectorised engine. It imports strategies from
:mod:`forex.strategies` -- the exact same code the live runner uses -- and adds
the two things a strategy must never do itself: the one-bar execution shift and
realistic cost accounting (spread + swap).
"""

from forex.backtest.engine import (
    BacktestResult,
    CostModel,
    detect_lookahead,
    run_backtest,
)
from forex.backtest.metrics import compute_metrics

__all__ = [
    "BacktestResult",
    "CostModel",
    "detect_lookahead",
    "run_backtest",
    "compute_metrics",
]
