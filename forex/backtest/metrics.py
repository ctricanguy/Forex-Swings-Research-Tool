"""Performance metrics.

Kept separate so both the backtester and any walk-forward / validation code can
share one definition of Sharpe, drawdown, etc.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

# Approximate number of bars per year by granularity, for annualisation.
# FX trades ~24h on ~252 weekday sessions.
PERIODS_PER_YEAR: Dict[str, float] = {
    "M1": 252 * 24 * 60,
    "M5": 252 * 24 * 12,
    "M15": 252 * 24 * 4,
    "M30": 252 * 24 * 2,
    "H1": 252 * 24,
    "H4": 252 * 6,
    "D": 252,
    "W": 52,
}


def periods_per_year(granularity: str) -> float:
    return PERIODS_PER_YEAR.get(granularity, 252.0)


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough fractional drop of an equity curve (<= 0)."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def compute_metrics(
    net_returns: pd.Series,
    traded_position: pd.Series,
    granularity: str = "D",
) -> Dict[str, float]:
    """Summarise a per-bar net-return series into standard stats."""
    ppy = periods_per_year(granularity)
    net_returns = net_returns.fillna(0.0)
    equity = (1.0 + net_returns).cumprod()

    n = len(net_returns)
    total_return = float(equity.iloc[-1] - 1.0) if n else 0.0
    mean = float(net_returns.mean()) if n else 0.0
    std = float(net_returns.std(ddof=1)) if n > 1 else 0.0

    ann_return = (1.0 + mean) ** ppy - 1.0 if n else 0.0
    ann_vol = std * np.sqrt(ppy)
    sharpe = (mean / std * np.sqrt(ppy)) if std > 0 else 0.0

    # Trade-level stats: a "trade" is a change in the traded position.
    trades = traded_position.diff().fillna(traded_position).abs()
    num_trades = int((trades > 1e-12).sum())
    exposure = float((traded_position.abs() > 1e-12).mean()) if n else 0.0

    wins = net_returns[net_returns > 0]
    losses = net_returns[net_returns < 0]
    win_rate = float(len(wins) / n) if n else 0.0
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(equity),
        "num_trades": num_trades,
        "exposure": exposure,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "bars": float(n),
    }
