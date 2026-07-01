"""Risk layer -- where long-run durability actually lives.

Per the design doc: risk management, not signal cleverness, is what keeps a
modest-edge system alive through the multi-year drawdowns even genuinely
positive-Sharpe strategies suffer. This module applies it at the
:class:`~forex.backtest.evaluation.Backtestable` level, so a per-pair strategy
and a cross-sectional portfolio get identical treatment -- and a risk-overlaid
strategy is itself a ``Backtestable``, validating through the same
walk-forward / deflated-Sharpe harness.

Two controls, both strictly point-in-time:

* **Volatility targeting** -- scale exposure by ``target_vol / trailing realized
  vol`` (computed through bar ``t-1``, capped at ``max_leverage``). Stabilises
  risk across regimes; a fixed-position strategy doubles its risk exactly when
  markets get violent, which is when carry crashes.
* **Max-drawdown kill switch** -- monitor the overlaid equity curve; when its
  drawdown breaches ``max_drawdown``, go flat. Re-enter once the *underlying*
  strategy has recovered ``resume_drawdown`` above its post-kill trough
  (default: half the kill threshold), so a broken strategy stays off but a
  drawdown that mean-reverts resumes.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

from forex.backtest.engine import BacktestResult
from forex.backtest.evaluation import Backtestable, Factory, _reslice
from forex.backtest.metrics import compute_metrics, periods_per_year


def fixed_fractional_units(
    equity: float, risk_fraction: float, entry: float, stop: float
) -> float:
    """Classic fixed-fractional sizing: units such that hitting the stop loses
    ``risk_fraction`` of equity (e.g. 0.01 = risk 1% per trade)."""
    distance = abs(entry - stop)
    if distance <= 0:
        raise ValueError("entry and stop must differ")
    return equity * risk_fraction / distance


class RiskOverlay(Backtestable):
    """Wrap any :class:`Backtestable` with vol targeting and a kill switch."""

    def __init__(
        self,
        base: Backtestable,
        *,
        target_ann_vol: Optional[float] = None,
        vol_window: int = 60,
        max_leverage: float = 3.0,
        max_drawdown: Optional[float] = None,
        resume_drawdown: Optional[float] = None,
    ) -> None:
        self.base = base
        self.granularity = base.granularity
        self.name = f"risk[{base.name}]"
        self.target_ann_vol = target_ann_vol
        self.vol_window = vol_window
        self.max_leverage = max_leverage
        self.max_drawdown = max_drawdown
        self.resume_drawdown = (
            resume_drawdown
            if resume_drawdown is not None
            else (max_drawdown / 2.0 if max_drawdown else None)
        )
        # State after the most recent run(); the live runner reads these to
        # apply the same scaling to the next real position.
        self.last_scale: float = 1.0
        self.last_gate: float = 1.0

    def _vol_scale(self, returns: pd.Series) -> pd.Series:
        if not self.target_ann_vol:
            return pd.Series(1.0, index=returns.index)
        ppy = periods_per_year(self.granularity)
        realized = returns.rolling(self.vol_window).std(ddof=1) * math.sqrt(ppy)
        # Scale for bar t uses vol through t-1 only.
        scale = (self.target_ann_vol / realized).shift(1)
        return scale.clip(lower=0.0, upper=self.max_leverage).fillna(1.0)

    def _gate(self, returns: pd.Series, scale: pd.Series) -> np.ndarray:
        gate = np.ones(len(returns))
        if not self.max_drawdown:
            return gate
        r = returns.to_numpy()
        s = scale.to_numpy()
        overlay_eq = overlay_peak = 1.0
        under_eq = 1.0
        trough = None  # underlying equity trough since the kill fired
        g = 1.0
        for i in range(len(r)):
            gate[i] = g
            overlay_eq *= 1.0 + r[i] * s[i] * g
            overlay_peak = max(overlay_peak, overlay_eq)
            under_eq *= 1.0 + r[i]
            if g == 1.0:
                if overlay_eq / overlay_peak - 1.0 <= -self.max_drawdown:
                    g = 0.0
                    trough = under_eq
            else:
                trough = min(trough, under_eq)
                if under_eq >= trough * (1.0 + self.resume_drawdown):
                    g = 1.0
                    # Reset the peak so a resumed book isn't instantly re-killed
                    # by the drawdown it already took.
                    overlay_peak = overlay_eq
        return gate

    def run(self, start=None, end=None) -> BacktestResult:
        base_result = self.base.run(None, end)  # full history for warm-up
        returns = base_result.returns.fillna(0.0)

        scale = self._vol_scale(returns)
        gate = pd.Series(self._gate(returns, scale), index=returns.index)

        adj_returns = returns * scale * gate
        adj_position = base_result.position * scale * gate
        equity = (1.0 + adj_returns).cumprod()

        self.last_scale = float(scale.iloc[-1]) if len(scale) else 1.0
        self.last_gate = float(gate.iloc[-1]) if len(gate) else 1.0

        metrics = compute_metrics(adj_returns, adj_position, granularity=self.granularity)
        result = BacktestResult(
            returns=adj_returns,
            equity=equity,
            position=adj_position,
            metrics=metrics,
            granularity=self.granularity,
        )
        return _reslice(result, start, self.granularity)


def risk_overlay_factory(base_factory: Factory, **risk_kwargs) -> Factory:
    """Wrap a factory so every config is risk-overlaid -- the harness then
    validates the strategy *as it would actually be run*, risk controls and all."""

    def factory(config: Dict) -> Backtestable:
        return RiskOverlay(base_factory(config), **risk_kwargs)

    return factory
