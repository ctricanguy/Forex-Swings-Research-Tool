"""Validation harness (roadmap step A).

Everything here depends only on the :class:`~forex.backtest.evaluation.Backtestable`
interface and a ``Factory`` (config -> Backtestable). It never references the
per-pair structure directly, so the cross-sectional portfolio engine plugs in by
supplying its own factory -- no changes to this module.

What it provides:

* :func:`walk_forward` -- rolling (or anchored) re-fit: on each fold, pick the
  best config on the *train* window, then measure it out-of-sample on the *test*
  window. Concatenate the OOS windows into one honest, stitched return stream.
* :func:`holdout_split` -- carve an untouched out-of-sample tail you only look at
  once, to sanity-check the developed strategy.
* Deflated / Probabilistic Sharpe on the stitched OOS stream, deflated by the
  number of configurations tried.
* :func:`validation_report` -- a formatted summary + a graduation verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

from forex.backtest.engine import BacktestResult
from forex.backtest.evaluation import Backtestable, Factory
from forex.backtest.metrics import compute_metrics
from forex.backtest.statistics import (
    deflated_sharpe_ratio,
    per_period_sharpe,
    probabilistic_sharpe_ratio,
    trial_sharpe_std,
)


def sharpe_score(result: BacktestResult) -> float:
    """Default config-selection objective: annualised Sharpe on the train window."""
    return result.metrics.get("sharpe", float("nan"))


@dataclass(slots=True)
class Fold:
    index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_config: Dict
    train_score: float
    test_sharpe: float


@dataclass(slots=True)
class WalkForwardResult:
    oos_returns: pd.Series
    oos_equity: pd.Series
    oos_position: pd.Series
    metrics: Dict[str, float]
    folds: List[Fold]
    n_trials: int
    deflated_sharpe: float
    psr: float
    granularity: str = "D"
    #: Full-sample per-period Sharpe of every config tried (deflation input);
    #: exposed so a caller can pool trials ACROSS sleeves and re-deflate.
    trial_sharpes: List[float] = field(default_factory=list)
    #: What the current deflated_sharpe accounts for: "per-sleeve" (this grid
    #: only) or "project-wide" (set by redeflate()).
    deflation_scope: str = "per-sleeve"

    def redeflate(
        self, n_trials: int, sr_trials_std: float, scope: str = "project-wide"
    ) -> None:
        """Re-deflate against a wider search than this sleeve's own grid.

        Per-sleeve deflation answers "is the best config in this grid luck?".
        It is too generous when many sleeves are evaluated against the same
        data: the selection the DSR must guard is the *whole* search. Pass the
        total trial count and the pooled cross-trial Sharpe dispersion.
        """
        self.n_trials = int(n_trials)
        self.deflated_sharpe = deflated_sharpe_ratio(
            self.oos_returns, n_trials=self.n_trials, sr_trials_std=sr_trials_std
        )
        self.deflation_scope = scope

    def summary(self) -> str:
        m = self.metrics
        lines = [
            "Walk-forward (out-of-sample, stitched)",
            f"  folds            : {len(self.folds)}",
            f"  OOS bars         : {int(m.get('bars', 0))}",
            f"  OOS total return : {m.get('total_return', 0):+.2%}",
            f"  OOS ann. return  : {m.get('ann_return', 0):+.2%}",
            f"  OOS Sharpe       : {m.get('sharpe', 0):.2f}",
            f"  OOS max drawdown : {m.get('max_drawdown', 0):.2%}",
            f"  trials deflated  : {self.n_trials} ({self.deflation_scope})",
            f"  PSR (SR>0)       : {self.psr:.3f}",
            f"  Deflated Sharpe  : {self.deflated_sharpe:.3f}",
        ]
        return "\n".join(lines)


@dataclass(slots=True)
class HoldoutResult:
    in_sample: BacktestResult
    out_of_sample: BacktestResult
    cut: datetime

    def summary(self) -> str:
        return (
            "Holdout split\n"
            f"  cut date         : {self.cut:%Y-%m-%d}\n"
            f"  in-sample Sharpe : {self.in_sample.metrics.get('sharpe', 0):.2f}"
            f"  ({self.in_sample.metrics.get('total_return', 0):+.2%})\n"
            f"  out-of-sample SR : {self.out_of_sample.metrics.get('sharpe', 0):.2f}"
            f"  ({self.out_of_sample.metrics.get('total_return', 0):+.2%})"
        )


def walk_forward(
    factory: Factory,
    configs: Sequence[Dict],
    *,
    train: int,
    test: int,
    step: Optional[int] = None,
    anchored: bool = False,
    embargo: int = 0,
    score=sharpe_score,
    granularity: str = "D",
    n_trials: Optional[int] = None,
) -> WalkForwardResult:
    """Rolling-refit walk-forward over a parameter grid.

    Parameters
    ----------
    factory:
        Maps a config dict to a :class:`Backtestable`.
    configs:
        Candidate configs to choose among on each train window (e.g. from
        :func:`~forex.backtest.evaluation.expand_grid`).
    train, test:
        Window lengths in bars. ``step`` defaults to ``test`` (contiguous,
        non-overlapping OOS windows).
    anchored:
        If True, the train window always starts at the beginning (expanding);
        otherwise it rolls with a fixed length.
    embargo:
        Bars skipped between each ``train_end`` and ``test_start`` (a
        purge/embargo gap). **Size it to the label/holding horizon, not the
        feature lookback**: our label is the next-bar return, so a handful of
        bars is ample padding. Features looking back across the boundary
        (e.g. a 1000-bar momentum at ``test_start`` reading train-period
        prices) is NOT leakage in a forward-chaining split -- it is exactly
        what the deployed strategy does at every live rebalance, and no
        train-window return observation is reused in test scoring. The gap
        exists as cheap conservatism and to stay valid if a future strategy
        introduces multi-bar forward-looking labels.
    n_trials:
        Trials for the deflation. Defaults to ``len(configs)`` (this sleeve's
        grid). NOTE: that per-sleeve default is too generous when many sleeves
        are evaluated against the same data -- pool trials across sleeves and
        call :meth:`WalkForwardResult.redeflate` (or pass the project-wide
        count here) for an honest project-level deflation.
    """
    configs = list(configs)
    if not configs:
        raise ValueError("walk_forward needs at least one config")
    if embargo < 0:
        raise ValueError("embargo must be >= 0")
    step = step or test

    # Reference timeline from a full run of the first config.
    reference = factory(configs[0]).run()
    idx = reference.returns.index
    n = len(idx)
    if train + embargo + test > n:
        raise ValueError(
            f"not enough bars ({n}) for train={train} + embargo={embargo} + test={test}"
        )

    folds: List[Fold] = []
    ret_parts: List[pd.Series] = []
    pos_parts: List[pd.Series] = []

    pos = train
    fold_i = 0
    while pos + embargo + test <= n:
        train_lo = 0 if anchored else pos - train
        train_start, train_end = idx[train_lo], idx[pos - 1]
        test_start = idx[pos + embargo]
        test_end = idx[pos + embargo + test - 1]

        best_score = None
        best_config = configs[0]
        for cfg in configs:
            s = score(factory(cfg).run(train_start, train_end))
            if s is not None and not pd.isna(s) and (best_score is None or s > best_score):
                best_score, best_config = s, cfg

        test_result = factory(best_config).run(test_start, test_end)
        ret_parts.append(test_result.returns)
        pos_parts.append(test_result.position)
        folds.append(
            Fold(
                index=fold_i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_config=best_config,
                train_score=float(best_score) if best_score is not None else float("nan"),
                test_sharpe=test_result.metrics.get("sharpe", float("nan")),
            )
        )
        fold_i += 1
        pos += step

    oos_returns = pd.concat(ret_parts)
    oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")].sort_index()
    oos_position = pd.concat(pos_parts)
    oos_position = oos_position[~oos_position.index.duplicated(keep="first")].sort_index()
    oos_equity = (1.0 + oos_returns).cumprod()
    metrics = compute_metrics(oos_returns, oos_position, granularity=granularity)

    # Deflation: dispersion of per-period Sharpes across the full-sample trials.
    trial_returns = [factory(cfg).run().returns for cfg in configs]
    trial_sharpes = [per_period_sharpe(r) for r in trial_returns]
    sr_std = trial_sharpe_std(trial_returns)
    trials = n_trials if n_trials is not None else len(configs)
    dsr = deflated_sharpe_ratio(oos_returns, n_trials=trials, sr_trials_std=sr_std)
    psr = probabilistic_sharpe_ratio(oos_returns, benchmark_sr=0.0)

    return WalkForwardResult(
        oos_returns=oos_returns,
        oos_equity=oos_equity,
        oos_position=oos_position,
        metrics=metrics,
        folds=folds,
        n_trials=trials,
        deflated_sharpe=dsr,
        psr=psr,
        granularity=granularity,
        trial_sharpes=trial_sharpes,
        deflation_scope="per-sleeve" if n_trials is None else "caller-specified",
    )


def holdout_split(
    backtestable: Backtestable,
    *,
    in_sample: Union[float, str, datetime] = 0.7,
) -> HoldoutResult:
    """Split a fixed strategy into an in-sample part and an untouched OOS tail.

    ``in_sample`` is either a fraction (0..1) of the timeline used for the
    in-sample part, or a cut date (everything from the cut onward is OOS).
    """
    reference = backtestable.run()
    idx = reference.returns.index
    n = len(idx)
    if isinstance(in_sample, float):
        k = max(1, min(n - 1, int(n * in_sample)))
    else:
        k = int(idx.searchsorted(pd.Timestamp(in_sample, tz=idx.tz)))
        k = max(1, min(n - 1, k))
    cut = idx[k]

    is_result = backtestable.run(idx[0], idx[k - 1])
    oos_result = backtestable.run(idx[k], idx[-1])
    return HoldoutResult(in_sample=is_result, out_of_sample=oos_result, cut=cut)


def graduation_verdict(wf: WalkForwardResult, holdout: Optional[HoldoutResult] = None) -> str:
    """A blunt go/no-go per the doc: a strategy graduates only if the deflated
    Sharpe stays convincingly positive and OOS doesn't contradict in-sample."""
    dsr = wf.deflated_sharpe
    if pd.isna(dsr) or dsr < 0.5:
        return "FAIL - deflated Sharpe not credibly positive; treat as a learning tool, do not risk capital."
    consistent = True
    if holdout is not None:
        is_sr = holdout.in_sample.metrics.get("sharpe", 0.0)
        oos_sr = holdout.out_of_sample.metrics.get("sharpe", 0.0)
        consistent = (is_sr > 0) == (oos_sr > 0)
    if dsr >= 0.95 and consistent:
        return "PASS - deflated Sharpe strong and holdout consistent; candidate for paper trading."
    return "WEAK - positive but not strong after deflation; needs more data / out-of-sample before any capital."


def validation_report(
    wf: WalkForwardResult,
    holdout: Optional[HoldoutResult] = None,
) -> str:
    parts = [wf.summary()]
    if holdout is not None:
        parts.append("")
        parts.append(holdout.summary())
    parts.append("")
    parts.append(f"Verdict: {graduation_verdict(wf, holdout)}")
    return "\n".join(parts)
