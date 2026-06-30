#!/usr/bin/env python3
"""Validate a strategy: walk-forward refit + untouched holdout + deflated Sharpe.

Runs against the Backtestable interface, so the same command will work for the
cross-sectional portfolio engine once it lands -- only the factory changes.

Example
-------
    python scripts/run_validation.py --strategy tsmom --train 250 --test 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forex.backtest import (
    CostModel,
    StrategyBacktest,
    expand_grid,
    holdout_split,
    strategy_factory,
    validation_report,
    walk_forward,
)
from forex.data import SyntheticDataProvider
from forex.strategies import MovingAverageCross, TimeSeriesMomentum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="tsmom", choices=["tsmom", "ma_cross"])
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--granularity", default="D")
    parser.add_argument("--bars", type=int, default=1500)
    parser.add_argument("--train", type=int, default=250)
    parser.add_argument("--test", type=int, default=60)
    parser.add_argument("--anchored", action="store_true")
    parser.add_argument("--spread-fill", type=float, default=1.0)
    args = parser.parse_args()

    provider = SyntheticDataProvider()
    data = provider.get_candles(args.instrument, args.granularity, count=args.bars)
    costs = CostModel(spread_fill=args.spread_fill)

    if args.strategy == "tsmom":
        cls = TimeSeriesMomentum
        grid = {"lookback": [20, 40, 60, 90, 120], "deadband": [0.0, 0.005]}
    else:
        cls = MovingAverageCross
        grid = {"fast": [10, 20, 30], "slow": [50, 80, 120]}

    factory = strategy_factory(cls, data, costs, args.granularity)
    configs = expand_grid(grid)
    if cls is MovingAverageCross:
        configs = [c for c in configs if c["fast"] < c["slow"]]

    wf = walk_forward(
        factory,
        configs,
        train=args.train,
        test=args.test,
        anchored=args.anchored,
        granularity=args.granularity,
    )

    # Holdout: develop on the first 70%, judge the untouched last 30% using the
    # config the walk-forward picked most often.
    from collections import Counter

    picked = Counter(tuple(sorted(f.best_config.items())) for f in wf.folds).most_common(1)[0][0]
    chosen_config = dict(picked)
    chosen = StrategyBacktest(cls(**chosen_config), data, costs, args.granularity)
    holdout = holdout_split(chosen, in_sample=0.7)

    print(f"Strategy: {cls.__name__}  Instrument: {args.instrument}")
    print(f"Grid: {len(configs)} configs  |  most-picked OOS config: {chosen_config}\n")
    print(validation_report(wf, holdout))


if __name__ == "__main__":
    main()
