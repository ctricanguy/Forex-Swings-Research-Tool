#!/usr/bin/env python3
"""Backtest and validate a cross-sectional long/short factor portfolio.

Builds a universe panel, ranks pairs by a factor (momentum or carry) into a
dollar-neutral long/short book, and validates it through the SAME walk-forward /
holdout / deflated-Sharpe harness used for single-pair strategies -- the only
difference is the factory.

Example
-------
    python scripts/run_portfolio.py --signal xsmom --train 250 --test 60
    python scripts/run_portfolio.py --signal xscarry
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forex.backtest import (
    CostModel,
    PortfolioBacktest,
    expand_grid,
    holdout_split,
    portfolio_factory,
    validation_report,
    walk_forward,
)
from forex.data import Panel, SyntheticDataProvider
from forex.strategies import XSCarry, XSMomentum

UNIVERSE = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD",
    "USD_CAD", "NZD_USD", "USD_CHF", "EUR_GBP",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", default="xsmom", choices=["xsmom", "xscarry"])
    parser.add_argument("--granularity", default="D")
    parser.add_argument("--bars", type=int, default=1500)
    parser.add_argument("--train", type=int, default=250)
    parser.add_argument("--test", type=int, default=60)
    parser.add_argument("--quantile", type=float, default=1 / 3)
    parser.add_argument("--anchored", action="store_true")
    parser.add_argument("--spread-fill", type=float, default=1.0)
    args = parser.parse_args()

    needs_rates = args.signal == "xscarry"
    provider = SyntheticDataProvider(with_rate_diff=needs_rates)
    panel = Panel.from_provider(provider, UNIVERSE, args.granularity, count=args.bars)
    costs = CostModel(spread_fill=args.spread_fill)

    if args.signal == "xsmom":
        cls = XSMomentum
        configs = expand_grid({"lookback": [20, 40, 60, 90, 120]})
    else:
        cls = XSCarry
        configs = expand_grid({})  # carry has no parameters to sweep

    factory = portfolio_factory(
        cls, panel, costs, args.granularity, quantile=args.quantile
    )

    # Single full-sample backtest for context.
    full = factory(configs[0]).run()
    print(f"Signal: {cls.__name__}  Universe: {len(UNIVERSE)} pairs  "
          f"quantile: {args.quantile:.2f}\n")
    print(full.summary())
    print()

    # Validate through the shared harness.
    wf = walk_forward(
        factory, configs, train=args.train, test=args.test,
        anchored=args.anchored, granularity=args.granularity,
    )

    if wf.folds:
        picked = Counter(
            tuple(sorted(f.best_config.items())) for f in wf.folds
        ).most_common(1)[0][0]
        chosen = PortfolioBacktest(
            cls(**dict(picked)), panel, costs=costs,
            granularity=args.granularity, quantile=args.quantile,
        )
        holdout = holdout_split(chosen, in_sample=0.7)
    else:
        holdout = None

    print(validation_report(wf, holdout))


if __name__ == "__main__":
    main()
