#!/usr/bin/env python3
"""Run a backtest from the command line.

Examples
--------
    # Time-series momentum on synthetic data (no keys needed)
    python scripts/run_backtest.py --strategy tsmom --lookback 60

    # Compare against a one-bar feature delay to check for lookahead
    python scripts/run_backtest.py --strategy tsmom --check-lookahead
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forex.backtest import CostModel, detect_lookahead, run_backtest
from forex.data import SyntheticDataProvider
from forex.strategies import Carry, MovingAverageCross, TimeSeriesMomentum


def build_strategy(args: argparse.Namespace):
    if args.strategy == "tsmom":
        return TimeSeriesMomentum(lookback=args.lookback)
    if args.strategy == "ma_cross":
        return MovingAverageCross(fast=args.fast, slow=args.slow)
    if args.strategy == "carry":
        return Carry()
    raise ValueError(f"unknown strategy: {args.strategy}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy", default="tsmom", choices=["tsmom", "ma_cross", "carry"]
    )
    parser.add_argument("--instrument", default="EUR_USD")
    parser.add_argument("--granularity", default="D")
    parser.add_argument("--bars", type=int, default=750)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--fast", type=int, default=20)
    parser.add_argument("--slow", type=int, default=50)
    parser.add_argument("--spread-fill", type=float, default=1.0)
    parser.add_argument("--check-lookahead", action="store_true")
    args = parser.parse_args()

    provider = SyntheticDataProvider(with_rate_diff=(args.strategy == "carry"))
    data = provider.get_candles(args.instrument, args.granularity, count=args.bars)
    strategy = build_strategy(args)
    costs = CostModel(spread_fill=args.spread_fill)

    result = run_backtest(strategy, data, costs=costs, granularity=args.granularity)
    print(f"Strategy: {strategy.name}  Instrument: {args.instrument}\n")
    print(result.summary())

    if args.check_lookahead:
        diag = detect_lookahead(strategy, data)
        print("\nLookahead check (truncation test):")
        print(f"  bars probed     : {int(diag['probe'])}")
        print(f"  leaking bars    : {int(diag['leaking_bars'])}")
        print(f"  max pos. diff   : {diag['max_position_diff']:.4f}")
        verdict = "PASS (point-in-time)" if diag["leaking_bars"] == 0 else "FAIL (future leak!)"
        print(f"  verdict         : {verdict}")


if __name__ == "__main__":
    main()
