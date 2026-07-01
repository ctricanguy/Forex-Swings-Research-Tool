#!/usr/bin/env python3
"""Full validation suite on REAL OANDA data. Run this on your own machine.

Pulls ~10 years of daily candles for an 8-pair universe, attaches real FRED
rate differentials (keyless), and runs every factor sleeve through walk-forward
+ holdout + deflated Sharpe -- plus a risk-overlaid ensemble, i.e. the sleeve as
it would actually be deployed. Archives the report under docs/validation/ so
the pre-live evidence is preserved (see docs/SERVICE.md).

    export OANDA_ACCESS_TOKEN=...      # practice token is fine
    python scripts/validate_real.py

Costs are modelled at the full quoted bid/ask spread (conservative).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forex.backtest import (
    CostModel,
    expand_grid,
    holdout_split,
    portfolio_factory,
    strategy_factory,
    validation_report,
    walk_forward,
)
from forex.data import Panel
from forex.data.oanda import OandaDataProvider
from forex.risk import risk_overlay_factory
from forex.strategies import (
    TimeSeriesMomentum,
    XSCarry,
    XSEnsemble,
    XSMomentum,
    XSValue,
)

UNIVERSE = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD",
    "USD_CAD", "NZD_USD", "USD_CHF", "EUR_GBP",
]


def run_sleeve(title, factory, configs, train, test):
    lines = [f"## {title}", ""]
    wf = walk_forward(factory, configs, train=train, test=test)
    holdout = None
    if wf.folds:
        picked = Counter(
            tuple(sorted(f.best_config.items())) for f in wf.folds
        ).most_common(1)[0][0]
        lines.append(f"most-picked OOS config: {dict(picked)}")
        holdout = holdout_split(factory(dict(picked)), in_sample=0.7)
    lines += ["", "```", validation_report(wf, holdout), "```", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=int, default=2600, help="~10y of daily bars")
    parser.add_argument("--train", type=int, default=750, help="~3y train window")
    parser.add_argument("--test", type=int, default=125, help="~6mo test window")
    parser.add_argument("--out", default="docs/validation")
    args = parser.parse_args()

    token = os.getenv("OANDA_ACCESS_TOKEN")
    if not token:
        raise SystemExit("OANDA_ACCESS_TOKEN is not set")
    provider = OandaDataProvider(token, os.getenv("OANDA_ENVIRONMENT", "practice"))

    print(f"Fetching {args.bars} daily bars for {len(UNIVERSE)} pairs from OANDA ...")
    frames = {}
    for instrument in UNIVERSE:
        df = provider.get_candles(instrument, "D", count=args.bars)
        frames[instrument] = df
        print(f"  {instrument}: {len(df)} bars  {df.index[0]:%Y-%m-%d} .. {df.index[-1]:%Y-%m-%d}")

    have_rates = False
    try:
        from forex.data.rates import FredRates

        print("Attaching FRED rate differentials ...")
        frames = FredRates().attach(frames)
        have_rates = True
    except Exception as exc:  # noqa: BLE001 - report and continue price-only
        print(f"WARNING: FRED rates unavailable ({exc}); carry/ensemble skipped.")

    panel = Panel(frames)
    costs = CostModel(spread_fill=1.0)  # full quoted spread, conservative
    train, test = args.train, args.test

    sections = [
        f"# Real-data validation — {date.today():%Y-%m-%d}",
        "",
        f"Universe: {', '.join(UNIVERSE)}  |  daily bars: ~{args.bars}  |  "
        f"walk-forward train/test: {train}/{test} bars  |  costs: full quoted spread"
        + ("  |  FRED rates attached" if have_rates else "  |  NO rates (price-only)"),
        "",
    ]

    def pf(cls, **construction):
        return portfolio_factory(cls, panel, costs, "D", quantile=1 / 3, **construction)

    sections.append(run_sleeve(
        "Cross-sectional momentum", pf(XSMomentum),
        expand_grid({"lookback": [20, 40, 60, 120, 250]}), train, test,
    ))
    sections.append(run_sleeve(
        "Cross-sectional value (long-horizon reversal)", pf(XSValue),
        expand_grid({"lookback": [750, 1000, 1250]}), train, test,
    ))
    if have_rates:
        sections.append(run_sleeve(
            "Cross-sectional carry (real rate differentials)", pf(XSCarry),
            expand_grid({}), train, test,
        ))

        def ensemble_cls(**_):
            return XSEnsemble([XSMomentum(60), XSCarry(), XSValue(1000)])

        sections.append(run_sleeve(
            "Diversified sleeve: momentum + carry + value", pf(ensemble_cls),
            expand_grid({}), train, test,
        ))
        sections.append(run_sleeve(
            "Deployed configuration: sleeve + risk overlay (10% vol target, 15% kill)",
            risk_overlay_factory(
                pf(ensemble_cls), target_ann_vol=0.10, max_drawdown=0.15
            ),
            expand_grid({}), train, test,
        ))

    sections.append(run_sleeve(
        "Baseline: per-pair time-series momentum on EUR_USD",
        strategy_factory(TimeSeriesMomentum, frames["EUR_USD"], costs, "D"),
        expand_grid({"lookback": [60, 120, 250]}), train, test,
    ))

    report = "\n".join(sections)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date.today():%Y-%m-%d}-real.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\n{report}\n\nReport archived to {out_path}")


if __name__ == "__main__":
    main()
