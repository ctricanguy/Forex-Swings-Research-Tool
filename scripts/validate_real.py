#!/usr/bin/env python3
"""Full validation suite on REAL OANDA data. Run this on your own machine.

Pulls ~10 years of daily candles for a 16-pair universe (majors + liquid
crosses -- cross-sectional factors need breadth), attaches real FRED rate
differentials, and runs every factor sleeve through walk-forward + holdout +
deflated Sharpe -- including regime-gated variants (vol-gated carry,
trend-gated momentum) and the risk-overlaid deployed configuration. Archives
the report under docs/validation/ so the pre-live evidence is preserved.

    export OANDA_ACCESS_TOKEN=...      # practice token is fine
    python scripts/validate_real.py

Costs are modelled at the full quoted bid/ask spread (conservative).

HONESTY NOTE: every sleeve in this report is one more trial against the same
data. Per-sleeve deflated Sharpe accounts for its own parameter grid, not for
sleeve-shopping across the report -- so if you pick the best-looking section,
apply extra skepticism, and prefer the sleeve that was the *prior* hypothesis
(the diversified ensemble) over whatever happens to shine.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from datetime import datetime
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
    TrendGate,
    VolGate,
    XSCarry,
    XSEnsemble,
    XSMomentum,
    XSValue,
)

# Majors + liquid crosses. Cross-sectional ranking with 16 names has real
# granularity; with 8 it starves. All currencies are covered by the FRED map.
UNIVERSE = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD",
    "USD_CAD", "NZD_USD", "USD_CHF", "EUR_GBP",
    "EUR_JPY", "GBP_JPY", "AUD_JPY", "CAD_JPY",
    "EUR_CHF", "EUR_AUD", "GBP_CHF", "NZD_JPY",
]

# Cost-aware momentum lookbacks: the 20-bar config churns too much to survive
# the spread (the first real-data run picked it in-sample and lost OOS).
MOMENTUM_GRID = {"lookback": [60, 120, 250]}
VALUE_GRID = {"lookback": [750, 1000, 1250]}


def run_sleeve(title, factory, configs, train, test):
    lines = [f"## {title}", ""]
    wf = walk_forward(factory, configs, train=train, test=test)
    holdout = None
    if wf.folds:
        picked = Counter(
            tuple(sorted(f.best_config.items())) for f in wf.folds
        ).most_common(1)[0][0]
        if picked:
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

        print("Attaching FRED rate differentials (patient fetch, retries) ...")
        rates = FredRates(start="2014-01-01", timeout=120, retries=4)
        frames = rates.attach(frames)
        have_rates = True
    except Exception as exc:  # noqa: BLE001 - report and continue price-only
        print(f"WARNING: FRED rates unavailable ({exc}); carry/ensemble limited.")

    panel = Panel(frames)
    costs = CostModel(spread_fill=1.0)  # full quoted spread, conservative
    train, test = args.train, args.test

    sections = [
        f"# Real-data validation — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"Universe: {len(UNIVERSE)} pairs  |  daily bars: ~{args.bars}  |  "
        f"walk-forward train/test: {train}/{test} bars  |  costs: full quoted spread"
        + ("  |  FRED rates attached" if have_rates else "  |  NO rates (price-only)"),
        "",
    ]

    def pf(cls, **construction):
        return portfolio_factory(cls, panel, costs, "D", quantile=1 / 3, **construction)

    sleeves = []

    sleeves.append((
        "Cross-sectional momentum", pf(XSMomentum), expand_grid(MOMENTUM_GRID),
    ))

    def gated_momentum(**config):
        return TrendGate(XSMomentum(**config))

    sleeves.append((
        "Cross-sectional momentum, trend-gated (ER)", pf(gated_momentum),
        expand_grid(MOMENTUM_GRID),
    ))
    sleeves.append((
        "Cross-sectional value (long-horizon reversal)", pf(XSValue),
        expand_grid(VALUE_GRID),
    ))

    if have_rates:
        sleeves.append((
            "Cross-sectional carry (real rate differentials)", pf(XSCarry),
            expand_grid({}),
        ))

        def gated_carry(**_):
            return VolGate(XSCarry())

        sleeves.append((
            "Cross-sectional carry, vol-gated (flat when stressed)", pf(gated_carry),
            expand_grid({}),
        ))

        def ensemble_cls(**_):
            return XSEnsemble(
                [TrendGate(XSMomentum(120)), VolGate(XSCarry()), XSValue(1000)],
                require_all=False,
            )

        sleeves.append((
            "Diversified sleeve: gated momentum + gated carry + value",
            pf(ensemble_cls), expand_grid({}),
        ))
        sleeves.append((
            "DEPLOYED CONFIG: sleeve + risk overlay (10% vol target, 15% kill)",
            risk_overlay_factory(
                pf(ensemble_cls), target_ann_vol=0.10, max_drawdown=0.15
            ),
            expand_grid({}),
        ))

    sleeves.append((
        "Baseline: per-pair time-series momentum on EUR_USD",
        strategy_factory(TimeSeriesMomentum, frames["EUR_USD"], costs, "D"),
        expand_grid({"lookback": [60, 120, 250]}),
    ))

    sections.append(
        f"Sleeves evaluated in this report: {len(sleeves)} — per-sleeve deflated "
        "Sharpe does NOT account for choosing the best sleeve afterwards.\n"
    )
    for title, factory, configs in sleeves:
        print(f"validating: {title} ...")
        sections.append(run_sleeve(title, factory, configs, train, test))

    report = "\n".join(sections)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{datetime.now():%Y-%m-%d-%H%M}-real.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\n{report}\n\nReport archived to {out_path}")


if __name__ == "__main__":
    main()
