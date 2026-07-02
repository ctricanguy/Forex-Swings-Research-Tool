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

import pandas as pd

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
from forex.data.storage import Storage
from forex.risk import risk_overlay_factory
from forex.strategies import (
    EventGate,
    TimeSeriesMomentum,
    TrendGate,
    VolGate,
    XSCarry,
    XSEnsemble,
    XSMomentum,
    XSPositioning,
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


def run_sleeve(title, factory, configs, train, test, embargo):
    """Run one sleeve; rendering happens later, after project-wide deflation."""
    wf = walk_forward(factory, configs, train=train, test=test, embargo=embargo)
    picked_line = None
    holdout = None
    if wf.folds:
        picked = Counter(
            tuple(sorted(f.best_config.items())) for f in wf.folds
        ).most_common(1)[0][0]
        if picked:
            picked_line = f"most-picked OOS config: {dict(picked)}"
        holdout = holdout_split(factory(dict(picked)), in_sample=0.7)
    return {
        "title": title,
        "wf": wf,
        "holdout": holdout,
        "picked_line": picked_line,
        # Grid-only deflation, captured before the project-wide re-deflation.
        "per_sleeve_dsr": wf.deflated_sharpe,
        "per_sleeve_trials": wf.n_trials,
    }


def render_sleeve(result):
    lines = [f"## {result['title']}", ""]
    if result["picked_line"]:
        lines.append(result["picked_line"])
    lines.append(
        f"per-sleeve DSR (grid of {result['per_sleeve_trials']} only, too "
        f"generous): {result['per_sleeve_dsr']:.3f}"
    )
    lines += ["", "```", validation_report(result["wf"], result["holdout"]), "```", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=int, default=2600, help="~10y of daily bars")
    parser.add_argument("--train", type=int, default=750, help="~3y train window")
    parser.add_argument("--test", type=int, default=125, help="~6mo test window")
    parser.add_argument("--out", default="docs/validation")
    parser.add_argument("--db", default="forex.db",
                        help="SQLite db holding positioning snapshots (fetch_positioning.py)")
    parser.add_argument("--embargo", type=int, default=5,
                        help="bars purged between train_end and test_start; sized to the "
                             "1-bar label horizon plus buffer (NOT the feature lookback -- "
                             "see walk_forward docstring)")
    parser.add_argument("--prior-trials", type=int, default=23,
                        help="configs already burned against this data in previous rounds "
                             "(round 1: 11, round 2: 12); added to the project-wide deflation. "
                             "NOTE: when RE-running configs from a previous round (regression "
                             "runs), count only the OTHER rounds here -- re-evaluating the same "
                             "configs is not new trials.")
    parser.add_argument("--price-only", action="store_true",
                        help="skip FRED rates, positioning, and event calendars entirely -- "
                             "price-only sleeves. For regression runs of the harness itself.")
    parser.add_argument("--no-value-robustness", action="store_true",
                        help="skip the extra 8-major value sleeve (exact round-2 sleeve parity)")
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
    if args.price_only:
        print("--price-only: skipping FRED rates (carry/ensemble sleeves excluded).")
    else:
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

    mode = "PRICE-ONLY REGRESSION RUN" if args.price_only else (
        "FRED rates attached" if have_rates else "NO rates (price-only fallback)"
    )
    sections = [
        f"# Real-data validation — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"Universe: {len(UNIVERSE)} pairs  |  daily bars: ~{args.bars}  |  "
        f"walk-forward train/test: {train}/{test} bars, embargo {args.embargo}  |  "
        f"costs: full quoted spread  |  {mode}",
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
    # Value on both universes: round two showed value's edge did not survive
    # widening 8 -> 16 pairs, so universe-robustness is now a standing check.
    sleeves.append((
        "Cross-sectional value (16 pairs)", pf(XSValue), expand_grid(VALUE_GRID),
    ))
    if not args.no_value_robustness:
        majors = UNIVERSE[:8]
        majors_panel = Panel({k: v for k, v in frames.items() if k in majors})
        sleeves.append((
            "Cross-sectional value (8 majors only — robustness check)",
            portfolio_factory(XSValue, majors_panel, costs, "D", quantile=1 / 3),
            expand_grid(VALUE_GRID),
        ))

    # Retail-positioning contrarian sleeve, if snapshots have been fetched
    # (scripts/fetch_positioning.py). Validated only on actual coverage.
    # Excluded by --price-only (non-price source; regression runs).
    if not args.price_only:
        try:
            from forex.data.positioning import attach_positioning

            with Storage(args.db) as storage:
                pos_frames = attach_positioning(storage, frames)
            coverage = pd.DataFrame(
                {k: v["net_long_pct"].notna() for k, v in pos_frames.items()}
            )
            enough = coverage.sum(axis=1) >= max(8, len(UNIVERSE) // 2)
            covered_bars = int(enough.sum())
            if covered_bars >= train + test:
                start_ts = enough[enough].index[0]
                pos_panel = Panel(
                    {k: v[v.index >= start_ts] for k, v in pos_frames.items()}
                )
                sections.append(
                    f"Positioning coverage: {covered_bars} bars from {start_ts:%Y-%m-%d}.\n"
                )
                sleeves.append((
                    "Retail positioning, contrarian (OANDA position book)",
                    portfolio_factory(XSPositioning, pos_panel, costs, "D", quantile=1 / 3),
                    expand_grid({}),
                ))
            else:
                print(f"positioning: only {covered_bars} covered bars -- sleeve skipped "
                      "(run scripts/fetch_positioning.py)")
        except Exception as exc:  # noqa: BLE001
            print(f"positioning unavailable ({exc}); sleeve skipped")

    # Scheduled-event calendar, if fetched (scripts/fetch_events.py).
    # Excluded by --price-only (non-price source; regression runs).
    event_dates = None
    events_dir = Path("data/events")
    if not args.price_only and events_dir.is_dir() and any(events_dir.glob("*.csv")):
        from forex.data.events import load_event_dates

        event_dates = load_event_dates(events_dir)
        print(f"event calendar: {len(event_dates)} dates loaded from {events_dir}")

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

        if event_dates is not None:

            def event_carry(**_):
                return EventGate(VolGate(XSCarry()), event_dates, before=1, after=1)

            sleeves.append((
                "Carry, vol-gated + event-avoiding (flat around CPI/NFP/FOMC)",
                pf(event_carry), expand_grid({}),
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

    # Pass 1: run every sleeve, collecting results and their trial Sharpes.
    results = []
    for title, factory, configs in sleeves:
        print(f"validating: {title} ...")
        results.append(run_sleeve(title, factory, configs, train, test, args.embargo))

    # Pass 2: project-wide deflation. The selection the DSR must guard is the
    # WHOLE search -- every config of every sleeve, this round and previous
    # rounds -- not each sleeve's grid in isolation.
    all_sharpes = pd.Series(
        [s for r in results for s in r["wf"].trial_sharpes]
    ).dropna()
    trials_this_report = sum(len(r["wf"].trial_sharpes) for r in results)
    total_trials = trials_this_report + args.prior_trials
    pooled_std = float(all_sharpes.std(ddof=1)) if len(all_sharpes) > 1 else float("nan")
    if pooled_std == pooled_std:  # not NaN
        for r in results:
            r["wf"].redeflate(total_trials, pooled_std, scope="project-wide")

    sections.append(
        f"Deflation: project-wide — {trials_this_report} trials this report "
        f"+ {args.prior_trials} prior (rounds 1–2) = {total_trials} total; "
        f"pooled trial-Sharpe dispersion σ = {pooled_std:.4f}. Each section also "
        "shows its grid-only per-sleeve DSR for comparison; the project-wide "
        "number is the one verdicts use.\n\n"
        f"Embargo: {args.embargo} bars purged between every train_end and "
        "test_start (sized to the 1-bar label horizon + buffer; feature "
        "lookback needs no embargo in a forward-chaining split).\n"
    )
    for r in results:
        sections.append(render_sleeve(r))

    report = "\n".join(sections)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{datetime.now():%Y-%m-%d-%H%M}-real.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\n{report}\n\nReport archived to {out_path}")


if __name__ == "__main__":
    main()
