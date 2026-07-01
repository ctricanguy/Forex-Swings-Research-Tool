# Swing-Forex Research & Signal Tool

A modular, honest toolkit for researching and running **swing** (daily / 4-hour)
forex strategies. It is built around one architectural rule:

> **The same strategy code is imported by both the backtester and the live
> runner.** What you backtest is *exactly* what runs live.

That single decision eliminates the most common reason retail systems fail —
backtest/live divergence and lookahead bias. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design and an honest
survey of where edge in swing FX actually comes from (spoiler: macro factor
premia — carry, momentum, value, trend — not chart-pattern indicators, and even
those are modest net of costs).

This is a **research and alert tool by design** — it emits signals, it does not
auto-trade.

## Why this layout

```
forex/
├── data/            # canonical bar schema + providers (synthetic, OANDA),
│                    #   SQLite store, Panel (multi-pair universe), FRED rates
├── strategies/      # THE SHARED LIBRARY: pure, point-in-time signal logic
│                    #   (per-pair + cross-sectional factors) + config registry
├── backtest/        # honest per-pair engine + cross-sectional PortfolioBacktest,
│                    #   the Backtestable interface, and the validation harness
│                    #   (walk-forward, holdout, PSR / deflated Sharpe)
├── risk.py          # RiskOverlay: vol targeting + max-drawdown kill switch
├── tracking.py      # signal outcomes -> auditable track record
└── live/            # scheduled runners (per-pair + portfolio), notifiers, bot
scripts/             # run_backtest.py, run_live.py
deploy/              # systemd unit for a Raspberry Pi
tests/               # parity, cost, storage, and lookahead-bias tests
docs/ARCHITECTURE.md # design rationale + the honest edge landscape
```

A strategy is a pure function of *past* bars (`generate_signals(data) -> position`).
The **one-bar execution shift** (decide on bar *t*'s close, get filled on *t+1*)
and **cost accounting** live in exactly one place — the engine — and the live
runner mirrors the same convention. Strategies never reach into a broker SDK;
they only see the canonical bar frame, so providers are swappable.

The validation harness depends on one abstraction — `Backtestable`, which yields
a **dated return/position stream over a window** (`run(start, end) -> BacktestResult`).
A single-pair strategy is adapted via `StrategyBacktest`; the upcoming
cross-sectional portfolio engine will implement the *same* interface, so it plugs
into walk-forward / holdout with zero harness changes.

## Quickstart (no API keys needed)

The synthetic data provider lets the whole pipeline run offline.

```bash
pip install -r requirements-dev.txt

# Backtest time-series momentum and run the lookahead self-check
python scripts/run_backtest.py --strategy tsmom --lookback 60 --check-lookahead

# Other built-in strategies
python scripts/run_backtest.py --strategy ma_cross
python scripts/run_backtest.py --strategy carry

# One live polling cycle (synthetic data, alerts to the console)
python scripts/run_live.py --once

# Run the tests
pytest
```

> The synthetic numbers are **not** an edge — they exercise the machinery. Real
> results come only from real data, walk-forward validation, and honest costs.

## Going live with OANDA + Telegram

1. Create an OANDA practice (demo) account and an API token.
2. Create a Telegram bot via [@BotFather] and find your chat id.
3. Configure:
   ```bash
   cp .env.example .env                       # fill in tokens
   cp config/config.example.yaml config/config.yaml
   # set data_source: oanda in config.yaml
   ```
4. Dry run one cycle, then start the scheduler:
   ```bash
   python scripts/run_live.py --once --config config/config.yaml
   python scripts/run_live.py --schedule --config config/config.yaml
   ```
5. On a Raspberry Pi, run it under systemd (auto-restart, starts on boot):
   ```bash
   sudo cp deploy/forex-bot.service /etc/systemd/system/
   sudo systemctl enable --now forex-bot
   journalctl -u forex-bot -f
   ```

Secrets come from environment variables (`.env`) and override the YAML, so
tokens never get committed.

## Built-in strategies

**Per-pair** (`Strategy.generate_signals(data) -> position`, one instrument):

| Strategy | Module | Notes |
|---|---|---|
| `TimeSeriesMomentum` | `forex/strategies/momentum.py` | Trend/momentum factor; sign of trailing return. |
| `MovingAverageCross` | `forex/strategies/ma_cross.py` | Classic SMA cross — a baseline, not an edge. |
| `Carry` | `forex/strategies/carry.py` | Interest-rate-differential factor (needs a `rate_diff` column). |
| `Ensemble` | `forex/strategies/ensemble.py` | Diversified multi-factor sleeve (averages members). |

**Cross-sectional** (`CrossSectionalSignal.score(panel) -> wide scores`, ranked across pairs):

| Signal | Module | Notes |
|---|---|---|
| `XSMomentum` | `forex/strategies/cross_sectional.py` | Rank pairs by trailing return; long top / short bottom. |
| `XSCarry` | `forex/strategies/cross_sectional.py` | Rank pairs by rate differential (needs `rate_diff`). |

| `XSValue` | `forex/strategies/cross_sectional.py` | Long-horizon reversal (PPP proxy); or a real-value field. |
| `XSEnsemble` | `forex/strategies/cross_sectional.py` | Diversified sleeve: z-scored, weighted factor average. |

Cross-sectional signals feed `PortfolioBacktest`, which builds a dollar-neutral
long/short book (`rank_weights`), accrues **real per-pair swap financing** from
rate differentials, and is itself a `Backtestable` — so it validates through the
exact same harness. Wrap anything in `RiskOverlay` for vol targeting and a
max-drawdown kill switch, and it's *still* a `Backtestable`:

```bash
python scripts/run_portfolio.py --signal xsens --vol-target 0.10 --max-drawdown 0.15
python scripts/run_portfolio.py --signal xscarry
```

Real rate differentials come from FRED with **no API key**
(`forex/data/rates.py`; verify the series map before trusting it). Strategies
are buildable from config specs (`forex/strategies/registry.py`), so the
deployed sleeve is a YAML edit — see `config/config.example.yaml`.

Add your own per-pair strategy by subclassing `forex.strategies.base.Strategy`,
or a cross-sectional one by subclassing
`forex.strategies.cross_sectional.CrossSectionalSignal`. The point-in-time and
parity tests apply to every strategy automatically once you add it to the
parametrised test lists.

## Validation discipline (don't skip this)

A strategy should only "graduate" if it survives all four checks — and the harness
implements them:

1. **No lookahead** — `detect_lookahead()` returns `leaking_bars == 0`.
2. **Honest costs** — spread (from bid/ask) and swap modelled explicitly in `CostModel`.
3. **Walk-forward / out-of-sample** — `walk_forward()` refits the best config on a
   rolling (or anchored) train window and stitches the out-of-sample test windows
   into one honest return stream; `holdout_split()` keeps an untouched tail.
4. **Deflated Sharpe** — `deflated_sharpe_ratio()` discounts the result by how many
   configurations you tried; a high backtest Sharpe is trivial to find by luck.

```bash
# Walk-forward + holdout + PSR/deflated-Sharpe report (synthetic data)
python scripts/run_validation.py --strategy tsmom --train 250 --test 60
```

The report ends in a blunt verdict (`PASS` / `WEAK` / `FAIL`). On the synthetic
demo, momentum lands at `WEAK` (deflated Sharpe ~0.89) and the MA cross at `FAIL` —
the kind of discrimination you want before risking anything.

The base rate you are fighting: regulators report **~74–89%** of retail forex
accounts lose money. Treat this project as a rigorous learning instrument first.

## Track record & running as a service

Every emitted signal is timestamped at emission and marked to market afterwards
— never edited retroactively:

```bash
python scripts/track_record.py --db forex.db   # auditable performance digest
```

Portfolio mode runs the whole universe as one long/short book
(`mode: portfolio` in config), emitting a rebalance alert only when the book
changes. An optional interactive Telegram bot (`forex/live/bot.py`, needs
`python-telegram-bot`) serves `/status`, `/signals`, `/book`, `/track`.

If you intend to charge for signals, read **`docs/SERVICE.md`** first — it
covers the service architecture, and the regulatory reality (CTA/FCA rules
around paid trading advice) that you must clear with a lawyer before taking
money.

## Status

All roadmap stages from `docs/ARCHITECTURE.md` are implemented behind the single
`Backtestable` interface: shared strategy library (per-pair + cross-sectional
carry/momentum/value + ensembles), honest engines with real per-pair swap
financing, the validation harness (walk-forward, holdout, deflated Sharpe), the
risk layer (vol targeting + kill switch), FRED rate ingestion, config-driven
live runners (pair + portfolio), outcome tracking, and CI. Remaining before real
money: wire your OANDA token (`scripts/smoke_oanda.py`), verify the FRED series
map, then paper-trade the sleeve for 3–6 months and compare live vs backtest.

[@BotFather]: https://t.me/BotFather
