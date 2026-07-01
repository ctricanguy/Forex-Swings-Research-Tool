# Building a Swing-Forex Research & Signal Tool: Architecture and the Honest Edge Landscape

> This document is the design rationale and the honest research basis for the
> code in this repository. The first section maps the concepts here to the
> modules that implement them; the rest is the full architecture and edge survey.

## How this document maps to the code

| Concept in this doc | Where it lives |
|---|---|
| One shared strategy library, two runners | `forex/strategies/` imported by both `forex/backtest/` and `forex/live/` |
| Point-in-time `generate_signals(data)` | `forex/strategies/base.py` (`Strategy` ABC) |
| Act on completed bars only | `complete` flag in `forex/data/base.py`; dropped in `run_backtest` and fetched with `complete_only=True` in `live/runner.py` |
| One-bar execution shift (central, single place) | `forex/backtest/engine.py` (`position.shift(1)`) mirrored by `live/runner.py` |
| Explicit spread + swap costs | `CostModel` in `forex/backtest/engine.py` |
| Lookahead self-test | `detect_lookahead()` in `forex/backtest/engine.py`; `tests/test_no_lookahead.py` |
| Unified evaluation interface (per-pair **and** cross-sectional) | `Backtestable` in `forex/backtest/evaluation.py` (`StrategyBacktest` + `PortfolioBacktest`) |
| Walk-forward refit + untouched holdout | `walk_forward()`, `holdout_split()` in `forex/backtest/validation.py` |
| Deflated / Probabilistic Sharpe Ratio | `forex/backtest/statistics.py` |
| Cross-sectional factor premia (rank pairs, long/short) | `forex/strategies/cross_sectional.py` (signals) + `forex/backtest/portfolio.py` (`PortfolioBacktest`, `rank_weights`) over a `forex/data/panel.py` universe |
| Per-pair carry / momentum / trend factors | `forex/strategies/{carry,momentum,ma_cross}.py`, combined in `ensemble.py` |
| Value factor (PPP / long-horizon reversal) + diversified sleeve | `XSValue`, `XSEnsemble` in `forex/strategies/cross_sectional.py` |
| Real swap/rollover per pair | rate-differential financing in `forex/backtest/portfolio.py` |
| Risk management (vol targeting, drawdown kill switch, sizing) | `forex/risk.py` (`RiskOverlay` — itself a `Backtestable`) |
| Real interest-rate data (FRED, keyless) | `forex/data/rates.py` |
| Config-driven deployment | `forex/strategies/registry.py` + `forex/config.py` |
| Auditable track record | `forex/tracking.py` + `signals`/`outcomes`/`books` tables |
| Service packaging & regulatory caveats | `docs/SERVICE.md` |
| Point-in-time SQLite persistence | `forex/data/storage.py` |
| OANDA v20 bid/ask candles | `forex/data/oanda.py` |
| Telegram alerts + APScheduler on a Pi under systemd | `forex/live/`, `deploy/forex-bot.service` |

---

## TL;DR
- **Build it modular, with one shared strategy library that the backtester and the live Telegram bot both import** — this is the single most important architectural decision because it guarantees what you backtest is exactly what runs live, eliminating the lookahead bias that destroys most retail systems. Use `python-telegram-bot` + `APScheduler` + SQLite + `vectorbt` (research) / a thin custom event loop (live) on a 64-bit Raspberry Pi 4/5, pulling daily/4-hour bars from OANDA's v20 API.
- **The honest edge map: documented, durable sources of return in swing FX are the macro FACTOR premia — carry, momentum, value, and trend — not chart-pattern indicators.** Published net-of-cost Sharpe ratios are modest (carry ~0.74, momentum ~3.9% p.a. net of full spread, a diversified combo perhaps ~0.7–1.0 gross), they suffer crash risk and regime dependence, and several have visibly decayed since publication. Single-pair technical-indicator systems mostly do not survive data-snooping corrections.
- **For a solo retail trader, the realistic expectation is near-zero to modestly positive edge after costs, and durability comes from risk management and diversification rather than signal cleverness.** Treat this as a learning/research instrument: combine factors as a diversified sleeve, validate with walk-forward and the deflated Sharpe ratio, model spread + swap explicitly, and size positions conservatively.

## Key Findings

1. **Separation of backtest and live workflows via a shared, point-in-time strategy module is the rigor linchpin.** The same `generate_signals(data_up_to_t)` function must be callable both in a historical loop and in the live scheduler. Lookahead bias — using data that wouldn't have been available at decision time — is "invisible": backtests look reasonable (55% win rate, 1.6 profit factor) but the strategy fails live because "that tiny edge was the entire edge."

2. **The academic finance literature documents four real, cross-asset return premia in currencies: carry, momentum, value, and trend.** These are the serious candidates for edge, not RSI/MACD crossovers. Menkhoff, Sarno, Schmeling & Schrimpf (2012, *Journal of Finance*, 67(2):681–718) report a carry trade with mean excess return ~7.2% p.a. and a net-of-bid-ask Sharpe of ~0.74 (all countries) / ~0.56 (developed). Asness, Moskowitz & Pedersen (2013, "Value and Momentum Everywhere," *Journal of Finance*) document value and momentum premia across currencies and other asset classes with a strong common factor structure.

3. **Carry's returns are compensation for crash risk, not free money.** Brunnermeier, Nagel & Pedersen (2008, "Carry Trades and Currency Crashes," *NBER Macroeconomics Annual*) show carry earns the forward-premium-puzzle return but is exposed to sudden crashes when global volatility spikes and positions unwind. Menkhoff et al. (2012, JF) find global FX volatility innovations explain **more than 90%** of the cross-sectional spread in carry portfolios — i.e., the premium is paid for bearing volatility/crash risk.

4. **Currency momentum profits shrink sharply after transaction costs and concentrate in hard-to-trade currencies.** Menkhoff, Sarno, Schmeling & Schrimpf (2012, *Journal of Financial Economics*, 106(3):660–684) find a gross cross-sectional spread "of up to 10% per annum" (best strategy MOM(1,1) gross Sharpe 0.95), but applying the full quoted bid-ask spread drops that best strategy from ~10% to **3.92% p.a.**; at more realistic 50–75% effective spreads it is ~5.3–6.6%. They conclude there are "very effective limits to arbitrage" — momentum concentrates in high-idiosyncratic-volatility, high-country-risk currencies (≈8% p.a. vs ≈4% p.a. for low-risk) and turnover can exceed 70%/month.

5. **There is direct evidence that published anomalies decay — possibly because traders learn them.** Hutchinson, Kyziropoulos, O'Brien, O'Reilly & Sharma (2022, *International Review of Financial Analysis*, Vol. 83, art. 102245, "Are carry, momentum and value still there in currencies?") found the average out-of-sample Sharpe falls "from +0.39 to −0.32" after publication, with "currencies no longer respond[ing] to interest rate and real exchange rate differentials … consistent with investors learning about mispricing from academic research."

6. **Pure technical trading rules largely fail once you correct for data-snooping.** Qi & Wu (2006) applied White's Reality Check to 2,127 FX trading rules and found profitability "becomes much weaker in the more recent period." Park & Irwin (2005) found that of 17 US futures markets, only two showed statistically significant profits after data-snooping correction. The multiple-testing problem is the central trap: scanning many pairs × many indicators × many parameters guarantees you'll find something that looks good by luck.

7. **Retail forex traders lose money at high rates.** ESMA's product-intervention analysis across EU jurisdictions found "74–89% of retail investment accounts typically lose money on their investments, with average losses per client ranging from €1,600 to €29,000" (ESMA, 2018; the range is reaffirmed in ongoing broker disclosures). The US CFTC's consumer advisory states that "about two out of three retail foreign exchange traders end each quarter in the red," based on quarterly profitability data disclosed by registered US forex dealers.

8. **Documented swing-relevant information with predictive value:** interest-rate differentials/carry (yes), monetary-policy expectations (yes, with caveats), real-exchange-rate/PPP deviations for value (yes, long-horizon), and trend/momentum (yes, decaying). COT positioning and economic-data-surprise indices are weaker — useful as context/confirmation but with limited standalone predictive power and reporting lags.

## Details

### PART 1 — ARCHITECTURE

#### The core principle: one strategy library, two runners

The most important design decision is to physically separate **strategy logic** from **execution context**. Structure the project so that:

- A `strategies/` package contains pure functions/classes: `generate_signals(df) -> signals`, taking a DataFrame of bars (and any exogenous series like rate differentials) and returning entry/exit/position signals. This code knows nothing about whether it's being backtested or run live.
- A `backtest/` runner feeds historical data into the strategy bar-by-bar (or via vectorized slicing) and simulates fills, spread, and swap.
- A `live/` runner is invoked on a schedule, fetches the latest *closed* bars, calls the **identical** `generate_signals`, and emits Telegram messages.

This guarantees backtest/live parity. The rule that prevents lookahead: **only act on completed bars** and **shift signals by one bar** (yesterday's MA cross → today's action, never the still-forming candle). OANDA's candle endpoint exposes a `complete: true/false` flag precisely so you can drop the still-forming bar — use it. A useful sanity test (borrowed from Freqtrade's lookahead-analysis idea): artificially delay every feature by one bar and confirm performance degrades gracefully rather than collapsing — a strategy that collapses under a one-bar delay was relying on future information.

> Implementation note: this repo applies the one-bar shift **once**, centrally, in
> `forex/backtest/engine.py` (`position.shift(1)`), and the live runner mirrors the
> exact same convention. The lookahead self-test is implemented as the more robust
> *truncation test* in `detect_lookahead()` — recompute each bar's position using
> only data up to that bar and confirm it is unchanged from the full-history value —
> because delaying the price column alone would shift the signal and the return
> series together and could hide a `shift(-1)` leak.

#### Layer-by-layer

**Data ingestion layer.** Fetch OHLC bars on a schedule, store point-in-time, never overwrite history. For a swing system on daily/4h bars, OANDA's v20 REST API via the mature `oandapyV20` Python wrapper is the natural fit: free API access on demo and live accounts (no separate data subscription), an `InstrumentsCandlesFactory` that auto-paginates around the 5,000-candle-per-request limit, and bid/ask/mid candles (`price=BA` gives you the spread directly, which you need for cost modeling). Caveat from OANDA's own docs: the candle endpoint returns base-pricing candles that can differ slightly from your live execution feed, and OANDA does not publish an unlimited tick archive via the API. For long macro history (rates, CPI for value/PPP work), add FRED (free).

> Implementation note: this repo does **not** use the ``oandapyV20`` SDK -- its
> setup.py no longer builds under modern setuptools (Raspberry Pi OS included).
> ``forex/data/oanda.py`` speaks the v20 REST API directly with the standard
> library (auth header + JSON + pagination), with an injectable HTTP layer so
> the provider is fully tested offline.

**Persistence.** SQLite is the right default for a Pi: zero-config, single-file, ships with Python, and handles this data volume trivially. Use one table for bars (instrument, granularity, timestamp, o/h/l/c, bid/ask, volume, complete), one for emitted signals (timestamp, instrument, direction, entry, SL, TP, strategy, rationale), and one for tracked outcomes. Flat Parquet files are a fine complement for bulk historical research data that vectorbt reads. Avoid heavier DBs (Postgres) — unnecessary overhead on a Pi.

**Strategy/signal layer.** Keep indicator computation deterministic and vectorized with pandas. For indicators, prefer **`pandas-ta`** (pure Python/pandas, `pip`-installable) over **`TA-Lib`** on a Pi — TA-Lib requires compiling an underlying C library and notoriously fails to build on ARM (`error: command 'arm-linux-gnueabihf-gcc' failed`), though recent TA-Lib wheels (0.6.5+) bundling the C library and a 64-bit OS reduce this pain. On 64-bit Raspberry Pi OS (ARMv8), most scientific wheels resolve from piwheels.

**Backtesting layer.** For research, **vectorbt** is the best fit: it vectorizes with NumPy/Numba and runs thousands of parameter combinations fast — exactly what you need for walk-forward and multiple-testing-aware studies (and, ironically, what makes overfitting easy, so discipline matters). **backtesting.py** is the gentler on-ramp for a Python beginner (define a strategy in two methods, get a report in seconds) but is research-only with a small ecosystem and no live trading. **backtrader** is event-driven and mirrors live fills well (limit/stop orders, commissions, slippage) and has an OANDA bridge, but original development stalled around 2018–2021 (community fork `backtrader2` is maintenance-only). Recommendation: prototype in backtesting.py, scale parameter studies in vectorbt, and keep your *own* thin event loop for live so the live path stays simple and auditable.

> Implementation note: this repo ships its **own** thin, auditable engine
> (`forex/backtest/engine.py`) so the backtest and live paths share one shift/cost
> convention. vectorbt / backtesting.py are listed as optional research extras in
> `requirements-dev.txt` for parameter sweeps.

**Telegram layer.** `python-telegram-bot` (PTB) is the most popular, best-documented choice and is ideal for a personal command/alert bot (`/status`, `/signals`, `/backtest`); `aiogram` is the async-first alternative favored for high-concurrency bots — overkill here. For a personal tool, PTB plus a simple "push message to my chat_id" function is enough. Get your bot token from @BotFather and your chat ID from a helper bot.

> Implementation note: one-way alerts here use only the standard library
> (`urllib`) in `forex/live/notifier.py`, so no extra dependency is needed just to
> push messages. Add PTB when you want interactive commands.

**Scheduling & process management on the Pi.**
- **Scheduling:** Use **APScheduler** inside the long-lived Python process rather than cron. It supports cron-style, interval, and one-off triggers, can persist jobs to SQLite so they survive restarts, and keeps everything in one process with your bot. Cron is fine for a dead-simple "run script every hour," but APScheduler integrates better with an always-on bot.
- **Process management:** Run the bot as a **systemd service**. systemd auto-restarts on crash and on reboot, handles logging via journald, and is the standard, lowest-overhead approach on Raspberry Pi OS. Docker is reasonable if you want reproducible deployments but adds overhead on a Pi; for a single personal process, systemd is simpler. Skip supervisor unless you specifically prefer it.
- **Resource notes:** A Pi 4/5 with 64-bit OS comfortably runs this workload (daily/4h polling of ~28 major/minor pairs is light). Keep vectorbt-heavy parameter sweeps for occasional research runs, not the live loop. Use a quality power supply and consider running off SSD rather than SD card for database durability.

#### Forex data API comparison (swing/daily-bar use case)

| Provider | Historical depth | Cost (entry) | Python ease | Best for |
|---|---|---|---|---|
| **OANDA v20** | Decades of daily; good intraday; bid/ask candles | Free API w/ account; spreads are the cost | Excellent (`oandapyV20`) | **Best default** — broker + data + bid/ask candles + (later) execution in one |
| **Polygon.io** | Tick + aggregates; institutional-sourced FX | Paid tiers; free tier ~5 calls/min | Good | Low-latency/intraday; FX coverage thinner than equities; no native indicators |
| **Twelve Data** | Multi-asset; some series shorter history | Free 800/day (8/min); Grow ~$29/mo | Very good, polished docs | All-in-one dashboards, easy start |
| **EODHD / others** | Long daily history | Paid | Good | Bulk EOD history |

For a swing system that trades on daily/4h closes and needs bid/ask for cost modeling and a path to (optional) execution, **OANDA v20 is the recommended primary source**; add Twelve Data or Polygon only if you need broader cross-asset context.

### PART 2 — THE HONEST EDGE LANDSCAPE

#### What actually has documented predictive power

**1. Carry (interest-rate differential / forward-premium puzzle).** The oldest and most robust FX anomaly. Uncovered interest parity predicts high-rate currencies should depreciate to offset the rate advantage; empirically they don't (and often appreciate slightly) — the "forward premium puzzle" (Fama, 1984). Going long high-rate, short low-rate currencies earned, in Menkhoff et al. (2012, JF), ~7.2% p.a. with a net-of-cost Sharpe of ~0.74 (all countries) / ~0.56 (developed); their earlier working-paper version (BIS WP 366) reported ~0.82, which is the figure most secondary sources quote. **Crucial caveat:** this is compensation for crash risk. Brunnermeier, Nagel & Pedersen (2008) show carry "works until it doesn't" — sudden unwinds during volatility spikes ("up by the stairs, down by the elevator"). Carry has high negative skew.

**2. Momentum.** Currencies trend. Menkhoff et al. (2012, JFE) found cross-sectional momentum spreads "of up to 10% per annum" gross (best-strategy gross Sharpe 0.95). **But:** the full quoted bid-ask spread cuts the best strategy from ~10% to ~3.92% p.a.; at 50–75% effective spreads it's ~5.3–6.6%. Profits concentrate in high-idiosyncratic-vol, high-country-risk currencies (≈8% vs ≈4% p.a.) and carry monthly turnover that can exceed 70% — so costs and limits-to-arbitrage matter enormously for a retail trader.

**3. Value (PPP / real-exchange-rate mean reversion).** Over long (multi-year) horizons, currencies that are cheap relative to PPP/real-exchange-rate fundamentals tend to revert. Asness, Moskowitz & Pedersen (2013) measure currency value as the ~5-year reversal in the real exchange rate; it earns a premium and, critically, is **negatively correlated with momentum**, which is why combining them is powerful. Value is slow-moving and low-turnover (cost-friendly).

**4. Trend following (time-series momentum).** The CTA staple. Documented over long histories, but **performance has visibly decayed**: trend strategies have shown long flat periods (one widely cited study notes the trend was "virtually flat" since ~2011), and trend-following CTAs have been among the weaker hedge-fund strategies in recent years. Note the psychology: with Sharpe ~0.7, the *typical* drawdown duration is ~1/S² ≈ 2 years, and 4-year drawdowns "are not exceptional."

#### The combination is greater than the parts

The single most important positive finding: **carry, momentum, and value are imperfectly (and sometimes negatively) correlated, so a diversified multi-factor sleeve has a meaningfully higher and more stable Sharpe than any single factor.** Asness et al.'s entire thesis is that value + momentum "everywhere," combined, produces a strong common-factor return. This is the opposite of the retail instinct to find the one perfect indicator. Diversify across factors AND across pairs.

#### Why most retail technical-indicator systems fail

- **Multiple-testing / data-snooping.** Scanning N pairs × M indicators × P parameter sets means thousands of trials; the best-looking one is almost certainly luck. White's Reality Check and Hansen's SPA exist precisely to deflate this. Qi & Wu (2006): FX rule profitability "becomes much weaker in the more recent period." Park & Irwin (2005): only 2 of 17 futures markets significant after correction.
- **In-sample overfitting / curve-fitting.** Bailey & López de Prado (2014, "The Deflated Sharpe Ratio") show high backtest Sharpe is trivially achievable after testing relatively few configurations, and overfit strategies *systematically underperform* (not merely fail to outperform) out-of-sample due to memory effects in financial series.
- **Ignoring the bid-ask spread and swap/rollover.** Swing positions held overnight pay/receive swap; every entry/exit pays the spread. The momentum result above (10% → 4% under full spread) is the canonical demonstration that costs are decisive.
- **Adverse selection & worse retail execution.** Retail traders face wider spreads, slippage in fast markets, and re-quotes that institutions don't.
- **Backtest ≠ live.** Only live trading is genuinely out-of-sample; backtesting "suffers from lookahead bias by design."

#### Information that matters vs. noise

- **Interest-rate differentials & central-bank policy expectations:** genuine, primary driver of medium-term FX direction. Highest signal. (A 2025 Federal Reserve Bank of Boston working paper, "The Predictability of Global Monetary Policy Surprises," even finds policy-rate surprises themselves are partly predictable from global rate moves — e.g., a 1pp rise in global short rates in the 15 days before a meeting is associated with a ~12bp surprise increase — underscoring how central rates dominate.)
- **Economic data surprises (actual vs. consensus):** move FX, and aggregate measures like the Citigroup Economic Surprise Index track beats/misses — but the predictive power of raw economic-news surprises for exchange rates is weak (regressions in the FX-predictability literature typically show very low adjusted R², on the order of well under 1%). Useful as context, weak as a standalone signal; the indices are mean-reverting and lagging.
- **COT positioning:** widely watched as "smart money," but academic evidence is skeptical — the report has a 3-day lag, and studies find position-based sentiment indices "do not produce significant average returns." Use for macro confirmation/extremes, not entry timing.
- **Risk sentiment / VIX / risk-on-risk-off:** matters mainly through its link to carry crash risk (high-rate currencies sell off when volatility spikes). Worth tracking as a regime filter for carry exposure.
- **Real-exchange-rate / PPP deviations:** genuine long-horizon value signal.

#### Realistic performance expectations

- **Sharpe:** Single factors net of cost are roughly 0.4–0.75; a well-diversified multi-factor FX sleeve might reach ~0.7–1.0 gross in academic studies, but a retail implementation with retail costs should expect *materially less*. Very few CTAs sustain Sharpe > 1 over long periods.
- **Drawdowns:** Expect multi-year drawdowns even for genuinely positive-Sharpe strategies (typical drawdown length ≈ 1/Sharpe² years).
- **Costs are the deciding variable.** Spread + swap can halve or eliminate a swing strategy's gross edge. Model them explicitly in every backtest.
- **Risk management, not signals, is where durability lives.** Fixed-fractional position sizing (e.g., risk ~1% of equity per trade), a maximum-drawdown kill-switch, and caps on concurrent correlated positions do more for long-run survival than signal tuning.

## Recommendations

**Stage 0 — Plumbing (weeks 1–3).** Stand up the skeleton: OANDA demo account + `oandapyV20`, SQLite schema for bars and signals, `python-telegram-bot` pushing a daily "heartbeat" message via APScheduler, all running under systemd on a 64-bit Pi. *Benchmark:* the bot reliably pulls daily closed bars for ~10 major pairs and messages you once a day for a week without manual restarts.

**Stage 1 — Shared strategy module + honest backtester (weeks 3–8).** Build the `generate_signals()` library and a backtester (backtesting.py first) that (a) acts only on `complete` bars, (b) shifts signals one bar, (c) charges the actual bid-ask spread and a swap estimate. Implement one *simple* factor first — e.g., cross-sectional carry using rate differentials, or 12-month time-series momentum. *Benchmark:* your backtest of a trivial rule reproduces a hand-calculated result, and a one-bar feature delay degrades — not destroys — performance, proving no lookahead.

**Stage 2 — Validation discipline (weeks 8–14).** Before believing any result: walk-forward analysis (rolling re-fit), an out-of-sample holdout you do NOT touch until the end, and the **deflated Sharpe ratio** / probability of backtest overfitting given how many configurations you tried. Keep a written log of every backtest run (López de Prado's advice: track the number of trials so the Sharpe can be properly deflated). *Benchmark:* a strategy only graduates if its deflated Sharpe stays positive and it survives walk-forward across regimes (incl. 2008, 2015 CHF, 2020, 2022).

**Stage 3 — Diversified factor sleeve, live in alert-only mode (months 4+).** Combine carry + momentum + value as a small portfolio of signals across pairs rather than a single indicator. Run live as **signal/alert-only** (no auto-execution) and paper-trade for at least 3–6 months, logging signal-vs-realized to measure backtest-to-live decay. *Benchmark to stop and rethink:* if live results diverge materially from backtest, or deflated Sharpe is ≤ 0, do not add capital.

**Thresholds that change the plan:**
- If, after costs, no factor or combo shows a positive deflated Sharpe out-of-sample → treat the project as a *learning tool*, not a money-maker, and do not risk meaningful capital.
- If TA-Lib won't build → use pandas-ta and move on; don't lose days to it.
- If the Pi struggles with vectorbt sweeps → run research sweeps on a laptop/cloud and deploy only the lightweight live loop to the Pi.
- If you ever want execution → backtrader's OANDA bridge or OANDA's order endpoints, but only after extended paper trading and with a hard daily-loss kill-switch.

## Caveats

- **Survivorship and selection bias pervade the rosy literature.** Many reported Sharpe ratios are gross, in-sample, or from working-paper versions; the published carry Sharpe is ~0.74 net, not the ~0.82 widely quoted from the working paper. Anomalies decay post-publication (Hutchinson et al. 2022 found Sharpe going from +0.39 to −0.32).
- **Practitioner/marketing sources overstate edge.** Broker blogs, "AI signal" vendors, and outlier papers (e.g., a piercing-line/dark-cloud-cover study claiming 600%+ cumulative) should be read skeptically; they rarely correct for multiple testing or realistic costs.
- **Backtest performance is an upper bound, not an expectation.** Live retail results are systematically worse due to costs, execution, and behavioral error. The 74–89% retail loss rate (ESMA) and "two out of three lose each quarter" (CFTC) are the base rates you're fighting.
- **One source attribution flagged for care:** the precise "economic-news surprises explain <1% of FX variance" claim is supported by the general FX-predictability literature; the specific Boston Fed (2025) working paper focuses on *monetary-policy-rate* surprise predictability rather than data-surprise → exchange-rate predictability, so treat the R² figure as illustrative of the literature's consensus rather than a single citable headline.
- **Forward-looking statements flagged:** claims about future factor performance, capacity, or further decay are inherently uncertain; the decay evidence is suggestive, not deterministic.
- **This is not investment advice.** The defensible goal is to build a rigorous research instrument that teaches you quant methodology and *might* surface a modest, well-validated edge — not a money printer.
