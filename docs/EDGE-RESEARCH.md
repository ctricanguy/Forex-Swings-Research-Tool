# Where Edge Can Still Come From in Today's FX Market — An Honest Map

*Written 2026-07, after two rounds of real-data validation (10y daily, OANDA,
full quoted spread). This is the research basis for what the tool tests next,
and — just as important — what it deliberately refuses to chase.*

## 1. What our own validation has established so far

Two rounds on real data, walk-forward + holdout + deflated Sharpe, at full
retail spread:

| Sleeve | Round 1 (8 majors) | Round 2 (16 pairs) | Reading |
|---|---|---|---|
| XS momentum | FAIL (DSR 0.01) | FAIL (DSR 0.01) | Robustly dead at daily/retail cost on liquid pairs — matches the post-cost literature. |
| XS momentum, trend-gated | — | FAIL (DSR 0.00) | The gate concentrated exposure but there was no edge to concentrate. |
| XS value (reversal) | WEAK (DSR 0.82) | FAIL (DSR 0.33) | **Did not survive a universe change** → the round-1 result was substantially luck. Universe-robustness is now a standing check. |
| Carry / ensembles | *never ran* | *never ran* | FRED fetch failed twice. **The most documented FX factor remains untested.** Fixed in this iteration (API key + disk cache + manual fallback). |

The meta-lesson: the harness is doing its job. Every "promising" result so far
has died under a robustness probe — which is exactly what the data-snooping
literature predicts happens to most retail findings.

## 2. Where the literature says edge still lives — and for whom

- Recent academic work ([Fan, 2025, *Financial Review* — "Optimizing Currency
  Factors"](https://onlinelibrary.wiley.com/doi/10.1111/fire.70000)) still finds
  momentum (SR ≈ 0.96) and carry (SR ≈ 0.61) statistically significant — **but
  gross, institutionally implemented, on broader universes including EM**, and
  finds value insignificant-negative (consistent with our round-2 collapse).
- Institutional FX factor products remain live ([J.P. Morgan Factor Views, Q2
  2026](https://am.jpmorgan.com/us/en/asset-management/institutional/insights/portfolio-insights/asset-class-views/factor/)
  reports FX carry currently leading macro factors).
- The gap between "significant in journals" and "profitable at retail" is
  costs, universe (EM carry needs brokers/spreads we don't have), and
  implementation (forward-implied carry vs. our proxy). That gap ate our
  momentum sleeve; it is the *expected* result, not an anomaly.

**Implication:** for a retail swing account, the residual candidates are (a)
carry — cheap to hold, low turnover, still untested by us; (b) *combinations*
(diversification is the one free lunch); (c) information sources the academic
factors don't use — positioning and event structure.

## 3. Information sources a retail account can actually access

Ranked by (evidence strength × accessibility):

1. **Rate differentials (carry)** — genuine, documented, low-turnover; our data
   path is now triple-redundant (FRED API key → fredgraph CSV → hand-dropped
   CSV cache). *Status: sleeve built, awaiting first real run.*
2. **Scheduled-event structure** — macro announcements account for roughly a
   third of FX price variation ([survey of the high-frequency announcement
   literature](https://www.sciencedirect.com/science/article/abs/pii/S221457961500012X);
   [what events matter for FX volatility, 2024](https://arxiv.org/abs/2411.16244)),
   and event days are precisely when carry books get hurt and spreads widen.
   Calendars (CPI/NFP via [FRED's release calendar
   API](https://fred.stlouisfed.org/releases/calendar), FOMC from the
   [Fed's published schedule](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm))
   are known **years in advance** — a rare, point-in-time-legitimate,
   non-price information source. *Status: `EventGate` built (avoid/only modes),
   calendar fetcher built.*
3. **Retail positioning (OANDA position book)** — aggregate client long/short
   percentages with historical snapshots via the API
   ([endpoint docs](https://developer.oanda.com/rest-live-v20/instrument-df/)).
   Practitioner convention is contrarian — fade the crowd at extremes
   ([overview](https://tradersunion.com/interesting-articles/forex-sentiment-analysis/);
   [OANDA's own tool](https://www.oanda.com/bvi-en/cfds/tools/orderbook/)).
   Honest read: OANDA-clients-only sample, and academic studies of
   position-based sentiment mostly find weak standalone power. It earns a test
   because it is *orthogonal to price*, not because it is proven.
   *Status: fetcher + `XSPositioning` built, awaiting snapshot history.*
4. **Vol regime** — not a return source itself, but the documented conditioner
   of carry (vol innovations price the carry cross-section). *Status: `VolGate`
   built.*
5. **COT futures positioning** — free, official, but 3-day lag and weak
   documented power; a later addition at best.

## 4. What we deliberately do NOT chase

- **True order flow / microstructure** (Evans & Lyons): the documented flow
  edge uses interdealer data (EBS/Refinitiv) that retail cannot buy at any
  sane price. OANDA's book is the accessible shadow of it — already in queue.
- **News-latency / event-reaction trading**: the 100-400 pip event spikes are
  real but are captured in the first seconds by machines colocated at the
  venues; a swing tool on daily bars structurally cannot compete there. Our
  event play is *avoidance/conditioning*, not reaction speed.
- **FX options vol premium**: documented, but needs an options-capable account
  and different infrastructure entirely.
- **ML on price alone**: with ~2,600 daily bars and the multiple-testing
  problem, fitting flexible models to price is a data-snooping machine. Price
  has been squeezed dry by round 1 and 2.

## 5. The test queue (in order, each one pre-registered here)

1. **Carry, plain** — the headline untested factor. Hypothesis: positive but
   modest; crash-prone.
2. **Carry + VolGate + EventGate** — carry, de-risked at the documented failure
   points. Hypothesis: similar return, better drawdown/DSR.
3. **Diversified sleeve (gated carry + value + gated momentum, abstaining
   ensemble)** — the literature's actual recommendation. Hypothesis: best DSR
   of anything tested.
4. **Positioning contrarian** — orthogonal information. Hypothesis: weak alone;
   worth testing as an ensemble member if it clears zero.
5. If all of the above land ≤ WEAK: the honest conclusion is that this account
   size/cost structure has **no deployable systematic edge at daily frequency**,
   and the tool's value is (a) the validation instrument itself, (b) alert-only
   research, (c) the service story built on transparent process rather than
   promised returns.

**Discipline reminder:** every sleeve above is one more trial against the same
10 years of data. As of validation v5 this is enforced quantitatively: the
report deflates every sleeve by the **project-wide** trial count (all configs
of all sleeves, this round plus `--prior-trials` from earlier rounds, with
pooled trial-Sharpe dispersion), and prints the grid-only per-sleeve DSR
alongside for comparison. A small train/test embargo (default 5 bars, sized to
the 1-bar label horizon) purges the walk-forward boundary. The queue is still
written down *before* the results are known, and the ensemble (item 3) remains
the pre-registered primary hypothesis, not whatever section looks best
afterwards.

## Sources

- [Fan (2025), "Optimizing Currency Factors", *Financial Review*](https://onlinelibrary.wiley.com/doi/10.1111/fire.70000)
- [J.P. Morgan Asset Management, Factor Views Q2 2026](https://am.jpmorgan.com/us/en/asset-management/institutional/insights/portfolio-insights/asset-class-views/factor/)
- [High-frequency macro-news FX literature survey (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S221457961500012X)
- [What events matter for exchange rate volatility? (arXiv 2411.16244)](https://arxiv.org/abs/2411.16244)
- [OANDA v20 instrument endpoints (order/position book)](https://developer.oanda.com/rest-live-v20/instrument-df/)
- [FRED economic release calendar](https://fred.stlouisfed.org/releases/calendar)
- [Federal Reserve FOMC calendars](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)
- [Forex sentiment analysis overview (Traders Union)](https://tradersunion.com/interesting-articles/forex-sentiment-analysis/)
- [OANDA order book tool](https://www.oanda.com/bvi-en/cfds/tools/orderbook/)
- [Retail sentiment as contrarian indicator (HolaPrime)](https://holaprime.com/prime-academy/fundamental-analysis/retail-trader-sentiment-as-an-indicator/)
