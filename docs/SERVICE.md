# Selling Signals as a Service: What This Codebase Gives You, and What It Can't

This document is the honest bridge between "research tool that works" and
"product people pay for". Read the caveats before the architecture.

## The two assets a signal service lives or dies on

1. **A verifiable, timestamped track record.** Nobody should (or will, for long)
   pay for backtests. The `signals` table records every alert at emission time
   with the bar that produced it; `forex/tracking.py` marks spans to market
   after the fact and never edits history; completed bars in `bars` are
   immutable by design. That chain — *emitted before the fact, marked after,
   tamper-evident* — is the product. Protect it: back up the SQLite file, never
   hand-edit it, and publish the track record with its methodology (mid-price
   marks, span convention) attached.

2. **A validation story you can defend.** Every deployed sleeve should have a
   walk-forward + holdout + deflated-Sharpe report (`run_validation.py` /
   `run_portfolio.py`) archived from *before* it went live, plus the ongoing
   live-vs-backtest comparison. When a subscriber asks "why should I believe
   this?", the answer is a process, not a promise.

## Service architecture on top of this repo

```
                        ┌──────────────────────────────┐
 OANDA v20 ──candles──► │  PortfolioLiveRunner (systemd)│──alerts──► Telegram channel
 FRED ──rate diffs────► │  + RiskOverlay sizing         │            (subscribers)
                        │  + SQLite (bars/signals/books)│──/track──► bot commands
                        └──────────────────────────────┘
```

* **Delivery:** a private Telegram channel (broadcast) for alerts; the
  interactive bot (`forex/live/bot.py`) for `/track`, `/book`, `/status` self
  -service. One process, systemd-managed, on a Pi or a $5 VPS.
* **Tiers that make sense here:** free = delayed signals or monthly track-record
  digest; paid = real-time alerts + the live book + the bot. The `books` table
  already separates "what the model holds" from per-pair alerts, so tiering is a
  routing decision, not a code change.
* **Operational hardening before charging money:** off-device DB backup (the
  track record IS the business), a dead-man heartbeat (daily "I'm alive"
  message; silence = investigate), pinned dependency versions, and the OANDA
  smoke test (`scripts/smoke_oanda.py`) gating any deploy.

## Regulatory reality — read this before taking a single payment

**This is not legal advice; get real legal advice for your jurisdiction before
charging anyone.** The short version of why:

* **US:** providing forex/futures trading advice for compensation can require
  CTA (Commodity Trading Advisor) registration with the CFTC/NFA. There is a
  "publisher's exemption" for impersonal, non-tailored publications, but its
  boundaries are fact-specific — a Telegram signal service sits uncomfortably
  close to the line.
* **UK/EU:** "investment advice" vs "generic recommendation" distinctions under
  FCA/MiFID rules matter; signal services have been enforcement targets,
  especially where CFDs/forex and retail audiences are involved.
* **Everywhere:** marketing rules are stricter than product rules. Publishing
  hypothetical/backtest performance without required disclaimers, or cherry-
  picked results, is the classic violation. The `DISCLAIMER` string on every
  alert is a floor, not a defense.

Structurally safest patterns (still: lawyer first): impersonal research
publication with no account-specific advice, no "buy now" urgency framing,
prominent risk disclosure, and published methodology + full (not curated) track
record.

## Honest product expectations

The edge landscape doc (ARCHITECTURE.md Part 2) applies to the *service* too:
factor premia are modest, decaying, and drawdown-prone. A service built on
"signals that print money" dies with its first 6-month drawdown and takes your
reputation with it. A service built on *transparent process* — here's the
validation, here's the live record including losses, here's the risk overlay
that kept the drawdown at -12% — is differentiated precisely because almost
nobody in the retail signal market does it. Sell the honesty; it's the only
part of this that's genuinely scarce.
