#!/usr/bin/env python3
"""Run the live signal runner (config-driven).

The config decides everything: data source, mode (per-pair strategy vs
cross-sectional portfolio), which strategy/signal spec to build, risk settings,
and where alerts go. Changing the deployed sleeve is a config edit, not a code
change.

Examples
--------
    # One dry-run cycle on synthetic data, alerts to the console
    python scripts/run_live.py --once

    # Long-running scheduled service using config.yaml + env-var secrets
    python scripts/run_live.py --schedule --config config/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forex.config import Config
from forex.data import SyntheticDataProvider
from forex.data.storage import Storage
from forex.live import ConsoleNotifier, LiveRunner, PortfolioLiveRunner, TelegramNotifier
from forex.strategies.registry import build_signal, build_strategy


def build_provider(cfg: Config):
    if cfg.data_source == "oanda":
        from forex.data.oanda import OandaDataProvider

        if not cfg.oanda.access_token:
            raise SystemExit("OANDA selected but OANDA_ACCESS_TOKEN is not set.")
        return OandaDataProvider(cfg.oanda.access_token, cfg.oanda.environment)
    return SyntheticDataProvider(with_rate_diff=(cfg.mode == "portfolio"))


def build_notifier(cfg: Config):
    if cfg.telegram.bot_token and cfg.telegram.chat_id:
        return TelegramNotifier(cfg.telegram.bot_token, cfg.telegram.chat_id)
    return ConsoleNotifier()


def build_runner(cfg: Config, storage: Storage):
    provider = build_provider(cfg)
    notifier = build_notifier(cfg)
    common = dict(
        instruments=cfg.instruments,
        granularity=cfg.granularity,
        history_bars=cfg.history_bars,
        notifier=notifier,
    )
    if cfg.mode == "portfolio":
        rates = None
        if cfg.use_fred_rates:
            from forex.data.rates import FredRates

            rates = FredRates()
        return PortfolioLiveRunner(
            provider,
            build_signal(cfg.signal),
            storage,
            quantile=cfg.quantile,
            rates=rates,
            vol_target=cfg.risk.get("target_ann_vol"),
            vol_window=cfg.risk.get("vol_window", 60),
            max_leverage=cfg.risk.get("max_leverage", 3.0),
            **common,
        )
    return LiveRunner(provider, build_strategy(cfg.strategy), storage, **common)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config YAML")
    parser.add_argument("--once", action="store_true", help="run one cycle (default)")
    parser.add_argument("--schedule", action="store_true", help="run the scheduler loop")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    storage = Storage(cfg.db_path)
    runner = build_runner(cfg, storage)

    if args.schedule:
        runner.start(cfg.schedule_cron)
    else:
        result = runner.run_once()
        if cfg.mode == "portfolio":
            print("\nrebalance emitted" if result else "\nbook unchanged")
        else:
            print(f"\n{len(result)} signal(s) emitted this cycle.")
        storage.close()


if __name__ == "__main__":
    main()
