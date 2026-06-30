#!/usr/bin/env python3
"""Run the live signal runner.

By default it does ONE polling cycle and exits (safe to wire to cron). Pass
``--schedule`` to start the in-process APScheduler loop instead (what the
systemd service runs).

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
from forex.live import ConsoleNotifier, LiveRunner, TelegramNotifier
from forex.strategies import TimeSeriesMomentum


def build_provider(cfg: Config):
    if cfg.data_source == "oanda":
        from forex.data.oanda import OandaDataProvider

        if not cfg.oanda.access_token:
            raise SystemExit("OANDA selected but OANDA_ACCESS_TOKEN is not set.")
        return OandaDataProvider(cfg.oanda.access_token, cfg.oanda.environment)
    return SyntheticDataProvider()


def build_notifier(cfg: Config):
    if cfg.telegram.bot_token and cfg.telegram.chat_id:
        return TelegramNotifier(cfg.telegram.bot_token, cfg.telegram.chat_id)
    return ConsoleNotifier()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config YAML")
    parser.add_argument("--once", action="store_true", help="run one cycle (default)")
    parser.add_argument("--schedule", action="store_true", help="run the scheduler loop")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    provider = build_provider(cfg)
    notifier = build_notifier(cfg)
    storage = Storage(cfg.db_path)
    strategy = TimeSeriesMomentum(lookback=60)

    runner = LiveRunner(
        provider,
        strategy,
        storage,
        instruments=cfg.instruments,
        granularity=cfg.granularity,
        history_bars=cfg.history_bars,
        notifier=notifier,
    )

    if args.schedule:
        runner.start()
    else:
        emitted = runner.run_once()
        print(f"\n{len(emitted)} signal(s) emitted this cycle.")
        storage.close()


if __name__ == "__main__":
    main()
