"""Configuration loading.

Config comes from a YAML file with secrets overridable by environment variables
(so tokens never live in the repo). Load order, lowest to highest precedence:
defaults -> YAML file -> environment variables.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass(slots=True)
class OandaConfig:
    access_token: Optional[str] = None
    account_id: Optional[str] = None
    environment: str = "practice"  # "practice" or "live"


@dataclass(slots=True)
class TelegramConfig:
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None


@dataclass(slots=True)
class Config:
    instruments: List[str] = field(
        default_factory=lambda: ["EUR_USD", "GBP_USD", "USD_JPY"]
    )
    granularity: str = "D"
    history_bars: int = 400
    db_path: str = "forex.db"
    # APScheduler cron fields for the live poll (default: 00:05 UTC daily).
    schedule_cron: str = "5 0 * * *"
    data_source: str = "synthetic"  # "synthetic" or "oanda"
    # "pair" runs a per-pair Strategy per instrument; "portfolio" runs one
    # cross-sectional signal over the whole instrument list as a long/short book.
    mode: str = "pair"
    # Strategy/signal specs consumed by forex.strategies.registry.
    strategy: dict = field(
        default_factory=lambda: {"type": "tsmom", "params": {"lookback": 60}}
    )
    signal: dict = field(
        default_factory=lambda: {"type": "xsmom", "params": {"lookback": 60}}
    )
    quantile: float = 1.0 / 3.0
    # Risk overlay settings, e.g. {target_ann_vol: 0.10, max_drawdown: 0.15}.
    risk: dict = field(default_factory=dict)
    # Attach real rate differentials from FRED before scoring (live only).
    use_fred_rates: bool = False
    oanda: OandaConfig = field(default_factory=OandaConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        data: dict = {}
        if path and Path(path).exists():
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}

        oanda = OandaConfig(**(data.get("oanda") or {}))
        telegram = TelegramConfig(**(data.get("telegram") or {}))
        known = {f.name for f in dataclasses.fields(cls)}
        top = {
            k: v
            for k, v in data.items()
            if k in known and k not in ("oanda", "telegram")
        }
        cfg = cls(oanda=oanda, telegram=telegram, **top)
        cfg._apply_env()
        return cfg

    def _apply_env(self) -> None:
        self.oanda.access_token = os.getenv("OANDA_ACCESS_TOKEN", self.oanda.access_token)
        self.oanda.account_id = os.getenv("OANDA_ACCOUNT_ID", self.oanda.account_id)
        self.oanda.environment = os.getenv("OANDA_ENVIRONMENT", self.oanda.environment)
        self.telegram.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", self.telegram.bot_token)
        self.telegram.chat_id = os.getenv("TELEGRAM_CHAT_ID", self.telegram.chat_id)
        self.db_path = os.getenv("FOREX_DB_PATH", self.db_path)
