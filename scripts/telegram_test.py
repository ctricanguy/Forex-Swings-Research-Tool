#!/usr/bin/env python3
"""Send a heartbeat to your Telegram chat to verify the wiring.

    export TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=...
    python scripts/telegram_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forex.live import TelegramNotifier


def main() -> int:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("FAIL: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set")
        return 2
    TelegramNotifier(token, chat_id).send(
        "Forex tool heartbeat — Telegram wiring works."
    )
    print("sent — check your Telegram")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
