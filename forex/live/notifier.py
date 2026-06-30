"""Notifiers -- where signals are delivered.

Two implementations:

* :class:`ConsoleNotifier` -- always works, used in tests and dry runs.
* :class:`TelegramNotifier` -- pushes to a chat via the Telegram Bot API using
  only the standard library (``urllib``), so no extra dependency is needed just
  to send a message. For an interactive command bot (``/status``, ``/signals``)
  add ``python-telegram-bot`` on top; for one-way alerts this is enough.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from typing import List

from forex.signals import Signal


class Notifier(ABC):
    @abstractmethod
    def send(self, message: str) -> None:
        raise NotImplementedError

    def send_signal(self, signal: Signal) -> None:
        self.send(signal.format_message())


class ConsoleNotifier(Notifier):
    """Print messages; collect them too so tests can assert on output."""

    def __init__(self) -> None:
        self.messages: List[str] = []

    def send(self, message: str) -> None:
        self.messages.append(message)
        print(message)


class TelegramNotifier(Notifier):
    def __init__(self, bot_token: str, chat_id: str, timeout: float = 10.0) -> None:
        if not bot_token or not chat_id:
            raise ValueError("TelegramNotifier requires bot_token and chat_id")
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.timeout = timeout

    def send(self, message: str) -> None:
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = urllib.parse.urlencode(
            {"chat_id": self.chat_id, "text": message}
        ).encode("utf-8")
        request = urllib.request.Request(url, data=payload)
        with urllib.request.urlopen(request, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        if not body.get("ok", False):
            raise RuntimeError(f"Telegram API error: {body}")
