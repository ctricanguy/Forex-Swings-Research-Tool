"""Live signal layer.

A thin, auditable runner that, on a schedule, fetches the latest *completed*
bars, calls the **same** ``forex.strategies`` code the backtester uses, and
pushes any change in desired position out as a :class:`~forex.signals.Signal`.
No auto-execution -- this is an alert/research tool by design.
"""

from forex.live.notifier import ConsoleNotifier, Notifier, TelegramNotifier
from forex.live.runner import LiveRunner

__all__ = ["ConsoleNotifier", "Notifier", "TelegramNotifier", "LiveRunner"]
