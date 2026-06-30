"""Signal domain model.

A :class:`Signal` is the unit emitted by the live runner and persisted for
later outcome tracking. It is deliberately decoupled from any particular
strategy or broker so the same object flows through Telegram alerts, the SQLite
store, and (optionally, later) an execution layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Optional


class Direction(IntEnum):
    """Desired position direction. Integer values double as the position sign."""

    SHORT = -1
    FLAT = 0
    LONG = 1

    @classmethod
    def from_position(cls, position: float) -> "Direction":
        if position > 0:
            return cls.LONG
        if position < 0:
            return cls.SHORT
        return cls.FLAT

    @property
    def label(self) -> str:
        return self.name.capitalize()


@dataclass(slots=True)
class Signal:
    """A point-in-time trading signal derived from a *completed* bar.

    ``bar_time`` is the timestamp of the bar whose close produced the signal,
    not the wall-clock time the alert was sent. This is what makes signals
    reproducible: re-running the strategy over history must reproduce the same
    ``(bar_time, instrument, direction)`` triples.
    """

    bar_time: datetime
    instrument: str
    direction: Direction
    strategy: str
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    rationale: str = ""
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def format_message(self) -> str:
        """Human-readable one-liner for a Telegram / console alert."""
        bits = [
            f"[{self.strategy}] {self.instrument} -> {self.direction.label}",
            f"bar={self.bar_time:%Y-%m-%d %H:%M UTC}",
        ]
        if self.entry is not None:
            bits.append(f"entry={self.entry:.5f}")
        if self.stop_loss is not None:
            bits.append(f"SL={self.stop_loss:.5f}")
        if self.take_profit is not None:
            bits.append(f"TP={self.take_profit:.5f}")
        line = " | ".join(bits)
        if self.rationale:
            line += f"\n{self.rationale}"
        return line
