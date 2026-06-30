"""Strategy interface -- the contract that guarantees backtest/live parity.

A strategy maps a canonical bar frame to a **desired position** series. The
single rule every strategy must obey:

    The position at row ``t`` may depend only on data at rows ``<= t``.

That is *point-in-time correctness*. Note what it does **not** include: the
one-bar execution shift (you decide on bar ``t``'s close, you are filled on
``t+1``). That shift is applied once, centrally, by the backtester and the live
runner -- never inside a strategy. Keeping it in exactly one place is what makes
"what you backtest is what runs live" literally true.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Base class for all strategies.

    Subclasses implement :meth:`generate_signals` and set :attr:`name`.
    """

    #: Stable identifier used in persisted signals and reports.
    name: str = "base"

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a frame indexed like ``data`` with a ``position`` column.

        ``position`` is in ``[-1, 1]`` (-1 short, 0 flat, +1 long; fractional
        values allowed for sized exposure). Implementations MUST be
        point-in-time: appending future rows to ``data`` must not change the
        positions computed for earlier rows. The test suite enforces this.
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        params = ", ".join(
            f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_")
        )
        return f"{type(self).__name__}({params})"


def positions_frame(index: pd.Index, position: pd.Series) -> pd.DataFrame:
    """Helper to build the standard return frame for a strategy."""
    return pd.DataFrame({"position": position.astype(float)}, index=index)
