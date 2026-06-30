"""Data ingestion layer.

The rest of the package only ever sees the canonical bar DataFrame defined in
:mod:`forex.data.base` (UTC index; ``open/high/low/close`` mid prices plus
``bid/ask`` closes, ``volume`` and ``complete``). Concrete providers
(:class:`~forex.data.synthetic.SyntheticDataProvider`,
:class:`~forex.data.oanda.OandaDataProvider`) are responsible for mapping their
native format onto that schema, so strategies and the backtester never depend
on a specific vendor.
"""

from forex.data.base import BAR_COLUMNS, Bar, DataProvider, validate_bars
from forex.data.panel import Panel
from forex.data.synthetic import SyntheticDataProvider

__all__ = [
    "BAR_COLUMNS",
    "Bar",
    "DataProvider",
    "validate_bars",
    "Panel",
    "SyntheticDataProvider",
]
