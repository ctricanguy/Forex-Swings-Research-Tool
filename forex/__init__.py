"""Swing-Forex research and signal tool.

A modular toolkit for researching and running swing (daily / 4-hour) forex
strategies. The defining architectural rule of this package: the **same**
strategy code (``forex.strategies``) is imported by both the backtester
(``forex.backtest``) and the live runner (``forex.live``). What you backtest
is exactly what runs live, which removes the single most common source of
retail system failure -- backtest/live divergence and lookahead bias.

See ``docs/ARCHITECTURE.md`` for the full design and the honest edge landscape.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
