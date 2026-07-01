"""Shared strategy library.

This package is the heart of the system. The backtester and the live runner
both import strategies from here, so a strategy is written and validated once
and runs identically in both contexts.
"""

from forex.strategies.base import Strategy, positions_frame
from forex.strategies.carry import Carry
from forex.strategies.cross_sectional import (
    CrossSectionalSignal,
    XSCarry,
    XSEnsemble,
    XSMomentum,
    XSValue,
)
from forex.strategies.ensemble import Ensemble
from forex.strategies.ma_cross import MovingAverageCross
from forex.strategies.momentum import TimeSeriesMomentum

__all__ = [
    "Strategy",
    "positions_frame",
    "Carry",
    "Ensemble",
    "MovingAverageCross",
    "TimeSeriesMomentum",
    "CrossSectionalSignal",
    "XSCarry",
    "XSEnsemble",
    "XSMomentum",
    "XSValue",
]
