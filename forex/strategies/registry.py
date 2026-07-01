"""Config-driven strategy construction.

Lets the live runner (and any script) build strategies from YAML/dict specs
instead of code edits -- necessary once this runs as a service where changing
the deployed sleeve should be a config change, not a release.

Spec format::

    {"type": "tsmom", "params": {"lookback": 60}}
    {"type": "ensemble", "members": [ ...specs... ], "weights": [2, 1]}
    {"type": "xs_ensemble", "members": [ ...signal specs... ]}
"""

from __future__ import annotations

from typing import Dict

from forex.strategies.base import Strategy
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

PAIR_STRATEGIES = {
    "tsmom": TimeSeriesMomentum,
    "ma_cross": MovingAverageCross,
    "carry": Carry,
}

XS_SIGNALS = {
    "xsmom": XSMomentum,
    "xscarry": XSCarry,
    "xsvalue": XSValue,
}


def build_strategy(spec: Dict) -> Strategy:
    kind = spec.get("type")
    if kind == "ensemble":
        members = [build_strategy(m) for m in spec["members"]]
        return Ensemble(members, spec.get("weights"), name=spec.get("name", "ensemble"))
    if kind in PAIR_STRATEGIES:
        return PAIR_STRATEGIES[kind](**spec.get("params", {}))
    raise KeyError(
        f"unknown strategy type {kind!r}; known: {sorted(PAIR_STRATEGIES) + ['ensemble']}"
    )


def build_signal(spec: Dict) -> CrossSectionalSignal:
    kind = spec.get("type")
    if kind == "xs_ensemble":
        members = [build_signal(m) for m in spec["members"]]
        return XSEnsemble(members, spec.get("weights"), name=spec.get("name", "xs_ensemble"))
    if kind in XS_SIGNALS:
        return XS_SIGNALS[kind](**spec.get("params", {}))
    raise KeyError(
        f"unknown signal type {kind!r}; known: {sorted(XS_SIGNALS) + ['xs_ensemble']}"
    )
