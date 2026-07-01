from datetime import datetime, timezone

import pytest

from forex import tracking
from forex.data.storage import Storage
from forex.signals import Direction, Signal
from forex.strategies import Ensemble, TimeSeriesMomentum, XSEnsemble, XSMomentum
from forex.strategies.registry import build_signal, build_strategy


# -- registry ---------------------------------------------------------------

def test_build_pair_strategy():
    strat = build_strategy({"type": "tsmom", "params": {"lookback": 90}})
    assert isinstance(strat, TimeSeriesMomentum)
    assert strat.lookback == 90


def test_build_pair_ensemble():
    strat = build_strategy(
        {
            "type": "ensemble",
            "members": [
                {"type": "tsmom", "params": {"lookback": 30}},
                {"type": "ma_cross", "params": {"fast": 10, "slow": 40}},
            ],
            "weights": [2, 1],
        }
    )
    assert isinstance(strat, Ensemble)
    assert len(strat.strategies) == 2


def test_build_xs_signal_and_ensemble():
    sig = build_signal({"type": "xsmom", "params": {"lookback": 40}})
    assert isinstance(sig, XSMomentum)
    ens = build_signal(
        {
            "type": "xs_ensemble",
            "members": [{"type": "xsmom"}, {"type": "xsvalue", "params": {"lookback": 500}}],
        }
    )
    assert isinstance(ens, XSEnsemble)


def test_unknown_types_raise():
    with pytest.raises(KeyError):
        build_strategy({"type": "nope"})
    with pytest.raises(KeyError):
        build_signal({"type": "nope"})


# -- tracking ---------------------------------------------------------------

def _signal(bar_day, direction, entry, strategy="tsmom_60", instrument="EUR_USD"):
    return Signal(
        bar_time=datetime(2024, 1, bar_day, tzinfo=timezone.utc),
        instrument=instrument,
        direction=direction,
        strategy=strategy,
        entry=entry,
    )


def test_outcomes_marked_from_signal_spans(tmp_path):
    db = Storage(tmp_path / "t.db")
    db.save_signal(_signal(1, Direction.LONG, 1.00))
    db.save_signal(_signal(10, Direction.SHORT, 1.10))
    db.save_signal(_signal(20, Direction.FLAT, 1.05))

    closed = tracking.update_outcomes(db)
    assert closed == 2

    out = db.outcomes()
    assert len(out) == 2
    # LONG 1.00 -> 1.10 = +10%; SHORT 1.10 -> 1.05 = +4.55%
    assert abs(out["pnl"].iloc[0] - 0.10) < 1e-9
    assert abs(out["pnl"].iloc[1] - (-1 * (1.05 - 1.10) / 1.10)) < 1e-9

    # Idempotent re-run.
    assert tracking.update_outcomes(db) == 2
    assert len(db.outcomes()) == 2
    db.close()


def test_open_position_reported_not_marked(tmp_path):
    db = Storage(tmp_path / "t.db")
    db.save_signal(_signal(1, Direction.LONG, 1.00))
    assert tracking.update_outcomes(db) == 0  # nothing closed yet
    report = tracking.track_record(db)
    assert "Open positions" in report
    assert "LONG" in report
    db.close()


def test_track_record_summary(tmp_path):
    db = Storage(tmp_path / "t.db")
    db.save_signal(_signal(1, Direction.LONG, 1.00))
    db.save_signal(_signal(10, Direction.FLAT, 1.02))
    tracking.update_outcomes(db)
    report = tracking.track_record(db)
    assert "1 closed" in report
    assert "win rate 100%" in report
    db.close()


# -- books storage ------------------------------------------------------------

def test_book_roundtrip(tmp_path):
    db = Storage(tmp_path / "t.db")
    when = datetime(2024, 3, 1, tzinfo=timezone.utc)
    db.save_book("xsmom_60", when, {"EUR_USD": 0.25, "USD_JPY": -0.25})
    got = db.last_book("xsmom_60")
    assert got is not None
    bar_time, weights = got
    assert bar_time == when
    assert weights == {"EUR_USD": 0.25, "USD_JPY": -0.25}
    assert db.last_book("other") is None
    db.close()
