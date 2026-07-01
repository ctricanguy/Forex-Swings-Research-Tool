import json

import numpy as np
import pandas as pd
import pytest

from forex.data import Panel
from forex.data.events import load_event_dates, save_event_dates
from forex.data.positioning import OandaPositioning, attach_positioning
from forex.data.rates import parse_fred_api_json
from forex.data.storage import Storage
from forex.strategies import EventGate, XSMomentum, XSPositioning


def _frame(n=100, price=1.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC", name="time")
    return pd.DataFrame(
        {
            "open": price, "high": price, "low": price, "close": price,
            "bid": price - 0.0001, "ask": price + 0.0001,
            "volume": 100.0, "complete": True,
        },
        index=idx,
    )


# -- EventGate ---------------------------------------------------------------

def test_event_gate_avoid_blanks_window():
    panel = Panel({"AAA_USD": _frame(), "BBB_USD": _frame(price=2.0)})
    event = pd.Timestamp("2024-02-15", tz="UTC")
    gated = EventGate(XSMomentum(10), [event], before=1, after=1).score(panel)
    raw = XSMomentum(10).score(panel)
    for day in ("2024-02-14", "2024-02-15", "2024-02-16"):
        assert gated.loc[day].isna().all().all()
    # Outside the window: untouched.
    pd.testing.assert_frame_equal(
        gated.loc["2024-02-20":], raw.loc["2024-02-20":]
    )


def test_event_gate_only_mode_is_complement():
    panel = Panel({"AAA_USD": _frame(), "BBB_USD": _frame(price=2.0)})
    event = pd.Timestamp("2024-02-15", tz="UTC")
    only = EventGate(XSMomentum(10), [event], before=0, after=0, mode="only").score(panel)
    assert not only.loc["2024-02-15"].isna().all().all()
    assert only.loc["2024-02-20"].isna().all().all()


def test_event_gate_rejects_bad_mode():
    with pytest.raises(ValueError):
        EventGate(XSMomentum(10), [], mode="sometimes")


def test_event_calendar_roundtrip(tmp_path):
    dates = [pd.Timestamp("2024-03-12"), pd.Timestamp("2024-04-10")]
    save_event_dates(dates, tmp_path / "cpi.csv")
    save_event_dates([pd.Timestamp("2024-03-20")], tmp_path / "fomc.csv")
    loaded = load_event_dates(tmp_path)  # directory: merges all CSVs
    assert len(loaded) == 3
    assert loaded.min() == pd.Timestamp("2024-03-12", tz="UTC")


# -- positioning ---------------------------------------------------------------

def _fake_book(time="2024-01-05T21:00:00Z", long_total=70.0):
    buckets = [
        {"price": "1.10", "longCountPercent": str(long_total / 2), "shortCountPercent": "10"},
        {"price": "1.11", "longCountPercent": str(long_total / 2), "shortCountPercent": str(90 - long_total)},
    ]
    return json.dumps({"positionBook": {"instrument": "EUR_USD", "time": time, "buckets": buckets}})


def test_snapshot_aggregates_buckets():
    client = OandaPositioning("tok", fetch=lambda url: _fake_book(long_total=70.0))
    snap = client.snapshot("EUR_USD")
    assert abs(snap["long_pct"] - 70.0) < 1e-9
    assert abs(snap["short_pct"] - 30.0) < 1e-9


def test_snapshot_none_on_error():
    def failing(url):
        raise OSError("404")

    assert OandaPositioning("tok", fetch=failing).snapshot("EUR_USD") is None


def test_attach_and_contrarian_score(tmp_path):
    storage = Storage(tmp_path / "t.db")
    frame = _frame(n=10)
    for ts in frame.index[3:]:
        storage.save_positioning("AAA_USD", ts, 75.0, 25.0)   # crowd long
        storage.save_positioning("BBB_USD", ts, 30.0, 70.0)   # crowd short
    frames = attach_positioning(storage, {"AAA_USD": frame, "BBB_USD": frame.copy()})
    panel = Panel(frames)
    score = XSPositioning().score(panel)
    # No coverage on early bars.
    assert np.isnan(score["AAA_USD"].iloc[0])
    # Contrarian: crowd-long pair scores negative, crowd-short positive.
    assert score["AAA_USD"].iloc[-1] == -25.0
    assert score["BBB_USD"].iloc[-1] == 20.0
    storage.close()


def test_xspositioning_requires_field():
    panel = Panel({"AAA_USD": _frame()})
    with pytest.raises(ValueError):
        XSPositioning().score(panel)


# -- FRED API json ---------------------------------------------------------------

def test_parse_fred_api_json():
    body = json.dumps({
        "observations": [
            {"date": "2024-01-01", "value": "5.33"},
            {"date": "2024-01-02", "value": "."},
            {"date": "2024-01-03", "value": "5.25"},
        ]
    })
    s = parse_fred_api_json(body)
    assert len(s) == 2
    assert s.iloc[-1] == 5.25


def test_fred_disk_cache_roundtrip(tmp_path):
    from forex.data.rates import FredRates

    calls = []

    def fetch(series_id):
        calls.append(series_id)
        return "DATE,DFF\n2024-01-01,5.33\n2024-01-02,5.30\n"

    a = FredRates(fetch=fetch, cache_dir=tmp_path)
    a.rate("USD")
    assert calls == ["DFF"]
    # Second instance reads the disk cache; no new fetch.
    b = FredRates(fetch=fetch, cache_dir=tmp_path)
    s = b.rate("USD")
    assert calls == ["DFF"]
    assert len(s) == 2 and s.iloc[0] == 5.33
