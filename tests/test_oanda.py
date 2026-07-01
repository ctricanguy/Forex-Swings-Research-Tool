"""Offline tests for the REST-based OANDA provider (injected HTTP layer)."""

import json
import urllib.parse

import pandas as pd
import pytest

from forex.data.oanda import OandaDataProvider


def _candle(time, price, complete=True, spread=0.0002):
    half = spread / 2
    def ohlc(px):
        return {"o": str(px), "h": str(px), "l": str(px), "c": str(px)}
    return {
        "time": time,
        "volume": 100,
        "complete": complete,
        "bid": ohlc(round(price - half, 6)),
        "ask": ohlc(round(price + half, 6)),
    }


def _params(url):
    return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))


T = [f"2024-01-0{i}T00:00:00.000000000Z" for i in range(1, 8)]


def _provider(fetch):
    return OandaDataProvider("dummy-token", "practice", fetch=fetch)


def test_parse_schema_and_mid_prices():
    def fetch(url):
        return json.dumps({"candles": [_candle(T[0], 1.1000), _candle(T[1], 1.2000)]})

    df = _provider(fetch).get_candles("EUR_USD", "D", count=2)
    assert list(df.index.tz_convert("UTC")) == [
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-01-02", tz="UTC"),
    ]
    assert abs(df["close"].iloc[0] - 1.1000) < 1e-9  # mid of bid/ask
    assert df["ask"].iloc[0] > df["bid"].iloc[0]
    assert df["complete"].all()


def test_incomplete_tail_dropped_by_default():
    def fetch(url):
        return json.dumps(
            {"candles": [_candle(T[0], 1.1), _candle(T[1], 1.2, complete=False)]}
        )

    provider = _provider(fetch)
    assert len(provider.get_candles("EUR_USD", "D", count=2)) == 1
    assert len(provider.get_candles("EUR_USD", "D", count=2, complete_only=False)) == 2


def test_backward_pagination_assembles_full_count():
    calls = []

    def fetch(url):
        p = _params(url)
        calls.append(p)
        if "to" not in p:  # newest chunk
            return json.dumps({"candles": [_candle(T[2], 1.3), _candle(T[3], 1.4)]})
        # older chunk ending at `to` (boundary candle included, must be deduped)
        return json.dumps(
            {"candles": [_candle(T[0], 1.1), _candle(T[1], 1.2), _candle(T[2], 1.3)]}
        )

    provider = _provider(fetch)
    provider.max_per_request = 2
    df = provider.get_candles("EUR_USD", "D", count=4)
    assert len(df) == 4
    assert df.index.is_monotonic_increasing
    assert not df.index.has_duplicates
    assert len(calls) == 2
    assert "to" in calls[1]


def test_backward_pagination_stops_when_history_exhausted():
    def fetch(url):
        p = _params(url)
        if "to" not in p:
            return json.dumps({"candles": [_candle(T[0], 1.1)]})
        return json.dumps({"candles": [_candle(T[0], 1.1)]})  # only overlap left

    provider = _provider(fetch)
    provider.max_per_request = 1
    df = provider.get_candles("EUR_USD", "D", count=10)
    assert len(df) == 1  # no infinite loop, returns what exists


def test_forward_range_pagination_and_end_filter():
    def fetch(url):
        p = _params(url)
        start = pd.Timestamp(p["from"])
        chunk = [
            _candle(t, 1.0 + i / 10)
            for i, t in enumerate(T[:4])
            if pd.Timestamp(t) >= start
        ][:2]
        return json.dumps({"candles": chunk})

    provider = _provider(fetch)
    provider.max_per_request = 2
    df = provider.get_candles(
        "EUR_USD",
        "D",
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-01-03", tz="UTC"),
    )
    assert len(df) == 3  # T1..T3; T4 filtered by end
    assert df.index.max() == pd.Timestamp("2024-01-03", tz="UTC")


def test_bad_environment_rejected():
    with pytest.raises(ValueError):
        OandaDataProvider("tok", "prod")
