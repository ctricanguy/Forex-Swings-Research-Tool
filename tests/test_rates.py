import pandas as pd

from forex.data.rates import FredRates, parse_fredgraph_csv

FAKE_CSV = {
    "DFF": "DATE,DFF\n2024-01-01,5.33\n2024-01-02,.\n2024-01-03,5.33\n",
    "ECBDFR": "DATE,ECBDFR\n2024-01-01,4.00\n2024-01-03,3.75\n",
}


def fake_fetch(series_id: str) -> str:
    return FAKE_CSV[series_id]


def test_parse_skips_missing_values():
    s = parse_fredgraph_csv(FAKE_CSV["DFF"])
    assert len(s) == 2  # the "." row dropped
    assert s.iloc[0] == 5.33
    assert s.index.tz is not None


def test_rate_diff_alignment_and_lag():
    rates = FredRates(fetch=fake_fetch, lag_bars=1)
    idx = pd.date_range("2024-01-02", periods=4, freq="D", tz="UTC", name="time")
    diff = rates.rate_diff("EUR_USD", idx)
    # Unlagged ffilled diff on 01-03 would be 3.75 - 5.33 = -1.58; with lag 1
    # that lands on 01-04.
    assert abs(diff.loc["2024-01-04"] - (3.75 - 5.33)) < 1e-9
    # Lagged: first bar has no prior value.
    assert pd.isna(diff.iloc[0])


def test_attach_adds_rate_diff_column():
    rates = FredRates(fetch=fake_fetch)
    idx = pd.date_range("2024-01-02", periods=3, freq="D", tz="UTC", name="time")
    df = pd.DataFrame(
        {
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
            "bid": 1.0, "ask": 1.0, "volume": 1.0, "complete": True,
        },
        index=idx,
    )
    out = rates.attach({"EUR_USD": df})
    assert "rate_diff" in out["EUR_USD"].columns
    # Original frame untouched (attach copies).
    assert "rate_diff" not in df.columns


def test_unknown_currency_raises():
    rates = FredRates(fetch=fake_fetch)
    try:
        rates.rate("XXX")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_caching_fetches_once():
    calls = []

    def counting_fetch(series_id):
        calls.append(series_id)
        return FAKE_CSV[series_id]

    rates = FredRates(fetch=counting_fetch)
    rates.rate("USD")
    rates.rate("USD")
    assert calls == ["DFF"]
