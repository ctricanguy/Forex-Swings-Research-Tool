import pandas as pd
import pytest

from forex.data import SyntheticDataProvider, validate_bars
from forex.data.base import BAR_COLUMNS


def test_synthetic_provider_schema():
    provider = SyntheticDataProvider()
    df = provider.get_candles("EUR_USD", "D", count=120)
    assert len(df) == 120
    for col in BAR_COLUMNS:
        assert col in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None
    assert df.index.is_monotonic_increasing
    # ask is always above bid (positive spread)
    assert (df["ask"] >= df["bid"]).all()


def test_synthetic_reproducible():
    a = SyntheticDataProvider(seed=42).get_candles("EUR_USD", "D", count=50)
    b = SyntheticDataProvider(seed=42).get_candles("EUR_USD", "D", count=50)
    pd.testing.assert_frame_equal(a, b)


def test_complete_flag_marks_last_bar_incomplete():
    provider = SyntheticDataProvider()
    df = provider.get_candles("EUR_USD", "D", count=30, complete_only=False)
    assert not bool(df["complete"].iloc[-1])
    assert df["complete"].iloc[:-1].all()


def test_validate_bars_rejects_naive_index():
    df = pd.DataFrame(
        {c: [1.0] for c in BAR_COLUMNS},
        index=pd.DatetimeIndex(["2020-01-01"]),  # tz-naive
    )
    with pytest.raises(ValueError):
        validate_bars(df)


def test_validate_bars_rejects_missing_column():
    idx = pd.DatetimeIndex(["2020-01-01"], tz="UTC")
    df = pd.DataFrame({"open": [1.0]}, index=idx)
    with pytest.raises(ValueError):
        validate_bars(df)
