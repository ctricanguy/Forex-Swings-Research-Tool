import numpy as np
import pandas as pd

from forex.data import Panel, SyntheticDataProvider

UNIVERSE = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]


def _panel(n=200, **kw):
    provider = SyntheticDataProvider(**kw)
    return Panel.from_provider(provider, UNIVERSE, "D", count=n)


def test_wide_close_shape_and_columns():
    panel = _panel()
    close = panel.close
    assert list(close.columns) == sorted(UNIVERSE)
    assert len(close) == 200
    assert isinstance(close.index, pd.DatetimeIndex)


def test_spread_frac_positive():
    panel = _panel()
    spread = panel.spread_frac()
    assert (spread.dropna() >= 0).all().all()


def test_cross_sectional_dispersion():
    """Different pairs must have genuinely different paths (not just noise),
    otherwise cross-sectional ranking is meaningless."""
    close = _panel().close
    # Pairwise correlation of returns should not be ~1 across the board.
    rets = close.pct_change(fill_method=None).dropna()
    corr = rets.corr().to_numpy()
    off_diag = corr[~np.eye(len(corr), dtype=bool)]
    assert off_diag.mean() < 0.9


def test_field_requires_presence():
    panel = _panel()  # no rate_diff
    assert not panel.has_field("rate_diff")
    try:
        panel.field("rate_diff")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_field_present_with_rate_diff():
    panel = _panel(with_rate_diff=True)
    assert panel.has_field("rate_diff")
    rd = panel.field("rate_diff")
    assert list(rd.columns) == sorted(UNIVERSE)


def test_upto_truncates_all_frames():
    panel = _panel()
    cut = panel.index[100]
    truncated = panel.upto(cut)
    assert truncated.close.index.max() <= cut
    assert len(truncated.close) == 101


def test_synthetic_deterministic_across_instances():
    a = SyntheticDataProvider(seed=3).get_candles("EUR_USD", "D", count=50)
    b = SyntheticDataProvider(seed=3).get_candles("EUR_USD", "D", count=50)
    pd.testing.assert_frame_equal(a, b)
