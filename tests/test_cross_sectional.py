import pandas as pd

from forex.data import Panel, SyntheticDataProvider
from forex.strategies import XSCarry, XSMomentum

UNIVERSE = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]


def _panel(n=300, **kw):
    return Panel.from_provider(SyntheticDataProvider(**kw), UNIVERSE, "D", count=n)


def test_xsmomentum_score_is_trailing_return():
    panel = _panel()
    sig = XSMomentum(lookback=40)
    score = sig.score(panel)
    expected = panel.close / panel.close.shift(40) - 1.0
    pd.testing.assert_frame_equal(score, expected)


def test_xsmomentum_point_in_time():
    """Appending future bars must not change earlier scores."""
    full = _panel(n=300)
    sig = XSMomentum(lookback=40)
    score_full = sig.score(full).iloc[:200]

    truncated = full.upto(full.index[199])
    score_trunc = sig.score(truncated)
    pd.testing.assert_frame_equal(score_full, score_trunc)


def test_xscarry_requires_rate_diff():
    panel = _panel()  # no rate_diff
    try:
        XSCarry().score(panel)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_xscarry_score_is_rate_diff():
    panel = _panel(with_rate_diff=True)
    score = XSCarry().score(panel)
    pd.testing.assert_frame_equal(score, panel.field("rate_diff"))
