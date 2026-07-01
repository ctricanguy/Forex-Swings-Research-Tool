import numpy as np
import pandas as pd

from forex.data import Panel, SyntheticDataProvider
from forex.strategies import XSCarry, XSEnsemble, XSMomentum, XSValue

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


def test_xsvalue_is_long_horizon_reversal():
    panel = _panel(n=400)
    score = XSValue(lookback=200).score(panel)
    expected = -(panel.close / panel.close.shift(200) - 1.0)
    pd.testing.assert_frame_equal(score, expected)
    # The pair that fell the most scores highest (cheapest).
    last_returns = (panel.close.iloc[-1] / panel.close.iloc[-201] - 1.0)
    assert score.iloc[-1].idxmax() == last_returns.idxmin()


def test_xsvalue_field_mode():
    panel = _panel(with_rate_diff=True)
    score = XSValue(field="rate_diff").score(panel)
    pd.testing.assert_frame_equal(score, panel.field("rate_diff"))


def test_xsensemble_is_weighted_zscore_average():
    panel = _panel(n=400, with_rate_diff=True)
    members = [XSMomentum(40), XSCarry()]
    ens = XSEnsemble(members, weights=[3, 1])
    combined = ens.score(panel)

    def zscore(df):
        mu = df.mean(axis=1)
        sd = df.std(axis=1, ddof=1).replace(0.0, np.nan)
        return df.sub(mu, axis=0).div(sd, axis=0)

    expected = 0.75 * zscore(members[0].score(panel)) + 0.25 * zscore(
        members[1].score(panel)
    )
    pd.testing.assert_frame_equal(combined, expected)


def test_xsensemble_point_in_time():
    full = _panel(n=300, with_rate_diff=True)
    ens = XSEnsemble([XSMomentum(40), XSCarry(), XSValue(100)])
    score_full = ens.score(full).iloc[:200]
    score_trunc = ens.score(full.upto(full.index[199]))
    pd.testing.assert_frame_equal(score_full, score_trunc)
