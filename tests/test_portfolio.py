import numpy as np
import pandas as pd

from forex.backtest import (
    Backtestable,
    CostModel,
    PortfolioBacktest,
    expand_grid,
    portfolio_factory,
    rank_weights,
    walk_forward,
)
from forex.data import Panel, SyntheticDataProvider
from forex.strategies import XSMomentum

UNIVERSE = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "NZD_USD"]


def _panel(n=800, **kw):
    return Panel.from_provider(SyntheticDataProvider(**kw), UNIVERSE, "D", count=n)


# -- rank_weights ---------------------------------------------------------

def test_rank_weights_long_top_short_bottom():
    cols = ["A", "B", "C", "D"]
    scores = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], columns=cols)
    w = rank_weights(scores, quantile=1 / 3, gross_leverage=1.0).iloc[0]
    assert w["D"] == 0.5  # highest score -> long
    assert w["A"] == -0.5  # lowest score -> short
    assert w["B"] == 0.0 and w["C"] == 0.0


def test_rank_weights_dollar_neutral_and_gross():
    cols = list("ABCDEF")
    scores = pd.DataFrame([[1, 2, 3, 4, 5, 6]], columns=cols, dtype=float)
    w = rank_weights(scores, quantile=1 / 3, gross_leverage=1.0).iloc[0]
    assert abs(w.sum()) < 1e-12  # dollar neutral
    assert abs(w.abs().sum() - 1.0) < 1e-12  # gross == leverage


def test_rank_weights_skips_nan():
    cols = ["A", "B", "C", "D"]
    scores = pd.DataFrame([[1.0, np.nan, 3.0, 4.0]], columns=cols)
    w = rank_weights(scores, quantile=1 / 3).iloc[0]
    assert w["B"] == 0.0  # NaN never gets weight
    assert w["D"] == 0.5 and w["A"] == -0.5


def test_rank_weights_flat_when_too_few():
    scores = pd.DataFrame([[1.0]], columns=["A"])
    w = rank_weights(scores).iloc[0]
    assert (w == 0).all()


# -- PortfolioBacktest ----------------------------------------------------

def test_portfolio_is_backtestable():
    bt = PortfolioBacktest(XSMomentum(lookback=40), _panel())
    assert isinstance(bt, Backtestable)
    res = bt.run()
    assert isinstance(res.returns, pd.Series)
    assert len(res.returns) == 800


def test_portfolio_first_bar_has_no_position():
    res = PortfolioBacktest(XSMomentum(lookback=40), _panel()).run()
    # Execution shift means nothing is held on the first bar.
    assert res.position.iloc[0] == 0.0
    assert res.returns.iloc[0] == 0.0


def test_portfolio_costs_reduce_returns():
    panel = _panel()
    free = PortfolioBacktest(
        XSMomentum(40), panel, costs=CostModel(spread_fill=0.0)
    ).run()
    costly = PortfolioBacktest(
        XSMomentum(40), panel, costs=CostModel(spread_fill=1.0)
    ).run()
    assert costly.metrics["total_return"] <= free.metrics["total_return"]


def test_portfolio_window_respects_warmup():
    panel = _panel()
    bt = PortfolioBacktest(XSMomentum(60), panel, costs=CostModel(spread_fill=0.0))
    full = bt.run()
    idx = full.returns.index
    start, end = idx[400], idx[600]
    windowed = bt.run(start, end)
    expected = full.returns[(idx >= start) & (idx <= end)]
    pd.testing.assert_series_equal(windowed.returns, expected, check_names=False)


# -- the payoff: same harness, no changes ---------------------------------

def test_portfolio_plugs_into_walk_forward():
    panel = _panel(n=1000)
    factory = portfolio_factory(
        XSMomentum, panel, CostModel(spread_fill=0.0), quantile=1 / 3
    )
    configs = expand_grid({"lookback": [20, 40, 60, 90]})
    wf = walk_forward(factory, configs, train=250, test=60)

    expected_folds = (1000 - 250) // 60
    assert len(wf.folds) == expected_folds
    assert wf.n_trials == 4
    for fold in wf.folds:
        assert fold.best_config["lookback"] in {20, 40, 60, 90}
    assert 0.0 <= wf.deflated_sharpe <= 1.0
