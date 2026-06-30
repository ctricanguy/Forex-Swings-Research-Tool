import numpy as np
import pandas as pd

from forex.backtest import CostModel, run_backtest
from forex.data import SyntheticDataProvider
from forex.strategies import TimeSeriesMomentum
from forex.strategies.base import Strategy, positions_frame


def _data(n=400, **kw):
    return SyntheticDataProvider(**kw).get_candles("EUR_USD", "D", count=n)


class _AlwaysLong(Strategy):
    name = "always_long"

    def generate_signals(self, data):
        return positions_frame(data.index, pd.Series(1.0, index=data.index))


def test_engine_applies_one_bar_shift():
    """The held position must be the desired position shifted by one bar."""
    data = _data()
    strat = TimeSeriesMomentum(lookback=30)
    result = run_backtest(strat, data, costs=CostModel())
    desired = strat.generate_signals(data)["position"]
    expected_held = desired.shift(1).fillna(0.0)
    pd.testing.assert_series_equal(result.position, expected_held, check_names=False)


def test_always_long_matches_buy_and_hold_minus_costs():
    """An always-long, no-cost backtest equals buy-and-hold of the mid price."""
    data = _data()
    result = run_backtest(_AlwaysLong(), data, costs=CostModel(spread_fill=0.0))
    bar_ret = data["close"].pct_change(fill_method=None).fillna(0.0)
    # First bar has no prior position (held=0), so compare from bar 1 onward.
    expected_equity = (1.0 + bar_ret.iloc[1:]).cumprod()
    np.testing.assert_allclose(
        result.equity.iloc[1:].values, expected_equity.values, rtol=1e-9
    )


def test_costs_reduce_returns():
    data = _data()
    strat = TimeSeriesMomentum(lookback=30)
    free = run_backtest(strat, data, costs=CostModel(spread_fill=0.0))
    costly = run_backtest(strat, data, costs=CostModel(spread_fill=1.0))
    assert costly.metrics["total_return"] <= free.metrics["total_return"]


def test_swap_drag_on_held_position():
    data = _data()
    no_swap = run_backtest(
        _AlwaysLong(), data, costs=CostModel(spread_fill=0.0, swap_per_bar=0.0)
    )
    with_drag = run_backtest(
        _AlwaysLong(), data, costs=CostModel(spread_fill=0.0, swap_per_bar=-1e-4)
    )
    assert with_drag.metrics["total_return"] < no_swap.metrics["total_return"]


def test_incomplete_bar_dropped():
    data = SyntheticDataProvider().get_candles(
        "EUR_USD", "D", count=100, complete_only=False
    )
    result = run_backtest(_AlwaysLong(), data, costs=CostModel(spread_fill=0.0))
    # 100 bars in, last one incomplete -> 99 used.
    assert int(result.metrics["bars"]) == 99


def test_metrics_keys_present():
    data = _data()
    result = run_backtest(TimeSeriesMomentum(lookback=30), data)
    for key in (
        "total_return",
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_drawdown",
        "num_trades",
        "exposure",
        "win_rate",
        "profit_factor",
    ):
        assert key in result.metrics
    assert "Sharpe" in result.summary()
