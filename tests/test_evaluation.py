import pandas as pd

from forex.backtest import CostModel, run_backtest
from forex.backtest.evaluation import (
    Backtestable,
    StrategyBacktest,
    expand_grid,
    strategy_factory,
)
from forex.data import SyntheticDataProvider
from forex.strategies import TimeSeriesMomentum


def _data(n=600):
    return SyntheticDataProvider().get_candles("EUR_USD", "D", count=n)


def test_strategy_backtest_is_backtestable():
    bt = StrategyBacktest(TimeSeriesMomentum(lookback=30), _data())
    assert isinstance(bt, Backtestable)
    res = bt.run()
    assert isinstance(res.returns, pd.Series)
    assert len(res.returns) == 600


def test_window_run_respects_warmup_not_truncation():
    """A window run must reproduce the full-run returns over the same dates --
    i.e. warm-up history is used, the window isn't re-warmed from scratch."""
    data = _data()
    strat = TimeSeriesMomentum(lookback=60)
    full = run_backtest(strat, data, costs=CostModel())
    idx = full.returns.index
    start, end = idx[300], idx[450]

    windowed = StrategyBacktest(strat, data).run(start, end)
    expected = full.returns[(full.returns.index >= start) & (full.returns.index <= end)]
    pd.testing.assert_series_equal(windowed.returns, expected, check_names=False)
    # Positions over the window match the full run (no lost warm-up).
    expected_pos = full.position[(full.position.index >= start) & (full.position.index <= end)]
    pd.testing.assert_series_equal(windowed.position, expected_pos, check_names=False)


def test_end_bound_excludes_future():
    data = _data()
    strat = TimeSeriesMomentum(lookback=30)
    idx = data.index
    end = idx[400]
    res = StrategyBacktest(strat, data).run(end=end)
    assert res.returns.index.max() <= end


def test_equity_rebased_to_window():
    data = _data()
    res = StrategyBacktest(TimeSeriesMomentum(lookback=30), data).run(data.index[300])
    # First windowed equity point is ~ (1 + first windowed return).
    assert abs(res.equity.iloc[0] - (1.0 + res.returns.iloc[0])) < 1e-12


def test_expand_grid_cartesian():
    configs = expand_grid({"a": [1, 2], "b": [10, 20, 30]})
    assert len(configs) == 6
    assert {"a": 1, "b": 10} in configs
    assert {"a": 2, "b": 30} in configs


def test_strategy_factory_builds_backtestable():
    factory = strategy_factory(TimeSeriesMomentum, _data())
    bt = factory({"lookback": 45})
    assert isinstance(bt, Backtestable)
    assert bt.strategy.lookback == 45
