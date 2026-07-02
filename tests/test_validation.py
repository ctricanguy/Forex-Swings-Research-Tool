import pandas as pd

from forex.backtest import (
    CostModel,
    StrategyBacktest,
    expand_grid,
    graduation_verdict,
    holdout_split,
    strategy_factory,
    walk_forward,
)
from forex.data import SyntheticDataProvider
from forex.strategies import TimeSeriesMomentum


def _data(n=1200):
    return SyntheticDataProvider().get_candles("EUR_USD", "D", count=n)


def _factory(data):
    return strategy_factory(TimeSeriesMomentum, data, CostModel(spread_fill=0.0))


def test_walk_forward_oos_is_contiguous_and_in_range():
    data = _data()
    factory = _factory(data)
    configs = expand_grid({"lookback": [20, 40, 60]})
    wf = walk_forward(factory, configs, train=250, test=60)

    n = len(data)
    expected_folds = (n - 250) // 60
    assert len(wf.folds) == expected_folds
    # Stitched OOS starts right after the first train window and is sorted.
    assert wf.oos_returns.index.is_monotonic_increasing
    assert not wf.oos_returns.index.has_duplicates
    assert wf.oos_returns.index.min() == data.index[250]


def test_walk_forward_selects_from_grid():
    data = _data()
    factory = _factory(data)
    configs = expand_grid({"lookback": [20, 40, 60]})
    wf = walk_forward(factory, configs, train=250, test=60)
    allowed = {20, 40, 60}
    for fold in wf.folds:
        assert fold.best_config["lookback"] in allowed


def test_walk_forward_reports_deflation():
    data = _data()
    factory = _factory(data)
    configs = expand_grid({"lookback": [20, 40, 60, 90]})
    wf = walk_forward(factory, configs, train=250, test=60)
    assert wf.n_trials == 4
    # PSR and DSR are probabilities; DSR can't exceed PSR (deflation only lowers).
    assert 0.0 <= wf.psr <= 1.0
    assert 0.0 <= wf.deflated_sharpe <= 1.0
    assert wf.deflated_sharpe <= wf.psr + 1e-9


def test_walk_forward_raises_when_too_short():
    data = _data(n=200)
    factory = _factory(data)
    try:
        walk_forward(factory, [{"lookback": 20}], train=250, test=60)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_holdout_split_partitions_timeline():
    data = _data()
    bt = StrategyBacktest(TimeSeriesMomentum(lookback=40), data, CostModel())
    ho = holdout_split(bt, in_sample=0.7)
    n = len(bt.run().returns)
    assert len(ho.in_sample.returns) + len(ho.out_of_sample.returns) == n
    assert ho.in_sample.returns.index.max() < ho.out_of_sample.returns.index.min()


def test_embargo_purges_gap_between_train_and_test():
    data = _data(n=1000)
    factory = _factory(data)
    configs = expand_grid({"lookback": [20, 40]})
    embargo = 10
    wf = walk_forward(factory, configs, train=250, test=60, embargo=embargo)

    assert len(wf.folds) == (1000 - 250 - embargo) // 60
    idx = data.index
    for fold in wf.folds:
        gap = idx.get_loc(fold.test_start) - idx.get_loc(fold.train_end)
        assert gap == embargo + 1  # embargo bars strictly between the windows
    # First OOS bar sits after the first embargo gap...
    assert wf.oos_returns.index.min() == idx[250 + embargo]
    # ...and the stitched stream is still contiguous (consecutive folds' test
    # windows cover the later embargo bars).
    assert wf.oos_returns.index.is_monotonic_increasing
    positions = [idx.get_loc(t) for t in wf.oos_returns.index]
    assert positions == list(range(positions[0], positions[0] + len(positions)))


def test_embargo_counts_toward_length_check():
    data = _data(n=320)
    factory = _factory(data)
    try:
        walk_forward(factory, [{"lookback": 20}], train=250, test=60, embargo=20)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_trial_sharpes_exported():
    data = _data()
    wf = walk_forward(
        _factory(data), expand_grid({"lookback": [20, 40, 60]}), train=250, test=60
    )
    assert len(wf.trial_sharpes) == 3
    assert wf.deflation_scope == "per-sleeve"


def test_redeflate_project_wide_is_stricter():
    data = _data()
    wf = walk_forward(
        _factory(data), expand_grid({"lookback": [20, 40, 60]}), train=250, test=60
    )
    wf.redeflate(30, sr_trials_std=0.05)
    dsr_30 = wf.deflated_sharpe
    assert wf.deflation_scope == "project-wide"
    assert wf.n_trials == 30
    assert "project-wide" in wf.summary()
    # More trials at the same dispersion -> higher benchmark -> lower DSR.
    wf.redeflate(300, sr_trials_std=0.05)
    assert wf.deflated_sharpe <= dsr_30 + 1e-12


def test_graduation_verdict_strings():
    data = _data()
    wf = walk_forward(
        _factory(data), expand_grid({"lookback": [20, 40, 60]}), train=250, test=60
    )
    verdict = graduation_verdict(wf)
    assert verdict.startswith(("PASS", "WEAK", "FAIL"))
