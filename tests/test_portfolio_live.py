import pandas as pd

from forex.data import Panel, SyntheticDataProvider
from forex.data.storage import Storage
from forex.live import ConsoleNotifier, PortfolioLiveRunner, format_book_message
from forex.strategies import XSMomentum

UNIVERSE = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "NZD_USD"]


def _runner(storage, **kw):
    return PortfolioLiveRunner(
        SyntheticDataProvider(),
        XSMomentum(lookback=40),
        storage,
        instruments=UNIVERSE,
        granularity="D",
        history_bars=200,
        notifier=ConsoleNotifier(),
        **kw,
    )


def test_run_once_emits_and_persists_book(tmp_path):
    storage = Storage(tmp_path / "t.db")
    runner = _runner(storage)
    book = runner.run_once()
    assert book is not None
    # Dollar-neutral within tolerance; something is actually held.
    assert abs(sum(book.values())) < 1e-9
    assert sum(abs(w) for w in book.values()) > 0.5
    stored = storage.last_book("xsmom_40")
    assert stored is not None
    assert len(runner.notifier.messages) == 1
    assert "not investment advice" in runner.notifier.messages[0].lower()
    storage.close()


def test_run_once_quiet_when_book_unchanged(tmp_path):
    storage = Storage(tmp_path / "t.db")
    runner = _runner(storage)
    first = runner.run_once()
    assert first is not None
    second = runner.run_once()  # same synthetic bars -> same book
    assert second is None
    assert len(runner.notifier.messages) == 1
    storage.close()


def test_vol_target_scales_book(tmp_path):
    storage_a = Storage(tmp_path / "a.db")
    storage_b = Storage(tmp_path / "b.db")
    plain = _runner(storage_a).run_once()
    # A tiny vol target must shrink gross exposure vs the unscaled book.
    scaled = _runner(storage_b, vol_target=0.005).run_once()
    gross_plain = sum(abs(w) for w in plain.values())
    gross_scaled = sum(abs(w) for w in scaled.values())
    assert gross_scaled < gross_plain
    storage_a.close()
    storage_b.close()


def test_format_book_message_turnover():
    when = pd.Timestamp("2024-03-01", tz="UTC")
    msg = format_book_message(
        "xs", when, {"EUR_USD": 0.5, "USD_JPY": -0.5}, {"EUR_USD": 0.5, "USD_JPY": 0.5}
    )
    assert "EUR_USD" in msg and "turnover" in msg


def test_bars_persisted_for_universe(tmp_path):
    storage = Storage(tmp_path / "t.db")
    _runner(storage).run_once()
    for instrument in UNIVERSE:
        assert len(storage.load_bars(instrument, "D")) == 200
    storage.close()
