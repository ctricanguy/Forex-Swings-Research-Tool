"""Interactive Telegram command bot (optional; requires python-telegram-bot).

One-way alerts need no dependency (see :mod:`forex.live.notifier`); this module
adds the service-facing interactive layer:

* ``/status``  -- database heartbeat: bars stored, latest signal, latest book
* ``/signals`` -- recent signals
* ``/book``    -- current portfolio book
* ``/track``   -- the auditable track record

Run with ``python -m forex.live.bot`` (reads config/env like the runners).
"""

from __future__ import annotations

from forex import tracking
from forex.data.storage import Storage
from forex.live.portfolio_runner import DISCLAIMER


def status_text(storage: Storage) -> str:
    bars = storage._conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
    sigs = storage._conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    books = storage._conn.execute(
        "SELECT COUNT(DISTINCT strategy || bar_time) FROM books"
    ).fetchone()[0]
    return f"bars: {bars} | signals: {sigs} | books: {books}"


def signals_text(storage: Storage, limit: int = 10) -> str:
    df = storage.recent_signals(limit)
    if df.empty:
        return "no signals yet"
    lines = []
    for _, row in df.iterrows():
        side = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(int(row["direction"]), "?")
        lines.append(
            f"{row['bar_time'][:16]} {row['strategy']} {row['instrument']} {side}"
        )
    return "\n".join(lines)


def book_text(storage: Storage, strategy: str) -> str:
    book = storage.last_book(strategy)
    if book is None:
        return f"no book stored for {strategy!r}"
    bar_time, weights = book
    lines = [f"{strategy} @ {bar_time:%Y-%m-%d}"]
    for instrument, weight in sorted(weights.items(), key=lambda kv: -kv[1]):
        if abs(weight) > 1e-9:
            lines.append(f"  {instrument:<9} {weight:+.1%}")
    return "\n".join(lines)


def run_bot(token: str, db_path: str, default_strategy: str) -> None:  # pragma: no cover
    try:
        from telegram.ext import Application, CommandHandler
    except ImportError as exc:
        raise ImportError(
            "python-telegram-bot is required for the interactive bot. "
            "Install it with `pip install python-telegram-bot`."
        ) from exc

    storage = Storage(db_path)

    async def status(update, _):
        await update.message.reply_text(status_text(storage))

    async def signals(update, _):
        await update.message.reply_text(signals_text(storage))

    async def book(update, context):
        name = context.args[0] if context.args else default_strategy
        await update.message.reply_text(book_text(storage, name))

    async def track(update, _):
        tracking.update_outcomes(storage)
        await update.message.reply_text(
            tracking.track_record(storage) + "\n" + DISCLAIMER
        )

    app = Application.builder().token(token).build()
    for name, handler in [
        ("status", status), ("signals", signals), ("book", book), ("track", track)
    ]:
        app.add_handler(CommandHandler(name, handler))
    app.run_polling()


if __name__ == "__main__":  # pragma: no cover
    from forex.config import Config

    cfg = Config.load("config/config.yaml")
    if not cfg.telegram.bot_token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set")
    run_bot(cfg.telegram.bot_token, cfg.db_path, cfg.signal.get("type", "xsmom"))
