"""Telegram formatters and activity log."""

from bot.learning.journal import TradeJournal
from bot.telegram.activity_log import log_activity, recent_activity
from bot.telegram.formatters import BotSnapshot, format_snapshot


def test_format_snapshot_includes_positions():
    snap = BotSnapshot(
        mode="DRY RUN",
        strategy="CHAMPAIGN",
        balance=500.0,
        available=420.0,
        initial_balance=500.0,
        open_count=1,
        max_positions=10,
    )
    text = format_snapshot(snap)
    assert "CHAMPAIGN" in text
    assert "500.00" in text


def test_activity_log_roundtrip(tmp_path, monkeypatch):
    from bot.telegram import activity_log

    db = tmp_path / "act.db"
    monkeypatch.setattr(activity_log, "DB_PATH", db)
    log_activity("OPEN", "test open", symbol="BTCUSDT")
    rows = recent_activity(5)
    assert len(rows) == 1
    assert rows[0]["symbol"] == "BTCUSDT"


def test_journal_recent_trades(tmp_path):
    j = TradeJournal(tmp_path / "j.db")
    tid = j.log_trade_open("ETHUSDT", "LONG", 100.0, 0.1)
    assert tid > 0
    recent = j.recent_trades(5)
    assert recent[0]["symbol"] == "ETHUSDT"
    assert recent[0]["closed_at"] is None
