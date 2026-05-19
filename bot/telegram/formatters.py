"""HTML messages for Telegram."""

from __future__ import annotations

import html
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from bot.learning.journal import TradeJournal
from bot.telegram.activity_log import recent_activity


def _e(s: Any) -> str:
    return html.escape(str(s), quote=False)


@dataclass
class PositionLine:
    symbol: str
    side: str
    entry: float
    mark: float
    sl: float
    tp: float
    stage: str = ""
    hedge: bool = False
    unrealized_usdt: float = 0.0
    locked_profit_usdt: float = -1.0


@dataclass
class BotSnapshot:
    mode: str  # DRY RUN | LIVE | TESTNET
    strategy: str  # CHAMPAIGN | SCALPER
    balance: float
    available: float
    initial_balance: float = 0.0
    unrealized_usdt: float = 0.0
    open_count: int = 0
    max_positions: int = 0
    positions: List[PositionLine] = field(default_factory=list)
    paper_closed: int = 0
    paper_wins: int = 0
    paper_costs: float = 0.0
    extra: str = ""


def format_help() -> str:
    return (
        "<b>Команди бота</b>\n"
        "/status — баланс, позиції, PnL\n"
        "/balance — коротко баланс\n"
        "/positions — відкриті позиції\n"
        "/history — останні події та угоди\n"
        "/stats — статистика закритих угод\n"
        "/help — цей список"
    )


def format_snapshot(s: BotSnapshot) -> str:
    eq = s.balance
    cash = eq - s.unrealized_usdt
    lines = [
        f"<b>{_e(s.strategy)}</b> · {_e(s.mode)}",
        f"Equity: <b>{eq:.2f}</b> USDT <i>(як bal у логах)</i>",
        f"Готівка: {cash:.2f} · uPnL {s.unrealized_usdt:+.2f}",
        f"Доступно: {s.available:.2f} USDT",
    ]
    if s.initial_balance > 0:
        pnl = eq - s.initial_balance
        lines.append(f"Старт: {s.initial_balance:.2f} → Δ {pnl:+.2f}")
    lines.append(
        f"Позиції: {s.open_count}/{s.max_positions}"
    )
    if s.paper_closed:
        wr = (s.paper_wins / s.paper_closed * 100) if s.paper_closed else 0
        lines.append(
            f"Paper закрито: {s.paper_closed} · WR {wr:.0f}% · costs {s.paper_costs:.2f}"
        )
    if s.positions:
        lines.append("")
        lines.append("<b>Відкриті</b>")
        for p in s.positions[:12]:
            h = " +hedge" if p.hedge else ""
            if p.locked_profit_usdt >= 0:
                st = f" [lock +{p.locked_profit_usdt:.0f}]"
            elif p.stage:
                st = f" [{_e(p.stage)}]"
            else:
                st = ""
            lines.append(
                f"• <b>{_e(p.symbol)}</b> {_e(p.side)}{h}{st}\n"
                f"  entry {_e(f'{p.entry:.6f}')} mark {_e(f'{p.mark:.6f}')} "
                f"uPnL {p.unrealized_usdt:+.2f}\n"
                f"  SL {_e(f'{p.sl:.6f}')} TP {_e(f'{p.tp:.6f}')}"
            )
        if len(s.positions) > 12:
            lines.append(f"… ще {len(s.positions) - 12}")
    if s.extra:
        lines.append("")
        lines.append(_e(s.extra))
    return "\n".join(lines)


def format_balance_short(s: BotSnapshot) -> str:
    cash = s.balance - s.unrealized_usdt
    return (
        f"<b>{_e(s.strategy)}</b> {_e(s.mode)}\n"
        f"Equity <b>{s.balance:.2f}</b> · cash {cash:.2f} · "
        f"uPnL {s.unrealized_usdt:+.2f} · avail {s.available:.2f}\n"
        f"Позицій: {s.open_count}/{s.max_positions}"
    )


def format_positions(s: BotSnapshot) -> str:
    if not s.positions:
        return f"<b>{_e(s.strategy)}</b>\nНемає відкритих позицій."
    lines = [f"<b>Позиції ({len(s.positions)})</b>"]
    for p in s.positions:
        h = " hedge" if p.hedge else ""
        lines.append(
            f"\n<b>{_e(p.symbol)}</b> {_e(p.side)}{h}\n"
            f"entry {p.entry:.6f} · mark {p.mark:.6f} · uPnL {p.unrealized_usdt:+.2f}\n"
            f"SL {p.sl:.6f} · TP {p.tp:.6f}"
            + (f" · {_e(p.stage)}" if p.stage else "")
        )
    return "\n".join(lines)


def format_history(journal: Optional[TradeJournal] = None, limit: int = 15) -> str:
    lines = ["<b>Останні події</b>"]
    for row in recent_activity(limit):
        ts = time.strftime("%H:%M", time.localtime(row["ts"]))
        sym = f" {row['symbol']}" if row.get("symbol") else ""
        pnl = row.get("pnl_usdt")
        pnl_s = f" {pnl:+.2f}" if pnl is not None else ""
        lines.append(f"{ts} [{_e(row['kind'])}]{_e(sym)}{pnl_s}\n{_e(row['message'])}")
    if journal:
        trades = journal.recent_trades(limit)
        if trades:
            lines.append("\n<b>Журнал угод</b>")
            for t in trades:
                ts = time.strftime(
                    "%m-%d %H:%M",
                    time.localtime(t.get("closed_at") or t.get("opened_at", 0)),
                )
                closed = "✓" if t.get("closed_at") else "…"
                pnl = t.get("pnl_usdt")
                pnl_s = f" {pnl:+.2f}" if pnl is not None and t.get("closed_at") else ""
                lines.append(
                    f"{ts} {closed} <b>{_e(t['symbol'])}</b> {_e(t['side'])}"
                    f"{pnl_s}"
                )
    if len(lines) == 1:
        return "Історія порожня."
    return "\n".join(lines[:40])


def format_stats(journal: Optional[TradeJournal], s: BotSnapshot) -> str:
    lines = [f"<b>Статистика</b> · {_e(s.strategy)} {_e(s.mode)}"]
    if journal:
        st = journal.stats()
        n = st["closed_trades"]
        if n:
            lines.append(
                f"Журнал: {n} угод · PnL {st['total_pnl']:+.2f} USDT · "
                f"WR {st['win_rate']*100:.1f}%"
            )
        else:
            lines.append("Журнал: ще немає закритих угод")
    if s.paper_closed:
        wr = s.paper_wins / s.paper_closed * 100
        lines.append(
            f"Paper wallet: {s.paper_closed} закритих · WR {wr:.0f}% · "
            f"fees/slip {s.paper_costs:.2f} USDT"
        )
    if s.initial_balance > 0:
        lines.append(
            f"Equity vs старт: {s.balance - s.initial_balance:+.2f} USDT "
            f"({s.initial_balance:.2f} → {s.balance:.2f})"
        )
    return "\n".join(lines)
