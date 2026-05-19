"""
Champaign engine: high-volatility pairs, directional entry, fast SL/TP + hedge monitor.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Dict, List

from bot.champaign.executor import ChampaignExecutor
from bot.champaign.monitor import ChampaignPositionMonitor
from bot.champaign.state import save_stacks
from bot.champaign.volatility import atr_percent, last_bar_range_pct, rank_by_volatility
from bot.config import Config, load_config
from bot.exchange.client import BinanceFuturesClient, Candle
from bot.exchange.stream import KlineStream
from bot.market.scanner import MarketScanner
from bot.learning.journal import TradeJournal
from bot.notify import send_message
from bot.telegram.bot import TelegramMonitor, push_trade
from bot.telegram.snapshots import champaign_snapshot
from bot.strategy.champaign import ChampaignStrategy
from bot.strategy.scalper import Side

logger = logging.getLogger(__name__)


class ChampaignEngine:
    def __init__(self, config: Config | None = None):
        self.cfg = config or load_config()
        self.client = BinanceFuturesClient(self.cfg)
        self.scanner = MarketScanner(self.client, self.cfg)
        self.strategy = ChampaignStrategy(self.cfg)
        self.journal = TradeJournal()
        self.executor = ChampaignExecutor(
            self.client, self.cfg, journal=self.journal
        )
        self._telegram = TelegramMonitor(
            self.cfg,
            lambda: champaign_snapshot(
                self.cfg, self.executor, self._prices, self.client
            ),
            self.journal,
        )
        self._candles: Dict[str, List[Candle]] = defaultdict(list)
        self._lock = threading.Lock()
        self._running = False
        self._stream: KlineStream | None = None
        self._vol_scan_offset = 0
        self._last_status = 0.0
        self._prices: Dict[str, float] = {}
        self._symbol_cooldown: Dict[str, float] = {}
        self._atr_pct_cache: Dict[str, float] = {}
        self._atr_updated_at: Dict[str, float] = {}
        self._last_top_vol: List[tuple[str, float]] = []
        self._last_vol_log_sig: str = ""
        self._last_ws_vol_sync = 0.0

        self.monitor = ChampaignPositionMonitor(
            self.cfg,
            self.cfg.champaign_fib_levels,
            on_open_hedge=lambda s, m: self.executor.open_hedge(s, m),
            on_close_hedge=lambda s, m: self.executor.close_hedge(s, m),
            on_close_main=self._on_close_main,
            on_update_orders=lambda s, sl, tp: self.executor.update_orders(s, sl, tp),
        )

    def _on_close_main(self, stack, mark: float, reason: str) -> None:
        self.executor.close_main(stack, mark, reason)
        if reason in ("SL", "TP", "MANUAL"):
            self._symbol_cooldown[stack.symbol] = time.time() + self.cfg.symbol_cooldown_sec

    def _update_candles(self, symbol: str, candles: List[Candle]) -> None:
        with self._lock:
            self._candles[symbol] = candles[-self.cfg.kline_limit :]

    def _fetch_candles(self, symbol: str) -> None:
        try:
            kl = self.client.get_klines(symbol, self.cfg.interval, self.cfg.kline_limit)
            self._update_candles(symbol, kl)
        except Exception as e:
            logger.debug("klines %s: %s", symbol, e)

    def _on_kline(self, symbol: str, k: dict) -> None:
        candle = Candle(
            open_time=int(k["t"]),
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
            closed=bool(k.get("x", False)),
        )
        with self._lock:
            buf = self._candles[symbol]
            if buf and buf[-1].open_time == candle.open_time:
                buf[-1] = candle
            else:
                buf.append(candle)
            if len(buf) > self.cfg.kline_limit:
                self._candles[symbol] = buf[-self.cfg.kline_limit :]
        self._prices[symbol] = float(k["c"])

    def _universe(self) -> List[str]:
        if self.cfg.scan_all_pairs:
            return list(self.scanner.universe)
        return list(self.cfg.symbols)

    def _next_vol_scan_batch(self) -> List[str]:
        uni = self._universe()
        if not uni:
            return []
        n = min(self.cfg.champaign_vol_scan_per_loop, len(uni))
        start = self._vol_scan_offset % len(uni)
        if start + n <= len(uni):
            batch = uni[start : start + n]
        else:
            batch = uni[start:] + uni[: (start + n) % len(uni)]
        self._vol_scan_offset = (start + n) % len(uni)
        return batch

    def _refresh_volatility_scores(
        self, symbols: List[str], *, force_fetch: frozenset[str] | None = None
    ) -> None:
        must_fetch = force_fetch or frozenset()
        for symbol in symbols:
            if (
                symbol in must_fetch
                or symbol not in self._candles
                or len(self._candles[symbol]) < 40
            ):
                self._fetch_candles(symbol)
            with self._lock:
                candles = list(self._candles.get(symbol, []))
            if len(candles) >= 40:
                self._atr_pct_cache[symbol] = atr_percent(
                    candles, self.cfg.atr_period
                )
                self._atr_updated_at[symbol] = time.time()

    def _pinned_vol_symbols(self) -> List[str]:
        """Always refresh current vol leaders + open stacks (REST every loop)."""
        n = max(self.cfg.champaign_vol_pin_leaders, 1)
        with self._lock:
            ranked = sorted(
                self._atr_pct_cache.items(), key=lambda x: x[1], reverse=True
            )
        pins = [s for s, _ in ranked[:n]]
        for sym in self.executor.stacks_keys():
            if sym not in pins:
                pins.append(sym)
        return pins

    def _vol_leaderboard(self, limit: int = 5) -> List[tuple[str, float]]:
        """Recompute ATR% from latest candles where available."""
        now = time.time()
        items: List[tuple[str, float]] = []
        with self._lock:
            symbols = set(self._atr_pct_cache) | set(self._candles)
            for sym in symbols:
                candles = self._candles.get(sym, [])
                if len(candles) >= 40:
                    pct = atr_percent(candles, self.cfg.atr_period)
                    self._atr_pct_cache[sym] = pct
                    items.append((sym, pct))
                elif sym in self._atr_pct_cache:
                    items.append((sym, self._atr_pct_cache[sym]))
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:limit]

    def _format_vol_leader(self, symbol: str, atr_pct: float) -> str:
        age = time.time() - self._atr_updated_at.get(symbol, 0)
        with self._lock:
            candles = self._candles.get(symbol, [])
        bar = last_bar_range_pct(candles) if candles else 0.0
        ws_syms = {s.upper() for s in self._stream.symbols} if self._stream else set()
        ws = "ws" if symbol in ws_syms else "rest"
        return f"{symbol}:ATR{atr_pct:.3f}% bar{bar:.3f}% {ws} {age:.0f}s"

    def _sync_vol_websocket(self) -> None:
        if not self._stream or self.cfg.champaign_vol_ws_extra <= 0:
            return
        now = time.time()
        if now - self._last_ws_vol_sync < 45.0:
            return
        self._last_ws_vol_sync = now
        base_n = max(self.cfg.max_ws_symbols, 1)
        base = list(self.scanner.top_for_websocket(base_n))
        extra_n = self.cfg.champaign_vol_ws_extra
        leaders = [s for s, _ in self._vol_leaderboard(extra_n)]
        merged: List[str] = []
        seen: set[str] = set()
        stack_keys = self.executor.stacks_keys()
        for s in base + leaders + stack_keys:
            if s not in seen:
                merged.append(s)
                seen.add(s)
        cap = base_n + extra_n + len(stack_keys)
        merged = sorted(merged[:cap])
        if self._stream.set_symbols(merged):
            logger.info("[CHAMP] WS watchlist → %d symbols (vol leaders included)", len(merged))

    def _top_volatile_candidates(self) -> List[tuple[str, float]]:
        now = time.time()
        with self._lock:
            snapshot = {
                s: list(self._candles.get(s, []))
                for s, p in self._atr_pct_cache.items()
                if len(self._candles.get(s, [])) >= 40
                and now - self._atr_updated_at.get(s, 0) <= self.cfg.champaign_vol_stale_sec
            }
        ranked = rank_by_volatility(
            snapshot,
            self.cfg.champaign_min_atr_pct,
            self.cfg.atr_period,
            top_n=max(self.cfg.champaign_vol_top_n * 5, 15),
        )
        slots = self.cfg.champaign_max_positions - self.executor.count_positions()
        if slots <= 0:
            return []
        out: List[tuple[str, float]] = []
        for sym, pct in ranked:
            if len(out) >= min(self.cfg.champaign_vol_top_n, slots):
                break
            if self.executor.has_position(sym):
                continue
            if time.time() < self._symbol_cooldown.get(sym, 0):
                continue
            out.append((sym, pct))
        return out

    def _try_open(self, symbol: str, atr_pct: float) -> None:
        self._refresh_volatility_scores([symbol])
        with self._lock:
            candles = list(self._candles.get(symbol, []))
        if len(candles) < 40:
            logger.debug("[CHAMP] skip %s — not enough candles", symbol)
            return
        if self.executor.has_position(symbol):
            return
        if time.time() < self._symbol_cooldown.get(symbol, 0):
            return
        if self.executor.count_positions() >= self.cfg.champaign_max_positions:
            return

        sig = self.strategy.evaluate(candles)
        if sig.side == Side.NONE:
            logger.debug("[CHAMP] skip %s — no signal", symbol)
            return

        bal = self.executor.get_balance()
        info = self.client.get_symbol_info(symbol)
        from bot.risk.manager import RiskManager

        rm = RiskManager(self.cfg)
        sl_mult = self.cfg.champaign_initial_sl_atr
        old_sl, old_tp = self.cfg.sl_atr_mult, self.cfg.tp_atr_mult
        self.cfg.sl_atr_mult = sl_mult
        self.cfg.tp_atr_mult = sl_mult * 2
        plan = rm.build_plan(
            sig.side,
            sig.price,
            sig.atr,
            bal.available_balance,
            self.cfg.leverage,
            info["min_qty"],
            max(info["min_notional"], self.cfg.min_notional_usdt),
            max_positions=self.cfg.champaign_max_positions,
        )
        self.cfg.sl_atr_mult, self.cfg.tp_atr_mult = old_sl, old_tp

        if not plan.valid:
            logger.info("[CHAMP] skip %s — plan: %s", symbol, plan.reason)
            return
        mark = self._prices.get(symbol) or self.client.get_mark_price(symbol)
        if mark > 0:
            self._prices[symbol] = mark
        opened = self.executor.open_main(symbol, sig, plan, mark, candles)
        if opened:
            logger.info(
                "[CHAMP] picked %s (ATR%%=%.2f rank top-%d)",
                symbol, atr_pct, self.cfg.champaign_vol_top_n,
            )

    def _monitor_loop(self) -> None:
        while self._running:
            try:
                for symbol, stack in self.executor.stacks_items():
                    # REST mark for SL/TP + profit-lock (WS cache can be stale if symbol not on stream)
                    try:
                        mark = self.client.get_mark_price(symbol)
                        self._prices[symbol] = mark
                    except Exception:
                        mark = self._prices.get(symbol) or 0.0
                        if mark <= 0:
                            continue
                    with self._lock:
                        candles = list(self._candles.get(symbol, []))
                    self.monitor.tick(stack, mark, candles)
                if self.executor.stacks_len():
                    self.executor._persist()
            except Exception as e:
                logger.error("[CHAMP] monitor: %s", e, exc_info=True)
            time.sleep(self.cfg.champaign_poll_sec)

    def _scan_loop(self) -> None:
        while self._running:
            try:
                self.executor.sync_positions()
                if self.cfg.scan_all_pairs:
                    self.scanner.rescan_if_due()

                scan_batch = self._next_vol_scan_batch()
                pinned = self._pinned_vol_symbols()
                for sym in pinned:
                    if sym not in scan_batch:
                        scan_batch.append(sym)
                self._refresh_volatility_scores(
                    scan_batch, force_fetch=frozenset(pinned)
                )
                self._sync_vol_websocket()

                self._last_top_vol = self._top_volatile_candidates()
                leaders = self._vol_leaderboard(5)
                now = time.time()
                if leaders:
                    lead_str = ", ".join(
                        self._format_vol_leader(s, p) for s, p in leaders
                    )
                    above = sum(
                        1
                        for p in self._atr_pct_cache.values()
                        if p >= self.cfg.champaign_min_atr_pct
                    )
                    sig = f"{leaders[0][0]}:{leaders[0][1]:.3f}"
                    log_vol = (
                        now - self._last_status
                        >= self.cfg.status_log_interval_sec - 1
                        or sig != self._last_vol_log_sig
                    )
                    if log_vol:
                        self._last_vol_log_sig = sig
                        logger.info(
                            "[CHAMP] Vol leaders: %s | %d/%d >= %.2f%% ATR",
                            lead_str,
                            above,
                            len(self._atr_pct_cache),
                            self.cfg.champaign_min_atr_pct,
                        )
                if self._last_top_vol:
                    pick_str = ", ".join(f"{s}:{p:.2f}%" for s, p in self._last_top_vol)
                    logger.info("[CHAMP] Entry candidates → %s", pick_str)
                elif self.executor.count_positions() < self.cfg.champaign_max_positions:
                    logger.info(
                        "[CHAMP] No entry candidates (slots free=%d)",
                        self.cfg.champaign_max_positions
                        - self.executor.count_positions(),
                    )
                for symbol, atr_pct in self._last_top_vol:
                    self._try_open(symbol, atr_pct)

                now = time.time()
                if now - self._last_status >= self.cfg.status_log_interval_sec:
                    bal = self.executor.get_balance()
                    cash = self.executor._wallet.balance if self.executor._wallet else bal.total_balance
                    open_syms = ",".join(self.executor.stacks_keys()) or "-"
                    vol_hint = ""
                    if leaders:
                        vol_hint = f" | vol1={leaders[0][0]}:{leaders[0][1]:.3f}%"
                    logger.info(
                        "[CHAMP] Heartbeat | stacks=%d open=%s | equity=%.2f cash=%.2f avail=%.2f | "
                        "universe=%d scored=%d%s",
                        self.executor.stacks_len(),
                        open_syms,
                        bal.total_balance,
                        cash,
                        bal.available_balance,
                        len(self._universe()),
                        len(self._atr_pct_cache),
                        vol_hint,
                    )
                    self._last_status = now
            except Exception as e:
                logger.error("[CHAMP] scan: %s", e, exc_info=True)
            time.sleep(self.cfg.loop_interval_sec)

    def start(self) -> None:
        errors = self.cfg.validate()
        if errors:
            for e in errors:
                logger.error("Config: %s", e)
            raise SystemExit(1)
        if not self.client.test_connection():
            raise SystemExit("Cannot connect to Binance Futures")
        self.client.load_exchange_info()

        if not self.cfg.dry_run:
            logger.warning(
                "=== CHAMPAIGN LIVE — real orders, algo SL/TP on Binance ==="
            )
            if self.cfg.champaign_hedge_enabled and self.cfg.position_mode != "hedge":
                logger.warning(
                    "Hedge enabled: set POSITION_MODE=hedge in .env for mirror hedge"
                )

        if self.cfg.champaign_reset_state_on_start:
            from bot.champaign.state import STATE_PATH

            if STATE_PATH.exists():
                STATE_PATH.unlink()
            self.executor.clear_stacks()
            logger.info(
                "[CHAMP] Cleared saved stack state (CHAMPAIGN_RESET_STATE_ON_START); "
                "exchange positions unchanged"
            )

        if self.cfg.scan_all_pairs:
            self.scanner.rescan_if_due(force=True)
            ws = list(self.scanner.top_for_websocket(self.cfg.max_ws_symbols))
            for sym in self.executor.stacks_keys():
                if sym not in ws:
                    ws.append(sym)
            init_batch = self._universe()[: self.cfg.champaign_vol_scan_per_loop]
            self._refresh_volatility_scores(init_batch)
        else:
            ws = self.cfg.symbols
            for s in ws:
                self._fetch_candles(s)

        mode = "DRY RUN" if self.cfg.dry_run else "LIVE"
        self._telegram.start()
        push_trade(
            "START",
            "CHAMPAIGN",
            f"{mode} · max_pos={self.cfg.champaign_max_positions} · "
            f"vol>={self.cfg.champaign_min_atr_pct:.2f}%",
            config=self.cfg,
        )
        logger.info(
            "=== CHAMPAIGN %s | vol>=%.2f%% | scan %d/loop → top %d entries | "
            "max_pos=%d hedge@%.0f USDT ===",
            mode,
            self.cfg.champaign_min_atr_pct,
            self.cfg.champaign_vol_scan_per_loop,
            self.cfg.champaign_vol_top_n,
            self.cfg.champaign_max_positions,
            self.cfg.champaign_hedge_trigger_usdt,
        )

        self._running = True
        if ws:
            self._stream = KlineStream(ws, self.cfg.interval, self._on_kline)
            self._stream.start()

        threading.Thread(target=self._scan_loop, daemon=True, name="champ-scan").start()
        threading.Thread(target=self._monitor_loop, daemon=True, name="champ-monitor").start()

        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        self._running = False
        if self._stream:
            self._stream.stop()
        if self.executor.stacks_len():
            self.executor._persist()
        self._telegram.stop()
        push_trade("STOP", "CHAMPAIGN", "Bot stopped", config=self.cfg)
        logger.info("[CHAMP] stopped")
