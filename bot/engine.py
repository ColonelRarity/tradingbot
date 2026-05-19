"""
Scalping bot engine: full-market scan + ML gate + paper/live execution.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set

from bot.config import Config, load_config
from bot.exchange.client import BinanceFuturesClient, Candle
from bot.exchange.stream import KlineStream
from bot.learning.journal import TradeJournal
from bot.market.scanner import MarketScanner
from bot.ml.features import extract_features
from bot.ml.predictor import MLPredictor
from bot.ml.trainer import MLTrainer
from bot.notify import send_message
from bot.telegram.bot import TelegramMonitor, push_trade
from bot.telegram.snapshots import scalper_snapshot
from bot.paper.trader import PaperTrader
from bot.risk.manager import RiskManager
from bot.strategy.scalper import ScalperStrategy, Side
from bot.trading_backend import create_trading_backend

logger = logging.getLogger(__name__)


class ScalpingEngine:
    def __init__(self, config: Config | None = None):
        self.cfg = config or load_config()
        self.client = BinanceFuturesClient(self.cfg)
        self.scanner = MarketScanner(self.client, self.cfg)
        self.strategy = ScalperStrategy(self.cfg)
        self.risk = RiskManager(self.cfg)
        self.journal = TradeJournal()
        self.trader = create_trading_backend(self.client, self.cfg, journal=self.journal)
        self._telegram = TelegramMonitor(
            self.cfg,
            lambda: scalper_snapshot(self.cfg, self.trader, self.client),
            self.journal,
        )
        if isinstance(self.trader, PaperTrader):
            self.trader.on_position_closed = self._on_paper_position_closed
        self.ml: Optional[MLPredictor] = MLPredictor(self.cfg) if self.cfg.ml_enabled else None
        self.ml_trainer = MLTrainer(self.journal, self.client, self.cfg) if self.cfg.ml_enabled else None
        self._candles: Dict[str, List[Candle]] = defaultdict(list)
        self._lock = threading.Lock()
        self._running = False
        self._stream: KlineStream | None = None
        self._scan_offset = 0
        self._last_status_log = 0.0
        self._last_ml_train = 0.0
        self._known_open: Set[str] = set()
        self._symbol_cooldown: Dict[str, float] = {}
        self._scan_stats = {
            "loops": 0,
            "symbol_checks": 0,
            "ta_signals": 0,
            "ml_blocks": 0,
            "entries": 0,
            "last_batch": [],
        }

    def _balance_for_risk(self):
        if isinstance(self.trader, PaperTrader):
            return self.trader.get_balance()
        return self.client.get_balance()

    def _on_paper_position_closed(self, symbol: str, reason: str) -> None:
        if reason in ("SL", "MANUAL"):
            until = time.time() + self.cfg.symbol_cooldown_sec
            self._symbol_cooldown[symbol] = until
            logger.info(
                "Cooldown %s for %.0fs after %s",
                symbol, self.cfg.symbol_cooldown_sec, reason,
            )

    def _in_symbol_cooldown(self, symbol: str) -> bool:
        return time.time() < self._symbol_cooldown.get(symbol, 0.0)

    def _update_candles(self, symbol: str, candles: List[Candle]) -> None:
        with self._lock:
            self._candles[symbol] = candles[-self.cfg.kline_limit :]

    def _fetch_candles(self, symbol: str) -> None:
        try:
            kl = self.client.get_klines(symbol, self.cfg.interval, self.cfg.kline_limit)
            self._update_candles(symbol, kl)
        except Exception as e:
            logger.debug("Klines %s: %s", symbol, e)

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

        if isinstance(self.trader, PaperTrader) and symbol in self.trader.wallet.positions:
            self.trader.update_markets({symbol: float(k["c"])})

    def _maybe_train_ml(self, force: bool = False) -> None:
        if not self.ml_trainer or not self.cfg.ml_enabled:
            return
        now = time.time()
        if not force and (now - self._last_ml_train) < self.cfg.ml_retrain_interval_sec:
            return
        symbols = (
            self.scanner.universe[:15]
            if self.scanner.universe
            else self.cfg.symbols
        )
        result = self.ml_trainer.train(
            bootstrap_symbols=symbols if self.cfg.ml_bootstrap_on_start else None
        )
        self._last_ml_train = now
        if result.get("ok") and self.ml:
            self.ml.reload()
            logger.info(
                "ML retrain OK: n=%s acc=%.3f",
                result.get("n_samples"),
                result.get("val_accuracy", 0),
            )

    def _evaluate_symbol(self, symbol: str) -> None:
        with self._lock:
            candles = list(self._candles.get(symbol, []))

        if len(candles) < 30:
            return

        self._scan_stats["symbol_checks"] += 1

        ta_sig = self.strategy.evaluate(candles)
        if ta_sig.side != Side.NONE:
            self._scan_stats["ta_signals"] += 1
            logger.info(
                "TA signal %s %s | RSI=%.1f | %s",
                symbol, ta_sig.side.value, ta_sig.rsi, ta_sig.reason,
            )

        features = extract_features(candles, self.cfg)
        labeled = self.journal.count_labeled_samples()

        sig = ta_sig
        ml_note = ""
        pred = None

        if self.ml and features is not None:
            pred = self.ml.predict_features(features)
            sig, ml_note = self.ml.gate_signal(ta_sig, pred, labeled)
        elif self.cfg.ml_enabled and ta_sig.side != Side.NONE:
            ml_note = "ML_NO_FEATURES"

        if sig.side == Side.NONE:
            if ta_sig.side != Side.NONE and ml_note.startswith("ML_BLOCK"):
                self._scan_stats["ml_blocks"] += 1
                logger.info("%s %s (p_long=%.3f)", symbol, ml_note, pred.p_long if pred else 0)
            return

        if self._in_symbol_cooldown(symbol):
            logger.debug("%s skip — cooldown active", symbol)
            return

        reason = sig.reason
        if ml_note:
            reason = f"{reason}|{ml_note}"

        bal = self._balance_for_risk()
        if self.risk.daily_loss_exceeded(bal.total_balance):
            return

        if self.trader.count_positions() >= self.cfg.max_positions:
            return

        if self.trader.has_position(symbol):
            return

        info = self.client.get_symbol_info(symbol)
        plan = self.risk.build_plan(
            sig.side,
            sig.price,
            sig.atr,
            bal.available_balance,
            self.cfg.leverage,
            info["min_qty"],
            max(info["min_notional"], self.cfg.min_notional_usdt),
        )

        if not plan.valid:
            logger.debug("%s plan rejected: %s", symbol, plan.reason)
            return

        mark = self.client.get_mark_price(symbol)
        sample_id: Optional[int] = None
        if features is not None and pred and pred.model_loaded:
            sample_id = self.journal.log_ml_sample(
                symbol, ta_sig.side.value, features, pred.p_long, pred.confidence
            )

        if isinstance(self.trader, PaperTrader):
            opened = self.trader.open_scalp(
                symbol, sig.side, plan, mark,
                signal_reason=reason, rsi=sig.rsi,
            )
        else:
            opened = self.trader.open_scalp(symbol, sig.side, plan, mark)

        self.journal.log_signal(symbol, sig.side.value, reason, sig.rsi, traded=opened)
        if opened:
            self._scan_stats["entries"] += 1
            trade_id = self.journal.log_trade_open(
                symbol,
                sig.side.value,
                mark,
                plan.quantity,
                meta={"reason": reason, "rsi": sig.rsi, "dry_run": self.cfg.dry_run},
                ml_sample_id=sample_id,
            )
            if sample_id and trade_id:
                self.journal.link_trade_to_sample(trade_id, sample_id)
            push_trade(
                "OPEN",
                symbol,
                f"{sig.side.value} @ {mark:.6f} · {reason}",
                config=self.cfg,
            )

    def _update_paper_exits(self, symbols: List[str]) -> None:
        if not isinstance(self.trader, PaperTrader):
            return
        prices: Dict[str, float] = {}
        for symbol in set(symbols) | set(self.trader.wallet.positions.keys()):
            try:
                prices[symbol] = self.client.get_mark_price(symbol)
            except Exception:
                continue
        self.trader.update_markets(prices)

    def _detect_closed_positions(self) -> None:
        if self.cfg.dry_run:
            return
        current = {p.symbol for p in self.client.get_positions()}
        closed = self._known_open - current
        for sym in closed:
            try:
                mark = self.client.get_mark_price(sym)
                self.journal.log_trade_close(sym, mark, 0.0)
            except Exception:
                self.journal.log_trade_close(sym, 0.0, 0.0)
        self._known_open = current

    def _restart_websocket(self, symbols: List[str]) -> None:
        if self._stream:
            self._stream.stop()
            time.sleep(0.5)
        if symbols:
            self._stream = KlineStream(symbols, self.cfg.interval, self._on_kline)
            self._stream.start()

    def _loop(self) -> None:
        while self._running:
            try:
                self.trader.sync_positions()
                self._detect_closed_positions()
                self._maybe_train_ml()

                if self.cfg.scan_all_pairs:
                    self.scanner.rescan_if_due()
                    batch, self._scan_offset = self.scanner.next_batch(
                        self._scan_offset,
                        self.cfg.scan_batch_size,
                    )
                    symbols_to_eval = batch
                else:
                    symbols_to_eval = self.cfg.symbols

                self._scan_stats["loops"] += 1
                self._scan_stats["last_batch"] = list(symbols_to_eval)

                if self.cfg.scan_verbose:
                    logger.info("Checking batch (%d): %s", len(symbols_to_eval), ", ".join(symbols_to_eval))

                for symbol in symbols_to_eval:
                    if symbol not in self._candles or len(self._candles[symbol]) < 30:
                        self._fetch_candles(symbol)
                    self._evaluate_symbol(symbol)

                self._update_paper_exits(symbols_to_eval)

                now = time.time()
                if now - self._last_status_log >= self.cfg.status_log_interval_sec:
                    stats = self.journal.stats()
                    uni = len(self.scanner.universe) if self.cfg.scan_all_pairs else len(self.cfg.symbols)
                    extra = ""
                    if isinstance(self.trader, PaperTrader):
                        extra = " | " + self.trader.summary_line()
                    ml_info = ""
                    if self.cfg.ml_enabled:
                        ml_info = f" | ML labeled={stats.get('ml_labeled_samples', 0)} mode={self.cfg.ml_decision_mode}"
                    st = self._scan_stats
                    batch_str = ",".join(st["last_batch"][:6])
                    if len(st["last_batch"]) > 6:
                        batch_str += "..."
                    logger.info(
                        "Heartbeat | universe=%d | loops=%d checks=%d | "
                        "TA=%d ML_block=%d entries=%d | pos=%d | batch=[%s]%s%s",
                        uni,
                        st["loops"],
                        st["symbol_checks"],
                        st["ta_signals"],
                        st["ml_blocks"],
                        st["entries"],
                        self.trader.count_positions(),
                        batch_str,
                        ml_info,
                        extra,
                    )
                    self._last_status_log = now

            except Exception as e:
                logger.error("Loop error: %s", e, exc_info=True)
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

        if self.cfg.scan_all_pairs:
            self.scanner.rescan_if_due(force=True)
            ws_symbols = self.scanner.top_for_websocket(self.cfg.max_ws_symbols)
            batch, self._scan_offset = self.scanner.next_batch(0, self.cfg.scan_batch_size)
            for s in batch:
                self._fetch_candles(s)
        else:
            ws_symbols = self.cfg.symbols
            for s in ws_symbols:
                self._fetch_candles(s)

        if not self.cfg.dry_run:
            self._known_open = {p.symbol for p in self.client.get_positions()}

        if self.cfg.ml_enabled:
            self._maybe_train_ml(force=True)

        if self.cfg.dry_run:
            mode = "DRY RUN (paper)"
        elif self.cfg.is_testnet():
            mode = "TESTNET"
        else:
            mode = "LIVE"

        scan_mode = (
            f"FULL MARKET ({len(self.scanner.universe)} pairs)"
            if self.cfg.scan_all_pairs
            else f"FIXED ({', '.join(self.cfg.symbols)})"
        )
        ml_mode = f"ML {self.cfg.ml_decision_mode}" if self.cfg.ml_enabled else "ML off"
        explore = " | EXPLORE" if self.cfg.paper_exploration_active else ""
        self._telegram.start()
        push_trade(
            "START",
            "SCALPER",
            f"{mode} · {scan_mode} · {ml_mode}{explore}",
            config=self.cfg,
        )
        logger.info(
            "=== %s | %s | %s | %s | lev=%dx ===",
            mode, scan_mode, ml_mode,
            self.cfg.strategy_mode.upper() + explore,
            self.cfg.leverage,
        )

        self._running = True
        self._restart_websocket(ws_symbols)

        threading.Thread(target=self._loop, daemon=True, name="scalp-loop").start()

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
        if isinstance(self.trader, PaperTrader):
            logger.info("Final: %s", self.trader.summary_line())
        if self.cfg.ml_enabled:
            logger.info("ML samples labeled: %d", self.journal.count_labeled_samples())
        self._telegram.stop()
        push_trade("STOP", "SCALPER", "Bot stopped", config=self.cfg)
        logger.info("Bot stopped")
