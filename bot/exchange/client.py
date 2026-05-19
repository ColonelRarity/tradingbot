"""
Binance USD-M Futures REST client (live + testnet via BINANCE_FUTURES_BASE_URL).
"""

from __future__ import annotations

import logging
import time
import types
from collections import deque
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any, Dict, List, Literal, Optional

import requests
from binance.error import ClientError, ServerError
from binance.um_futures import UMFutures

from bot.config import Config, load_config

logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class AccountBalance:
    total_balance: float
    available_balance: float
    unrealized_pnl: float


@dataclass
class Position:
    symbol: str
    side: str
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    leverage: int


@dataclass
class Order:
    order_id: int
    symbol: str
    side: str
    order_type: str
    status: str
    quantity: float
    stop_price: float
    is_algo: bool = False


@dataclass
class Candle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    closed: bool = True


class RateLimiter:
    def __init__(self, max_per_minute: int = 1000):
        self._times: deque = deque()
        self._max = max_per_minute
        self._lock = Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.time()
            while self._times and now - self._times[0] > 60:
                self._times.popleft()
            if len(self._times) >= self._max:
                sleep_for = 60 - (now - self._times[0]) + 0.05
                if sleep_for > 0:
                    time.sleep(sleep_for)
                now = time.time()
            self._times.append(now)


class BinanceFuturesClient:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or load_config()
        self._limiter = RateLimiter()
        self._time_offset_ms = 0
        self._client = UMFutures(
            key=self.config.api_key,
            secret=self.config.api_secret,
            base_url=self.config.base_url,
        )
        self._symbol_info: Dict[str, Dict[str, float]] = {}
        self._dual_side: Optional[bool] = None
        self._patch_signed_requests()
        logger.info("Futures client → %s", self.config.base_url)

    def sync_server_time(self) -> int:
        """Align signed request timestamps with Binance server (fixes -1021)."""
        url = self.config.base_url.rstrip("/") + "/fapi/v1/time"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        server_ms = int(r.json()["serverTime"])
        local_ms = int(time.time() * 1000)
        self._time_offset_ms = server_ms - local_ms
        logger.info("Binance time sync: offset %d ms", self._time_offset_ms)
        return self._time_offset_ms

    def _patch_signed_requests(self) -> None:
        offset_holder = self
        recv_window = self.config.binance_recv_window
        api = self._client

        def sign_request(self, http_method, url_path, payload=None, special=False):
            if payload is None:
                payload = {}
            payload["timestamp"] = int(time.time() * 1000) + offset_holder._time_offset_ms
            payload.setdefault("recvWindow", recv_window)
            query_string = self._prepare_params(payload, special)
            payload["signature"] = self._get_sign(query_string)
            return self.send_request(http_method, url_path, payload, special)

        def limited_encoded_sign_request(self, http_method, url_path, payload=None):
            if payload is None:
                payload = {}
            payload["timestamp"] = int(time.time() * 1000) + offset_holder._time_offset_ms
            payload.setdefault("recvWindow", recv_window)
            query_string = self._prepare_params(payload)
            url_path = (
                url_path + "?" + query_string + "&signature=" + self._get_sign(query_string)
            )
            return self.send_request(http_method, url_path)

        api.sign_request = types.MethodType(sign_request, api)
        api.limited_encoded_sign_request = types.MethodType(
            limited_encoded_sign_request, api
        )

    def _call(self, func, *args, is_order: bool = False, **kwargs) -> Any:
        self._limiter.wait()
        try:
            return func(*args, **kwargs)
        except ClientError as e:
            if e.error_code == -1021:
                logger.warning("Timestamp drift (%s), resyncing clock…", e.error_message)
                self.sync_server_time()
                time.sleep(0.2)
                return func(*args, **kwargs)
            raise
        except requests.exceptions.RequestException as e:
            logger.warning("Network error, retry once: %s", e)
            time.sleep(1)
            return func(*args, **kwargs)

    def _sign(self, method: str, path: str, params: Optional[Dict] = None, is_order: bool = False) -> Any:
        self._limiter.wait()
        return self._client.sign_request(method, path, dict(params or {}))

    def load_exchange_info(self) -> None:
        data = self._call(self._client.exchange_info)
        for s in data.get("symbols", []):
            if s.get("status") != "TRADING" or s.get("contractType") != "PERPETUAL":
                continue
            sym = s["symbol"]
            filters = {f["filterType"]: f for f in s.get("filters", [])}
            lot = filters.get("LOT_SIZE", {})
            price_f = filters.get("PRICE_FILTER", {})
            min_n = filters.get("MIN_NOTIONAL", filters.get("NOTIONAL", {}))
            self._symbol_info[sym] = {
                "step_size": float(lot.get("stepSize", "0.001")),
                "min_qty": float(lot.get("minQty", "0.001")),
                "max_qty": float(lot.get("maxQty", "1e15")),
                "tick_size": float(price_f.get("tickSize", "0.01")),
                "min_notional": float(min_n.get("notional", min_n.get("minNotional", "5"))),
            }
        logger.info("Exchange info: %d symbols", len(self._symbol_info))

    def get_symbol_info(self, symbol: str) -> Dict[str, float]:
        if symbol not in self._symbol_info:
            self.load_exchange_info()
        return self._symbol_info.get(symbol, {
            "step_size": 0.001, "min_qty": 0.001, "max_qty": 1e9,
            "tick_size": 0.01, "min_notional": 5.0,
        })

    def _round_step(self, value: float, step: float) -> float:
        if step <= 0:
            return value
        return float(int(value / step) * step)

    def format_qty(self, symbol: str, qty: float) -> str:
        info = self.get_symbol_info(symbol)
        q = max(info["min_qty"], min(info["max_qty"], self._round_step(qty, info["step_size"])))
        prec = max(0, len(str(info["step_size"]).rstrip("0").split(".")[-1]) if "." in str(info["step_size"]) else 0)
        return f"{q:.{prec}f}"

    def format_price(self, symbol: str, price: float) -> str:
        info = self.get_symbol_info(symbol)
        p = self._round_step(price, info["tick_size"])
        prec = max(0, len(str(info["tick_size"]).rstrip("0").split(".")[-1]) if "." in str(info["tick_size"]) else 0)
        return f"{p:.{prec}f}"

    def is_dual_side(self) -> bool:
        if self._dual_side is not None:
            return self._dual_side
        self._refresh_position_mode()
        return bool(self._dual_side)

    def _refresh_position_mode(self) -> None:
        try:
            r = self._call(self._client.get_position_mode)
            self._dual_side = str(r.get("dualSidePosition", False)).lower() == "true"
        except Exception:
            self._dual_side = False

    def set_dual_side_position(self, hedge_mode: bool) -> None:
        """Switch account to hedge (dual) or one-way mode. Fails if open positions exist."""
        want = "true" if hedge_mode else "false"
        self._call(
            self._client.change_position_mode,
            dualSidePosition=want,
            is_order=True,
        )
        self._dual_side = hedge_mode
        logger.info("Position mode → %s", "HEDGE (dual)" if hedge_mode else "ONE-WAY")

    def test_connection(self) -> bool:
        try:
            self.sync_server_time()
            self._call(self._client.ping)
            bal = self.get_balance()
            logger.info("Connected. Available: %.2f USDT", bal.available_balance)
            return True
        except Exception as e:
            logger.error("Connection failed: %s", e)
            return False

    def get_balance(self) -> AccountBalance:
        acc = self._call(self._client.account)
        for a in acc.get("assets", []):
            if a["asset"] == "USDT":
                return AccountBalance(
                    total_balance=float(a["walletBalance"]),
                    available_balance=float(a["availableBalance"]),
                    unrealized_pnl=float(a["unrealizedProfit"]),
                )
        raise ValueError("USDT balance not found")

    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        rows = self._call(self._client.get_position_risk, symbol=symbol)
        out: List[Position] = []
        for p in rows:
            amt = float(p["positionAmt"])
            if abs(amt) < 1e-12:
                continue
            ps = str(p.get("positionSide", "")).upper()
            side = ps if ps in ("LONG", "SHORT") else ("LONG" if amt > 0 else "SHORT")
            out.append(Position(
                symbol=p["symbol"],
                side=side,
                size=abs(amt),
                entry_price=float(p.get("entryPrice", 0)),
                mark_price=float(p.get("markPrice", 0)),
                unrealized_pnl=float(p.get("unRealizedProfit", 0)),
                leverage=int(p.get("leverage", 1)),
            ))
        return out

    def get_tradable_usdt_symbols(self) -> set[str]:
        """All TRADING USDT perpetual symbols from exchange info."""
        if not self._symbol_info:
            self.load_exchange_info()
        return {s for s in self._symbol_info if s.endswith("USDT")}

    def get_top_usdt_pairs(
        self,
        min_quote_volume_usdt: float = 1_000_000,
        max_pairs: int = 150,
    ) -> List[str]:
        """
        Rank USDT-M perpetuals by 24h quote volume (USDT).
        One API call — full market snapshot.
        """
        if not self._symbol_info:
            self.load_exchange_info()
        allowed = self.get_tradable_usdt_symbols()

        try:
            tickers = self._call(self._client.ticker_24hr_price_change)
        except AttributeError:
            tickers = self._call(self._client.ticker_24hr)

        if isinstance(tickers, dict):
            tickers = [tickers]

        ranked: List[tuple[str, float]] = []
        for t in tickers:
            sym = str(t.get("symbol", ""))
            if sym not in allowed:
                continue
            vol = float(t.get("quoteVolume") or 0)
            if vol >= min_quote_volume_usdt:
                ranked.append((sym, vol))

        ranked.sort(key=lambda x: x[1], reverse=True)
        symbols = [s for s, _ in ranked[:max_pairs]]
        logger.debug("Top pair by volume: %s (vol=%.0f)", symbols[0] if symbols else "-", ranked[0][1] if ranked else 0)
        return symbols

    def get_klines(self, symbol: str, interval: str, limit: int = 100) -> List[Candle]:
        raw = self._call(
            self._client.klines,
            symbol=symbol,
            interval=interval,
            limit=limit,
        )
        candles: List[Candle] = []
        for k in raw:
            candles.append(Candle(
                open_time=int(k[0]),
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
                closed=True,
            ))
        return candles

    def get_mark_price(self, symbol: str) -> float:
        r = self._call(self._client.mark_price, symbol=symbol)
        return float(r["markPrice"])

    def get_funding_rate(self, symbol: str) -> float:
        """Last funding rate (e.g. 0.0001 = 0.01% per 8h interval)."""
        r = self._call(self._client.mark_price, symbol=symbol)
        return float(r.get("lastFundingRate", 0))

    def set_leverage(self, symbol: str, leverage: int) -> None:
        try:
            self._call(self._client.change_leverage, symbol=symbol, leverage=leverage, is_order=True)
        except ClientError as e:
            if e.error_code not in (-4028, -4046):
                raise

    def set_margin_type(self, symbol: str, margin_type: str) -> None:
        try:
            self._call(
                self._client.change_margin_type,
                symbol=symbol,
                marginType=margin_type,
                is_order=True,
            )
        except ClientError as e:
            if e.error_code not in (-4046,):
                raise

    def _position_side_param(self, side: str) -> Optional[str]:
        if not self.is_dual_side():
            return None
        return "LONG" if side == "LONG" else "SHORT"

    def market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        position_side: Optional[str] = None,
        reduce_only: bool = False,
    ) -> Order:
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side.value,
            "type": "MARKET",
            "quantity": self.format_qty(symbol, quantity),
        }
        if self.is_dual_side() and position_side:
            params["positionSide"] = position_side
        elif reduce_only:
            params["reduceOnly"] = "true"

        r = self._call(self._client.new_order, is_order=True, **params)
        return Order(
            order_id=int(r["orderId"]),
            symbol=symbol,
            side=side.value,
            order_type="MARKET",
            status=r.get("status", ""),
            quantity=float(r.get("executedQty", params["quantity"])),
            stop_price=0,
        )

    def _algo_conditional_params(
        self,
        symbol: str,
        side: OrderSide,
        order_type: str,
        trigger_price: float,
        quantity: float,
        position_side: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "algoType": "CONDITIONAL",
            "symbol": symbol,
            "side": side.value,
            "type": order_type,
            "triggerPrice": self.format_price(symbol, trigger_price),
            "quantity": self.format_qty(symbol, quantity),
            "workingType": "MARK_PRICE",
        }
        if position_side:
            params["positionSide"] = position_side
        elif not self.is_dual_side():
            params["reduceOnly"] = "true"
        return params

    def _algo_order(self, params: Dict[str, Any]) -> Order:
        r = self._sign("POST", "/fapi/v1/algoOrder", params, is_order=True)
        trigger = float(r.get("triggerPrice") or r.get("stopPrice") or 0)
        return Order(
            order_id=int(r.get("algoId", 0)),
            symbol=str(r.get("symbol", "")),
            side=str(r.get("side", "")),
            order_type=str(r.get("orderType", r.get("type", ""))),
            status=str(r.get("algoStatus", "")),
            quantity=float(r.get("quantity") or 0),
            stop_price=trigger,
            is_algo=True,
        )

    def stop_loss(
        self,
        symbol: str,
        side: OrderSide,
        stop_price: float,
        quantity: float,
        position_side: Optional[str] = None,
    ) -> Order:
        params = self._algo_conditional_params(
            symbol, side, "STOP_MARKET", stop_price, quantity, position_side
        )
        return self._algo_order(params)

    def take_profit(
        self,
        symbol: str,
        side: OrderSide,
        stop_price: float,
        quantity: float,
        position_side: Optional[str] = None,
    ) -> Order:
        params = self._algo_conditional_params(
            symbol, side, "TAKE_PROFIT_MARKET", stop_price, quantity, position_side
        )
        return self._algo_order(params)

    def get_open_algo_orders(self, symbol: Optional[str] = None) -> List[Order]:
        params: Dict[str, Any] = {"algoType": "CONDITIONAL"}
        if symbol:
            params["symbol"] = symbol
        try:
            raw = self._sign("GET", "/fapi/v1/openAlgoOrders", params)
        except ClientError as e:
            logger.debug("openAlgoOrders: %s", e)
            return []
        if isinstance(raw, dict):
            raw = [raw]
        out: List[Order] = []
        for r in raw or []:
            trigger = float(r.get("triggerPrice") or 0)
            out.append(
                Order(
                    order_id=int(r.get("algoId", 0)),
                    symbol=str(r.get("symbol", "")),
                    side=str(r.get("side", "")),
                    order_type=str(r.get("orderType", r.get("type", ""))),
                    status=str(r.get("algoStatus", "")),
                    quantity=float(r.get("quantity") or 0),
                    stop_price=trigger,
                    is_algo=True,
                )
            )
        return out

    def cancel_algo_order(self, algo_id: int, symbol: str) -> None:
        if algo_id <= 0:
            return
        try:
            self._sign(
                "DELETE",
                "/fapi/v1/algoOrder",
                {"symbol": symbol, "algoId": algo_id},
            )
        except ClientError as e:
            if e.error_code not in (-2011, -2013):
                logger.debug("cancel algo %s: %s", algo_id, e)

    def cancel_all_algo(self, symbol: str) -> None:
        try:
            self._sign("DELETE", "/fapi/v1/algoOpenOrders", {"symbol": symbol, "algoType": "CONDITIONAL"})
        except ClientError as e:
            if e.error_code != -2011:
                logger.debug("cancel algo: %s", e)

    def wait_for_position(
        self,
        symbol: str,
        side: str,
        *,
        timeout_sec: float = 3.0,
        poll_sec: float = 0.25,
    ) -> Optional[Position]:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            for p in self.get_positions(symbol):
                if p.symbol == symbol and p.side == side and p.size > 0:
                    return p
            time.sleep(poll_sec)
        return None

    def cancel_all_orders(self, symbol: str) -> None:
        try:
            self._call(self._client.cancel_open_orders, symbol=symbol, is_order=True)
        except ClientError:
            pass
        self.cancel_all_algo(symbol)
