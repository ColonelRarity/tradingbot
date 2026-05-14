"""
WebSocket клієнт для Binance Futures Testnet
Отримує дані в реальному часі без HTTP polling
"""

import json
import os
import threading
import time
import logging
from typing import Dict, Callable, Optional, List
from collections import defaultdict
import websocket
from datetime import datetime

from config.settings import get_settings

logger = logging.getLogger(__name__)


class BinanceWebSocketClient:
    """
    WebSocket клієнт для Binance Futures Testnet
    
    Підтримує:
    - Price ticker streams
    - Kline/Candlestick streams  
    - Order book streams
    - User data stream (позиції, ордери)
    """
    
    def __init__(self, testnet: bool = True, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Ініціалізація WebSocket клієнта
        
        Args:
            testnet: Використовувати testnet endpoint
            api_key: API ключ (для user data stream)
            api_secret: API секрет (для user data stream)
        """
        self.testnet = testnet
        self.api_key = api_key
        self.api_secret = api_secret

        ex = get_settings().exchange
        # Combined market streams: prefix must end before "?streams=" (e.g. .../stream).
        if testnet:
            self._combined_prefix = ex.ws_stream_combined_prefix.rstrip("/")
            self._user_listen_prefix = ex.ws_user_listen_prefix.rstrip("/")
        else:
            # Live USD-M (2026+): override via env to routed URL, e.g. wss://fstream.binance.com/market/stream
            self._combined_prefix = os.getenv(
                "BINANCE_FUTURES_WS_STREAM_PREFIX",
                "wss://fstream.binance.com/market/stream",
            ).rstrip("/")
            self._user_listen_prefix = os.getenv(
                "BINANCE_FUTURES_WS_USER_PREFIX",
                "wss://fstream.binance.com/ws",
            ).rstrip("/")
        
        # WebSocket з'єднання
        self.ws = None
        self.user_data_ws = None
        self.ws_connections = []  # Список всіх WebSocket з'єднань
        
        # Callbacks для обробки даних
        self.callbacks = defaultdict(list)  # stream_name -> [callbacks]
        
        # Кеш даних
        self.data_cache = {}  # symbol -> {ticker, kline, orderbook}
        
        # Стан
        self.running = False
        self.connected = False
        self.user_data_connected = False
        
        # Потоки
        self.ws_thread = None
        self.user_data_thread = None
        self.heartbeat_thread = None
        
        # Lock для thread-safety
        self.lock = threading.Lock()
        
        # Listen key для user data stream
        self.listen_key = None
        self.listen_key_expires_at = None
        
    def _on_message(self, ws, message):
        """Обробка повідомлень від WebSocket"""
        try:
            data = json.loads(message)
            
            # Обробка різних типів повідомлень
            if 'stream' in data:
                # Combined stream
                stream = data['stream']
                payload = data.get('data', {})
                self._handle_stream_data(stream, payload)
            elif 'e' in data:
                # Одиночний stream
                event_type = data.get('e')
                if event_type == '24hrTicker':
                    self._handle_ticker(data)
                elif event_type == 'kline':
                    self._handle_kline(data)
                elif event_type == 'depthUpdate':
                    self._handle_orderbook(data)
                elif event_type in ['ACCOUNT_UPDATE', 'ORDER_TRADE_UPDATE', 'MARGIN_CALL']:
                    self._handle_user_data(data)
            else:
                # User data stream
                self._handle_user_data(data)
                
        except Exception as e:
            logger.error(f"❌ Помилка обробки WebSocket повідомлення: {e}", exc_info=True)
    
    def _handle_ticker(self, data: Dict):
        """Обробка ticker даних"""
        symbol = data.get('s', '').lower()
        if not symbol:
            return
        
        ticker_data = {
            'symbol': symbol.upper(),
            'price': float(data.get('c', 0)),
            'change_24h': float(data.get('P', 0)),
            'volume_24h': float(data.get('v', 0)),
            'high_24h': float(data.get('h', 0)),
            'low_24h': float(data.get('l', 0)),
            'timestamp': int(data.get('E', time.time() * 1000))
        }
        
        with self.lock:
            if symbol not in self.data_cache:
                self.data_cache[symbol] = {}
            self.data_cache[symbol]['ticker'] = ticker_data
        
        # Викликати callbacks
        for callback in self.callbacks.get(f'{symbol}@ticker', []):
            try:
                callback(ticker_data)
            except Exception as e:
                logger.error(f"❌ Помилка в callback для ticker {symbol}: {e}")
    
    def _handle_kline(self, data: Dict):
        """Обробка kline даних"""
        symbol = data.get('s', '').lower()
        kline_data = data.get('k', {})
        if not symbol or not kline_data:
            return
        
        kline = {
            'symbol': symbol.upper(),
            'interval': kline_data.get('i', '1m'),
            'open': float(kline_data.get('o', 0)),
            'high': float(kline_data.get('h', 0)),
            'low': float(kline_data.get('l', 0)),
            'close': float(kline_data.get('c', 0)),
            'volume': float(kline_data.get('v', 0)),
            'is_closed': kline_data.get('x', False),
            'timestamp': int(kline_data.get('t', time.time() * 1000))
        }
        
        with self.lock:
            if symbol not in self.data_cache:
                self.data_cache[symbol] = {}
            if 'klines' not in self.data_cache[symbol]:
                self.data_cache[symbol]['klines'] = {}
            self.data_cache[symbol]['klines'][kline['interval']] = kline
        
        # Викликати callbacks
        stream_name = f"{symbol}@kline_{kline['interval']}"
        for callback in self.callbacks.get(stream_name, []):
            try:
                callback(kline)
            except Exception as e:
                logger.error(f"❌ Помилка в callback для kline {symbol}: {e}")
    
    def _handle_orderbook(self, data: Dict):
        """Обробка orderbook даних"""
        symbol = data.get('s', '').lower()
        if not symbol:
            return
        
        orderbook_data = {
            'symbol': symbol.upper(),
            'bids': [[float(b[0]), float(b[1])] for b in data.get('b', [])],
            'asks': [[float(a[0]), float(a[1])] for a in data.get('a', [])],
            'timestamp': int(data.get('E', time.time() * 1000))
        }
        
        with self.lock:
            if symbol not in self.data_cache:
                self.data_cache[symbol] = {}
            self.data_cache[symbol]['orderbook'] = orderbook_data
        
        # Викликати callbacks
        for callback in self.callbacks.get(f'{symbol}@depth', []):
            try:
                callback(orderbook_data)
            except Exception as e:
                logger.error(f"❌ Помилка в callback для orderbook {symbol}: {e}")
    
    def _handle_user_data(self, data: Dict):
        """Обробка user data stream"""
        event_type = data.get('e')
        
        if event_type == 'ACCOUNT_UPDATE':
            # Оновлення балансу/позицій
            for callback in self.callbacks.get('user_data', []):
                try:
                    callback({'type': 'account_update', 'data': data})
                except Exception as e:
                    logger.error(f"❌ Помилка в callback для account_update: {e}")
        
        elif event_type == 'ORDER_TRADE_UPDATE':
            # Оновлення ордерів
            for callback in self.callbacks.get('user_data', []):
                try:
                    callback({'type': 'order_update', 'data': data})
                except Exception as e:
                    logger.error(f"❌ Помилка в callback для order_update: {e}")
    
    def _handle_stream_data(self, stream: str, payload: Dict):
        """Обробка даних з combined stream"""
        if '@ticker' in stream:
            self._handle_ticker(payload)
        elif '@kline' in stream:
            self._handle_kline(payload)
        elif '@depth' in stream:
            self._handle_orderbook(payload)
    
    def _on_error(self, ws, error):
        """Обробка помилок WebSocket"""
        logger.error(f"❌ WebSocket помилка: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Обробка закриття WebSocket"""
        logger.warning(f"⚠️ WebSocket закрито: {close_status_code} - {close_msg}")
        self.connected = False
        
        # Автоматичне переподключення
        if self.running:
            logger.info("🔄 Переподключення до WebSocket...")
            time.sleep(5)
            self.start()
    
    def _on_open(self, ws):
        """Обробка відкриття WebSocket"""
        logger.info("✅ WebSocket підключено")
        self.connected = True
    
    def subscribe_ticker(self, symbols: List[str], callback: Optional[Callable] = None):
        """
        Підписатися на ticker потоки
        
        Args:
            symbols: Список символів (наприклад, ['BTCUSDT', 'ETHUSDT'])
            callback: Функція для обробки даних
        """
        streams = [f"{s.lower()}@ticker" for s in symbols]
        self._subscribe_streams(streams, callback)
    
    def subscribe_klines(self, symbols: List[str], interval: str = '1m', callback: Optional[Callable] = None):
        """
        Підписатися на kline потоки
        
        Args:
            symbols: Список символів
            interval: Інтервал ('1m', '5m', '1h', etc.)
            callback: Функція для обробки даних
        """
        streams = [f"{s.lower()}@kline_{interval}" for s in symbols]
        self._subscribe_streams(streams, callback)
    
    def subscribe_orderbook(self, symbols: List[str], callback: Optional[Callable] = None):
        """
        Підписатися на orderbook потоки
        
        Args:
            symbols: Список символів
            callback: Функція для обробки даних
        """
        streams = [f"{s.lower()}@depth@100ms" for s in symbols]  # 100ms оновлення
        self._subscribe_streams(streams, callback)
    
    def subscribe_user_data(self, callback: Optional[Callable] = None):
        """
        Підписатися на user data stream (позиції, ордери)
        
        Args:
            callback: Функція для обробки даних
        """
        if callback:
            self.callbacks['user_data'].append(callback)
        
        if not self.listen_key:
            self._get_listen_key()
        
        if self.listen_key:
            self._start_user_data_stream()
    
    def _subscribe_streams(self, streams: List[str], callback: Optional[Callable] = None):
        """Підписатися на потоки"""
        if callback:
            for stream in streams:
                if callback not in self.callbacks[stream]:
                    self.callbacks[stream].append(callback)
        
        # Якщо вже запущено, потрібно перезапустити з новими потоками
        if self.running and not self.connected:
            # Перезапустити з новими потоками
            self.running = False
            # Закрити старі з'єднання
            for ws in self.ws_connections:
                try:
                    ws.close()
                except:
                    pass
            self.ws_connections = []
            time.sleep(1)
            self.start()
        elif not self.running:
            self.start()
    
    def _get_listen_key(self) -> Optional[str]:
        """Отримати listen key для user data stream"""
        try:
            import requests
            url = "https://testnet.binancefuture.com" if self.testnet else "https://fapi.binance.com"
            response = requests.post(
                f"{url}/fapi/v1/listenKey",
                headers={"X-MBX-APIKEY": self.api_key} if self.api_key else {},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                self.listen_key = data.get('listenKey')
                # Listen key дійсний 60 хвилин, оновлюємо кожні 50 хвилин
                self.listen_key_expires_at = time.time() + 50 * 60
                logger.info(f"✅ Отримано listen key для user data stream")
                return self.listen_key
        except Exception as e:
            logger.error(f"❌ Помилка отримання listen key: {e}")
        return None
    
    def _keep_alive_listen_key(self):
        """Оновити listen key (heartbeat)"""
        while self.running and self.listen_key:
            try:
                if self.listen_key_expires_at and time.time() >= self.listen_key_expires_at - 60:
                    # Оновити listen key за 1 хвилину до закінчення
                    import requests
                    url = "https://testnet.binancefuture.com" if self.testnet else "https://fapi.binance.com"
                    response = requests.put(
                        f"{url}/fapi/v1/listenKey",
                        headers={"X-MBX-APIKEY": self.api_key} if self.api_key else {},
                        params={"listenKey": self.listen_key},
                        timeout=10
                    )
                    if response.status_code == 200:
                        self.listen_key_expires_at = time.time() + 50 * 60
                        logger.debug("✅ Listen key оновлено")
                    else:
                        logger.warning(f"⚠️ Не вдалося оновити listen key: {response.status_code}")
                        # Спробувати отримати новий ключ
                        self._get_listen_key()
                        if self.user_data_ws:
                            self._start_user_data_stream()
            except Exception as e:
                logger.error(f"❌ Помилка оновлення listen key: {e}")
            
            # Перевіряти кожні 5 хвилин
            time.sleep(5 * 60)
    
    def _start_user_data_stream(self):
        """Запустити user data stream"""
        if not self.listen_key:
            return
        
        # Закрити попереднє з'єднання якщо є
        if self.user_data_ws:
            try:
                self.user_data_ws.close()
            except:
                pass
        
        url = f"{self._user_listen_prefix}/{self.listen_key}"
        
        def on_message(ws, message):
            self._on_message(ws, message)
        
        def on_error(ws, error):
            self._on_error(ws, error)
        
        def on_close(ws, close_status_code, close_msg):
            logger.warning(f"⚠️ User data stream закрито: {close_status_code}")
            self.user_data_connected = False
            if self.running:
                time.sleep(5)
                self._start_user_data_stream()
        
        def on_open(ws):
            logger.info("✅ User data stream підключено")
            self.user_data_connected = True
        
        self.user_data_ws = websocket.WebSocketApp(
            url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        def run_user_data():
            self.user_data_ws.run_forever()
        
        if self.user_data_thread and self.user_data_thread.is_alive():
            # Потік вже запущений
            return
        
        self.user_data_thread = threading.Thread(target=run_user_data, daemon=True)
        self.user_data_thread.start()
        
        # Запустити heartbeat для listen key
        if not self.heartbeat_thread or not self.heartbeat_thread.is_alive():
            self.heartbeat_thread = threading.Thread(target=self._keep_alive_listen_key, daemon=True)
            self.heartbeat_thread.start()
    
    def start(self):
        """Запустити WebSocket клієнт"""
        if self.running:
            return
        
        self.running = True
        
        # Створити combined stream URL з усіма підписаними потоками
        streams = list(self.callbacks.keys())
        # Виключити user_data з combined streams
        streams = [s for s in streams if s != 'user_data']
        
        if not streams:
            logger.warning("⚠️ Немає підписаних потоків")
            return
        
        # Binance підтримує до 200 потоків в одному combined stream
        # Розділити на батчі по 200
        batch_size = 200
        self.ws_connections = []  # Зберігати всі з'єднання
        
        for i in range(0, len(streams), batch_size):
            batch = streams[i:i+batch_size]
            # Правильний формат для combined streams: stream1/stream2/stream3
            # Binance приймає формат: wss://stream.binancefuture.com/stream?streams=stream1/stream2/stream3
            stream_url = "/".join(batch)
            url = f"{self._combined_prefix}?streams={stream_url}"
            logger.debug(f"📡 Підключення до WebSocket: {len(batch)} потоків")
            
            def on_message(ws, message):
                self._on_message(ws, message)
            
            def on_error(ws, error):
                self._on_error(ws, error)
            
            def on_close(ws, close_status_code, close_msg):
                self._on_close(ws, close_status_code, close_msg)
            
            def on_open(ws):
                self._on_open(ws)
            
            ws_app = websocket.WebSocketApp(
                url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            self.ws_connections.append(ws_app)
            
            def run_ws():
                ws_app.run_forever()
            
            thread = threading.Thread(target=run_ws, daemon=True)
            thread.start()
            if self.ws_thread is None:
                self.ws_thread = thread
    
    def stop(self):
        """Зупинити WebSocket клієнт"""
        self.running = False
        
        # Закрити всі WebSocket з'єднання
        for ws in self.ws_connections:
            try:
                ws.close()
            except:
                pass
        
        if self.user_data_ws:
            try:
                self.user_data_ws.close()
            except:
                pass
        
        # Видалити listen key
        if self.listen_key:
            try:
                import requests
                url = "https://testnet.binancefuture.com" if self.testnet else "https://fapi.binance.com"
                requests.delete(
                    f"{url}/fapi/v1/listenKey",
                    headers={"X-MBX-APIKEY": self.api_key} if self.api_key else {},
                    params={"listenKey": self.listen_key},
                    timeout=10
                )
            except:
                pass
        
        self.connected = False
        self.user_data_connected = False
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Отримати поточний ticker з кешу"""
        with self.lock:
            return self.data_cache.get(symbol.lower(), {}).get('ticker')
    
    def get_kline(self, symbol: str, interval: str = '1m') -> Optional[Dict]:
        """Отримати поточну kline з кешу"""
        with self.lock:
            return self.data_cache.get(symbol.lower(), {}).get('klines', {}).get(interval)
    
    def get_orderbook(self, symbol: str) -> Optional[Dict]:
        """Отримати поточний orderbook з кешу"""
        with self.lock:
            return self.data_cache.get(symbol.lower(), {}).get('orderbook')
