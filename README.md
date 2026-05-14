# Binance USDT-M Futures — multi-pair trading bot (testnet)

Python-бот для **Binance USDⓈ-M Futures testnet**: сканування ліквідних USDT-пар, сигнали (фічі + PyTorch), ризик, **MARKET / STOP_MARKET / TAKE_PROFIT_MARKET**, трекінг позицій, опційний хедж, Telegram.

## Вимоги

- Python 3.10+ (рекомендовано 3.11+)
- `pip install -r requirements.txt`

Залежність **`binance-futures-connector>=4.1.0`** відповідає актуальному REST USD-M (зокрема `/fapi/v3/account`, `/fapi/v3/positionRisk`, `/fapi/v2/ticker/price` у шарі конектора).

## Запуск

```bash
python main.py
python main.py --log-level DEBUG
```

Перевірка імпортів без запуску торгівлі:

```bash
python verify_setup.py
```

## Конфігурація

Усі параметри збираються в `config/settings.py`. Ключові змінні середовища:

| Змінна | Призначення |
|--------|-------------|
| `BINANCE_FUTURES_API_KEY` / `BINANCE_FUTURES_API_SECRET` | Testnet API |
| `BINANCE_DEMO_API_KEY` / `BINANCE_DEMO_API_SECRET` | Альтернативні імена для тих самих ключів |
| `DEFAULT_LEVERAGE` | Плече за замовчуванням |
| `MAX_PAIRS_TO_SCAN` | Скільки пар сканувати |
| `MAX_CONCURRENT_POSITIONS` | Ліміт одночасних позицій |
| `TELEGRAM_ENABLED`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Сповіщення |

### WebSocket (оновлення Binance 2026+)

На **production** Binance розділяє WebSocket на `/public`, `/market`, `/private`. Для **testnet** за замовчуванням використовуються префікси з `ExchangeConfig`:

- `BINANCE_FUTURES_WS_STREAM_PREFIX` — combined market streams (за замовчуванням `wss://stream.binancefuture.com/stream`)
- `BINANCE_FUTURES_WS_USER_PREFIX` — user stream до `/<listenKey>` (за замовчуванням `wss://fstream.binancefuture.com/ws`)

Для live-торгівлі (не testnet) у `utils/websocket_client.py` при `testnet=False` можна задати ті самі змінні; типовий production combined URL: `wss://fstream.binance.com/market/stream`.

## Структура проєкту

- `main.py` — головний цикл, скан ринку, сигнали, керування позиціями
- `exchange/binance_client.py` — REST, ордери, баланс, позиції
- `core/` — ринкові дані, сигнали, ризик, ордери, хедж, позиції
- `ml/` — модель, тренування, інференс
- `utils/websocket_client.py` — опційний WS-клієнт (URL з конфігу)
- `telegram_monitor.py` — Telegram

## Тести

```bash
pytest tests/ -q
```

## Важливо

- Репозиторій задуманий під **testnet**; `settings.validate()` вимагає `testnet` у `base_url`.
- Торгівля криптовалютами несе ризики; тестові ключі теж не варто публікувати у відкритих репозиторіях.

## Ліцензія

Освітній / експериментальний проєкт — використання на власний ризик.
