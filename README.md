# Binance Futures Scalper v2

Сучасний скальпінг-бот для **Binance USDⓈ-M Futures** (реальний акаунт або testnet).

## Можливості

- **Скан усієї біржі**: топ USDT perpetual за 24h об’ємом (`SCAN_ALL_PAIRS=true`)
- WebSocket на найліквідніші пари + REST round-robin по всьому universe
- **ML (PyTorch)**: FILTER / FULL режими, навчання з журналу + bootstrap з історії
- Журнал у `data/journal.db` + `data/models/scalp_mlp.pt`
- Стратегія: **EMA + RSI + об’єм** (`strict` = перетин EMA; `relaxed` = тренд EMA)
- Ризик: % від депозиту, SL/TP по **ATR**, денний ліміт збитку
- Ордери: MARKET + algo **STOP_MARKET** / **TAKE_PROFIT_MARKET**
- Секрети лише в `.env` / `env.local.ps1` (не в коді)

## Machine learning

| `ML_DECISION_MODE` | Поведінка |
|--------------------|-----------|
| `OFF` | Тільки TA (EMA+RSI) |
| `FILTER` | TA сигнал + ML має погодитись (рекомендовано) |
| `FULL` | Рішення за ML (опційно з `ML_REQUIRE_TA_SIGNAL`) |

Ручне навчання: `python scripts/train_ml.py`  
Авто-retrain кожні `ML_RETRAIN_INTERVAL_SEC` з урахуванням закритих paper/live угод.

## Champaign (експериментальна стратегія)

Увімкнення: `CHAMPAIGN=true` (хто не ризикує — той не п'є шампанське).

- До **`CHAMPAIGN_MAX_POSITIONS`** одночасних стеків (за замовч. 10); маржа в paper ділиться на цей ліміт
- Пари з **найвищою волатильністю зараз** (ATR%): кожен цикл сканує `CHAMPAIGN_VOL_SCAN_PER_LOOP` пар, вхід лише в топ `CHAMPAIGN_VOL_TOP_N`
- Сигнал **LONG** якщо EMA+RSI за зростання, інакше **SHORT**
- Два потоки: **скан входів** + **монітор позицій** (`CHAMPAIGN_POLL_SEC`, за замовч. 0.5 с)
- **2** protective ордери: **SL (profit-lock)** + **TP (далекий Fib)**
- SL піднімається сходинками **чистого прибутку в USDT** (не % ціни): `0,1,3,5,7,9,11,15,20`, далі кожні `+5` до +455/+460…
- При досягненні сходинки S SL фіксує **попередню** (напр. float +9.6 → lock +7 USDT)
- Початковий SL/TP від **ціни fill**, не від сигналу
- **Хедж** при net-збитку ≥ `CHAMPAIGN_HEDGE_TRIGGER_USDT` (10); закриття хеджу якщо main у плюсі

```env
CHAMPAIGN=true
DRY_RUN=true
PAPER_BALANCE=500
CHAMPAIGN_MAX_POSITIONS=10
CHAMPAIGN_MIN_ATR_PCT=1.2
CHAMPAIGN_VOL_SCAN_PER_LOOP=24
CHAMPAIGN_VOL_TOP_N=3
CHAMPAIGN_HEDGE_TRIGGER_USDT=10
CHAMPAIGN_MAX_PROTECTIVE_ORDERS=2
CHAMPAIGN_PROFIT_LOCK_STEPS=0,1,3,5,7,9,11,15,20
CHAMPAIGN_PROFIT_LOCK_STEP_AFTER_20=5
```

### Champaign LIVE

```env
CHAMPAIGN=true
DRY_RUN=false
CONFIRM_LIVE_TRADING=true
POSITION_MODE=hedge          # обовʼязково, якщо CHAMPAIGN_HEDGE_ENABLED=true
CHAMPAIGN_MAX_PROTECTIVE_ORDERS=2
```

- Після входу на біржу ставляться **algo** `STOP_MARKET` (SL) і `TAKE_PROFIT_MARKET` (TP).
- Profit-lock оновлює лише SL (TP не чіпається).
- Стан стеків: `data/champaign_state.json` (qty, entry, locked profit).
- При старті бот синхронізується з відкритими позиціями та algo-ордерами.

Скальпер (`CHAMPAIGN=false`) і Champaign **не сумішуються** в одному процесі.

## Dry run (paper trading)

Без реальних ордерів — ті самі сигнали, віртуальний баланс:

```env
DRY_RUN=true
PAPER_BALANCE=500
PAPER_USE_REAL_BALANCE=false
PAPER_RESET_ON_START=true
```

`PAPER_BALANCE` — віртуальний депозит для навчання (не ваш реальний баланс).  
Після зміни суми один раз увімкніть `PAPER_RESET_ON_START=true`, або в Champaign — `CHAMPAIGN_RESET_STATE_ON_START=true` (скидає і `data/paper_wallet.json` до `PAPER_BALANCE`), потім можна знову `false`.

У логах: `[DRY RUN] WOULD ENTER` / `WOULD EXIT` з розбивкою витрат.  
Paper враховує: **taker fee** (вхід+вихід), **slippage**, **spread**, **funding** (rate з Binance, pro-rata).  
Стан: `data/paper_wallet.json` — поля `total_commission`, `total_slippage`, `total_spread`, `total_funding`.

### Paper exploration (навчання ML)

Поки `DRY_RUN=true`, за замовчуванням увімкнено **`PAPER_EXPLORATION=auto`**: бот **рішучіше** входить у угоди (віртуальні гроші), щоб швидше наповнити журнал і перевірити логіку. На live цей профіль **не застосовується**.

| Що змінюється | Навіщо |
|---------------|--------|
| `strict` EMA cross + RSI 40–65 / 35–60 | Менше «сліпих» long |
| Cooldown 15 хв після SL | Не ловити той самий символ знову |
| **Без daily loss cap** у paper | Більше даних для ML |
| SL/TP 1.15 / 1.65 ATR, R:R ≥ 1.5 | Краще співвідношення ризик/прибуток |
| ML short трохи легший поріг | Більше short-сигналів |

Вимкнути профіль: `PAPER_EXPLORATION=false`  
Вимкнути лише daily cap на paper: `PAPER_SKIP_DAILY_LOSS_LIMIT=false`

Якщо довго `TA=0` навіть з exploration — ринок у флеті; можна додатково `ML_DECISION_MODE=OFF` лише для діагностики TA.

## Швидкий старт

```bash
pip install -r requirements.txt
copy .env.example .env
# Відредагуйте .env: ключі + CONFIRM_LIVE_TRADING=true
python verify_setup.py
python main.py
```

### API ключі Binance (новий інтерфейс 2025–2026)

Binance показує сторінку **Binance API** (Spot / Derivatives / Testnet). Ключі створюються так:

1. Відкрийте [Binance API](https://www.binance.com/en/binance-api) → **Create / Manage API Key**  
   або одразу [API Management](https://www.binance.com/en/my/settings/api-management)
2. **Create API** → тип **System generated (HMAC)** — не Ed25519/RSA (наш бот працює лише з HMAC)
3. Збережіть **API Key** і **Secret Key** (Secret показують один раз)
4. Дозволи: **Reading** + **Futures (USDⓈ-M)**; **без** Withdrawals; бажано **IP whitelist**
5. Вставте в `.env` → `python verify_setup.py`

Детальна інструкція з скріншотами кроків: [docs/API_SETUP.md](docs/API_SETUP.md)

### PowerShell

```powershell
Copy-Item env.example.ps1 env.local.ps1
# відредагуйте env.local.ps1
.\start_bot.ps1
```

## Telegram (моніторинг з телефону)

Працює в **paper і live**: push про входи/виходи + команди в чаті.

1. У [@BotFather](https://t.me/BotFather) створіть бота → скопіюйте **token**.
2. Напишіть боту `/start`, потім відкрийте  
   `https://api.telegram.org/bot<TOKEN>/getUpdates` — знайдіть `"chat":{"id":...}`.
3. У `.env`:

```env
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
TELEGRAM_COMMANDS=true
TELEGRAM_DIGEST_MIN=30
TELEGRAM_NOTIFY_OPENS=true
TELEGRAM_NOTIFY_CLOSES=true
```

| Команда | Що показує |
|---------|------------|
| `/status` | Баланс, позиції, uPnL, paper stats |
| `/balance` | Короткий баланс |
| `/positions` | Відкриті угоди (SL/TP, stage, hedge) |
| `/history` | Останні події + журнал угод |
| `/stats` | Win rate, PnL vs старт |

`TELEGRAM_DIGEST_MIN=30` — авто-дайджест кожні 30 хв (0 = вимкнено).  
Історія подій: `data/activity.db`.

## Важливі змінні

| Змінна | Опис |
|--------|------|
| `BINANCE_API_KEY` / `BINANCE_API_SECRET` | Futures API |
| `CONFIRM_LIVE_TRADING` | `true` — дозвіл торгівлі на fapi.binance.com |
| `SCAN_ALL_PAIRS` | `true` — аналізувати всю біржу (ліквідні USDT пари) |
| `MAX_PAIRS_TO_SCAN` | Скільки пар тримати в universe (напр. 80) |
| `SYMBOLS` | Додаткові пари завжди в списку (або єдиний список якщо `SCAN_ALL_PAIRS=false`) |
| `LEVERAGE` | Плече (рекомендовано 3–10 для скальпінгу) |
| `RISK_PERCENT` | Ризик на угоду від доступного балансу |
| `MAX_POSITIONS` | Макс. одночасних позицій |
| `STRATEGY_MODE` | `strict` (рідкі сигнали) або `relaxed` (більше входів для paper) |
| `MIN_VOLUME_RATIO` | Мін. об’єм vs SMA (для тесту можна `0.95`) |
| `ML_DECISION_MODE` | `OFF` — лише TA, без ML-фільтра |

## Аварійне закриття

```bash
python scripts/close_all.py
```

## Testnet

У `.env` (офіційний Futures testnet):

```
BINANCE_FUTURES_BASE_URL=https://demo-fapi.binance.com
CONFIRM_LIVE_TRADING=false
```

Окремі testnet-ключі створюються на [testnet.binancefuture.com](https://testnet.binancefuture.com) — не плутати з live API Management.

## Структура

```
bot/
  config.py          # конфіг з env
  engine.py          # головний цикл
  exchange/          # REST + WebSocket
  strategy/scalper.py
  risk/manager.py
  trader.py
main.py
```

Торгівля криптовалютами несе ризик втрати капіталу. Використовуйте на власний розсуд.
