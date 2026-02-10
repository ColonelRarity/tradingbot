$ErrorActionPreference = "Stop"

# ----------------------------
# Binance Scalping Bot - ENV template (PowerShell)
# ----------------------------
# Заповніть значення і виконайте в PowerShell перед запуском:
#   .\env.example.ps1
#   python .\main.py
#
# ⚠️ Не комітьте файли з секретами. Цей файл — шаблон (без реальних ключів).
# Рекомендовано: скопіюйте цей файл в env.local.ps1 і заповніть там (env.local.ps1 ігнорується git).

# --- Binance API keys ---
$env:BINANCE_FUTURES_API_KEY = "your_futures_api_key"
$env:BINANCE_FUTURES_API_SECRET = "your_futures_api_secret"

# --- Production opt-in (be careful) ---
# За замовчуванням використовується Testnet. Для продакшн потрібно явно увімкнути:
# $env:BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"
# $env:BINANCE_FUTURES_WS_URL = "wss://fstream.binance.com"
# $env:ALLOW_PRODUCTION_TRADING = "True"

# (опційно) Demo:
# $env:BINANCE_DEMO_API_KEY = "your_demo_api_key"
# $env:BINANCE_DEMO_API_SECRET = "your_demo_api_secret"
#
# Примітка про DEMO (варіант 5 у меню):
# У цьому проєкті "DEMO" працює через testnet endpoint для futures, тому найпростіше
# використовувати саме FUTURES testnet ключі (BINANCE_FUTURES_API_*).

# --- Telegram (опційно) ---
# Щоб увімкнути Telegram-пуші:
# 1) заповніть TELEGRAM_BOT_TOKEN
# 2) встановіть TELEGRAM_ENABLED="True"
# 3) запустіть telegram_monitor разом з ботом, або встановіть TELEGRAM_CHAT_ID
$env:TELEGRAM_BOT_TOKEN = ""
$env:TELEGRAM_ENABLED = "False"
# $env:TELEGRAM_CHAT_ID = "123456789"

# --- Dry-run (імітація ордерів) ---
# $env:DRY_RUN = "True"

# --- Self-learning (опційно) ---
# $env:ENABLE_LEARNING = "True"
# --- ML decision maker (можна увімкнути без навчання) ---
# Режими:
# - OFF: класичні сигнали (TA)
# - FILTER: TA входи, але ML може блокувати слабкі
# - FULL: ML обирає BUY/SELL (для Futures), порівнюючи p_success(BUY) vs p_success(SELL)
# $env:ML_DECISION_MODE = "FULL"
# $env:ENABLE_ML_DECISION = "True"
#
# Модель:
# $env:LEARNING_MODEL_KIND = "MLP"   # LOGISTIC | MLP
# $env:LEARNING_HIDDEN = "16"
#
# Safety:
# $env:ML_DECISION_MIN_EDGE = "0.03"
# $env:ML_DECISION_REQUIRE_TA_SIGNAL = "True"
#
# $env:LEARNING_MIN_PROBA = "0.55"
# $env:LEARNING_WARMUP_SAMPLES = "50"
# $env:LEARNING_DB_PATH = "data\learning.db"
# $env:LEARNING_MODEL_PATH = "data\learning_model.json"

# --- Market scanning (опційно) ---
# Максимальна кількість пар для аналізу (за замовчуванням: 150)
# На Binance Futures є сотні USDT пар, ви можете збільшити це значення
# $env:MAX_PAIRS_TO_SCAN = "200"  # Аналізувати топ 200 пар за об'ємом

# --- SL/TP tuning (опційно) ---
# $env:SL_BREAKEVEN_TRIGGER_USDT = "10.0"
# $env:SL_BREAKEVEN_TRIGGER_PERCENT = "0.10"
# $env:SL_TRAIL_ACTIVATION_PROFIT_USDT = "1.0"
# $env:SL_TRAIL_DISTANCE_ATR_MULT = "0.8"
# $env:SL_MIN_PROFIT_BUFFER_USDT = "0.10"
