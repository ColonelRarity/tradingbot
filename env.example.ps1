# Load before running the bot (PowerShell):
#   Copy-Item env.example.ps1 env.local.ps1
#   Edit env.local.ps1 with your keys
#   . .\env.local.ps1
#   python main.py

$ErrorActionPreference = "Stop"

$env:BINANCE_API_KEY = "YOUR_FUTURES_API_KEY"
$env:BINANCE_API_SECRET = "YOUR_FUTURES_API_SECRET"
$env:BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"
$env:BINANCE_WS_BASE = "wss://fstream.binance.com/ws"

# Required for real money
$env:TRADING_MODE = "live"
$env:CONFIRM_LIVE_TRADING = "true"

$env:SYMBOLS = "BTCUSDT,ETHUSDT"
$env:LEVERAGE = "5"
$env:MARGIN_TYPE = "ISOLATED"
$env:RISK_PERCENT = "0.5"
$env:MAX_POSITIONS = "2"
$env:LOG_LEVEL = "INFO"

$env:TELEGRAM_ENABLED = "true"
$env:TELEGRAM_BOT_TOKEN = "123456:ABC-your-bot-token"
$env:TELEGRAM_CHAT_ID = "your_numeric_chat_id"
$env:TELEGRAM_COMMANDS = "true"
$env:TELEGRAM_DIGEST_MIN = "30"
$env:TELEGRAM_NOTIFY_OPENS = "true"
$env:TELEGRAM_NOTIFY_CLOSES = "true"

Write-Host "Environment loaded. Run: python verify_setup.py" -ForegroundColor Green
