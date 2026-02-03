@echo off
REM Self-Learning Binance Futures Trading Bot - Startup Script
REM TESTNET ONLY

echo ============================================================
echo Self-Learning Binance USDT-M Futures Trading Bot
echo TESTNET ONLY - No Production Trading
echo ============================================================

REM API keys are built into config/settings.py
REM Telegram monitoring is ENABLED by default (requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
REM To disable: set TELEGRAM_ENABLED=False
echo.
echo Using Testnet API credentials from config
echo Telegram monitoring: ENABLED by default
echo.
echo Starting bot...
echo.

REM Optional log level - default to DEBUG for troubleshooting
set LOG_LEVEL=%1
if "%LOG_LEVEL%"=="" set LOG_LEVEL=DEBUG

python main.py --log-level %LOG_LEVEL%

pause
