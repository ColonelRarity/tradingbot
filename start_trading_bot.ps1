# Self-Learning Binance Futures Trading Bot - Startup Script
# TESTNET ONLY

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Self-Learning Binance USDT-M Futures Trading Bot" -ForegroundColor Cyan  
Write-Host "TESTNET ONLY - No Production Trading" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan

# Load environment variables if env.local.ps1 exists
if (Test-Path ".\env.local.ps1") {
    Write-Host "Loading environment variables from env.local.ps1..." -ForegroundColor Green
    . .\env.local.ps1
}

# API keys are now built into config/settings.py
# Environment variables are optional overrides
Write-Host ""
Write-Host "Using Testnet API credentials from config" -ForegroundColor Green
Write-Host ""

# Optional: Set log level
$LOG_LEVEL = if ($args[0]) { $args[0] } else { "INFO" }
Write-Host "Log Level: $LOG_LEVEL" -ForegroundColor Cyan

Write-Host ""
Write-Host "Starting bot (multi-pair mode)..." -ForegroundColor Green
Write-Host ""

# Run the bot
python main.py --log-level $LOG_LEVEL
