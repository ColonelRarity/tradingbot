$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (Test-Path "env.local.ps1") {
    . .\env.local.ps1
} elseif (Test-Path ".env") {
    Write-Host "Using .env via python-dotenv"
} else {
    Write-Host "Warning: no env.local.ps1 or .env — copy .env.example first" -ForegroundColor Yellow
}

python verify_setup.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python main.py
