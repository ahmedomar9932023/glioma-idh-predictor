# Wish Candle - PowerShell Starter
# Run with: .\start.ps1  OR  Right-click > Run with PowerShell

$Host.UI.RawUI.WindowTitle = "Wish Candle"

Write-Host ""
Write-Host "  ============================================" -ForegroundColor DarkYellow
Write-Host "   WISH CANDLE - Dark Fantasy Wish Site" -ForegroundColor Yellow
Write-Host "  ============================================" -ForegroundColor DarkYellow
Write-Host ""

# Move to script's directory
Set-Location $PSScriptRoot

# Check Node.js
try {
    $nodeVersion = node --version 2>&1
    Write-Host "  Node.js: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Node.js not found. Install from https://nodejs.org" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Install dependencies if missing
if (-not (Test-Path "node_modules")) {
    Write-Host ""
    Write-Host "  [1/3] Installing dependencies..." -ForegroundColor Cyan
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: npm install failed." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "  Dependencies installed!" -ForegroundColor Green
}

# Create .env if missing
if (-not (Test-Path ".env")) {
    Write-Host ""
    Write-Host "  [2/3] Creating .env file..." -ForegroundColor Cyan
    Copy-Item ".env.example" ".env"
    Write-Host "  Created .env from .env.example" -ForegroundColor Green
    Write-Host "  IMPORTANT: Edit .env with your real Stripe/email keys!" -ForegroundColor Yellow
}

# Push database schema
Write-Host ""
Write-Host "  [2/3] Setting up database..." -ForegroundColor Cyan
npx prisma db push --skip-generate 2>&1 | Out-Null
Write-Host "  Database ready!" -ForegroundColor Green

# Start server
Write-Host ""
Write-Host "  [3/3] Starting Wish Candle..." -ForegroundColor Cyan
Write-Host ""
Write-Host "  ============================================" -ForegroundColor DarkYellow
Write-Host "   Open: http://localhost:3000" -ForegroundColor Yellow
Write-Host "   Press Ctrl+C to stop" -ForegroundColor DarkGray
Write-Host "  ============================================" -ForegroundColor DarkYellow
Write-Host ""

npm run dev
