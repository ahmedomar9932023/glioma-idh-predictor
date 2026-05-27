@echo off
title Wish Candle - Starting...
color 0E

echo.
echo  ============================================
echo   WISH CANDLE - Dark Fantasy Wish Site
echo  ============================================
echo.

:: Check if node_modules exists
if not exist "node_modules\" (
    echo  [1/3] Installing dependencies...
    call npm install
    if errorlevel 1 (
        echo  ERROR: npm install failed. Make sure Node.js is installed.
        pause
        exit /b 1
    )
    echo  Dependencies installed!
    echo.
)

:: Check if .env exists
if not exist ".env" (
    echo  [2/3] Setting up .env file...
    copy ".env.example" ".env" >nul
    echo  Created .env from .env.example
    echo  IMPORTANT: Edit .env with your real keys before using Stripe!
    echo.
)

:: Push database schema
echo  [2/3] Setting up database...
call npx prisma db push --skip-generate >nul 2>&1
if errorlevel 1 (
    call npx prisma db push >nul 2>&1
)
echo  Database ready!
echo.

:: Check if DB needs seeding (optional)
for /f %%i in ('dir /b "prisma\dev.db" 2^>nul ^| find /c "dev.db"') do set DB_EXISTS=%%i
if "%DB_EXISTS%"=="0" (
    echo  Seeding database with sample wishes...
    call npx tsx prisma/seed.ts >nul 2>&1
)

:: Start dev server
echo  [3/3] Starting Wish Candle...
echo.
echo  ============================================
echo   Open: http://localhost:3000
echo   Press Ctrl+C to stop
echo  ============================================
echo.

call npm run dev
pause
