@echo off
TITLE CORTEX-V STUDIO // Unified Engine
SETLOCAL EnableDelayedExpansion

:: CONFIGURATION
SET PORT=8000
SET RETRY_COUNT=0
SET MAX_RETRIES=3

:START
cls
echo ====================================================
echo   CORTEX-V AI EMOTION ENGINE // PRODUCTION LAUNCHER
echo ====================================================

echo [1/4] CALIBRATING ENVIRONMENT...
IF NOT EXIST .venv (
    echo [ERROR] Virtual environment missing.
    echo Please run setup first.
    pause
    exit /b
)

:: Sanity check for TensorFlow
.venv\Scripts\python.exe -c "import tensorflow" 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python environment is unstable. 
    echo Detected broken dependencies or missing plugins.
    pause
    exit /b
)

echo [2/4] VERIFYING PRODUCTION ASSETS...
IF NOT EXIST frontend\dist\index.html (
    echo [ERROR] Production build missing. 
    echo Run 'npm run build' in frontend/ first.
    pause
    exit /b
)

echo [3/4] MANAGING NETWORK PORTS...
:: Surgical kill of port %PORT%
powershell -Command "Get-NetTCPConnection -LocalPort %PORT% -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"

echo [4/4] LAUNCHING UNIFIED ENGINE...
echo ----------------------------------------------------
echo SYSTEM READY. Listening on Port %PORT%
echo Access via: http://localhost:%PORT%
echo ----------------------------------------------------

:: Call uvicorn via python module directly
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port %PORT% --no-access-log --limit-concurrency 20 --timeout-keep-alive 10

:: If we reach here, the process terminated
echo [WARNING] AI Engine terminated unexpectedly.
SET /A RETRY_COUNT+=1

IF %RETRY_COUNT% GEQ %MAX_RETRIES% (
    echo [CRITICAL] System entered a crash loop. Stopping for safety.
    echo Please check backend logs and configuration.
    pause
    exit /b
)

echo [INFO] Self-healing sequence initializing (Attempt %RETRY_COUNT%/%MAX_RETRIES%)...
timeout /t 10 >nul
echo [INFO] Rebooting engine...
goto START
