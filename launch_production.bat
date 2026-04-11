@echo off
TITLE CORTEX-V STUDIO (Total Stability Lockdown)
SETLOCAL

:START
cls
echo [1/4] Calibrating Environment...
IF NOT EXIST .venv (
    echo [ERROR] Virtual environment missing.
    pause
    exit /b
)

echo [2/4] Verifying Production Assets...
IF NOT EXIST frontend\dist\index.html (
    echo [ERROR] Production build missing. Run 'npm run build' in frontend/ first.
    pause
    exit /b
)

echo [3/4] Scouring Port 8000 (Safety Guard)...
:: Use PowerShell for robust process identification and termination
powershell -Command "Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"

echo [4/4] Launching Unified Engine...
echo ----------------------------------------------------
echo SYSTEM READY. Running on Port 8000
echo Access via: http://localhost:8000
echo ----------------------------------------------------

:: Call uvicorn via python module directly
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --no-access-log --limit-concurrency 10 --timeout-keep-alive 5

:: If we reach here, the process terminated
echo [WARNING] AI Engine terminated unexpectedly.
echo [INFO] Self-healing sequence initializing in 10 seconds...
timeout /t 10 >nul
echo [INFO] Rebooting engine...
goto START
