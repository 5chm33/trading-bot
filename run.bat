@echo off
setlocal
title Trading Bot Console
color 0A
set PYTHONPATH=%PYTHONPATH%;%~dp0

echo Starting Trading Bot...
echo -----------------------

python scripts/run_paper_trading.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Bot crashed with code %errorlevel%
    timeout /t 30
) else (
    echo.
    echo Bot finished successfully
    timeout /t 10
)