@echo off
SETLOCAL
cd /d %~dp0

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Missing .venv\Scripts\python.exe
  exit /b 1
)

call .venv\Scripts\activate
.venv\Scripts\python.exe scheduled_report_select.py --mode previous-us-trading-day --run-sql --time-zone Europe/Bucharest
set "EXIT_CODE=%ERRORLEVEL%"

ENDLOCAL & exit /b %EXIT_CODE%
