@echo off
SETLOCAL
cd /d %~dp0
call .venv\Scripts\activate
streamlit run equipilot_app.py
ENDLOCAL
