@echo off
SETLOCAL
cd /d %~dp0
call .venv\Scripts\activate
streamlit run monthly_scoring_app.py
ENDLOCAL
