@echo off
echo Starting ResumeAI Frontend...
echo.
cd /d "%~dp0"
streamlit run frontend/app.py
pause
