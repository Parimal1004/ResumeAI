@echo off
echo Starting ResumeAI Backend...
echo.
cd /d "%~dp0"
uvicorn backend.main:app --reload --port 8000
pause
