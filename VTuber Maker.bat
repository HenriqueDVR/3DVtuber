@echo off
rem Starts the guided app (python -m app.server): its own window with pywebview installed, else the browser.
cd /d "%~dp0"
python -m app.server %*
if errorlevel 1 pause
