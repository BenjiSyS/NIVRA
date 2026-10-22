@echo off
rem Double-click: transcribes every video in this folder (see transcribe.py).
setlocal
cd /d "%~dp0"
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo Python is not installed. Install Python 3.12 from https://www.python.org/downloads/ ^(tick "Add to PATH"^), then run this again.
  pause
  exit /b 1
)
%PY% "%~dp0transcribe.py" %*
echo.
pause
