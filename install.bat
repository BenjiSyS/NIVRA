@echo off
rem Nivra installer for Windows. Runs scripts\install.ps1 (no admin rights needed).
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1" %*
set ERR=%ERRORLEVEL%
if not "%ERR%"=="0" (
  echo.
  echo Installation did not complete. Run "install.bat" again and choose 4 for diagnostics.
)
pause
exit /b %ERR%
