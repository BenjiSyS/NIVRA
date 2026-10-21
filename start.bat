@echo off
rem Start Nivra in this window. Close the window or press Ctrl+C to stop cleanly.
rem   start.bat            worker if configured, otherwise controller
rem   start.bat worker
rem   start.bat controller
setlocal
cd /d "%~dp0"
set "NIVRA=%~dp0bin\nivra.exe"
if not exist "%NIVRA%" set "NIVRA=%~dp0bin\nivra-windows-amd64.exe"
if not exist "%NIVRA%" set "NIVRA=%~dp0dist\bin\nivra-windows-amd64.exe"
if not exist "%NIVRA%" (
  echo Nivra is not installed yet. Run install.bat first.
  pause
  exit /b 1
)
set "ROLE=%~1"
if "%ROLE%"=="" if exist "config\worker.json" set "ROLE=worker"
if "%ROLE%"=="" if exist "config\controller.json" set "ROLE=controller"
if "%ROLE%"=="" (
  echo No configuration found. Run install.bat first.
  pause
  exit /b 1
)
if /I "%ROLE%"=="worker" (
  "%NIVRA%" worker run --config config\worker.json --open
) else (
  "%NIVRA%" controller run --config config\controller.json
)
pause
