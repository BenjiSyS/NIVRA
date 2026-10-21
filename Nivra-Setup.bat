@echo off
rem ======================================================================
rem  Nivra one-file setup for Windows 10/11 (64-bit). No admin rights needed
rem  except for what winget itself may ask when it installs a missing tool.
rem
rem  Send only this file. It:
rem    1. installs Git if missing (winget, after asking),
rem    2. clones (or updates) Nivra from GitHub into %USERPROFILE%\Nivra,
rem    3. runs scripts\install.ps1 -Role app, which installs Go, Node.js and
rem       Python if missing, detects the hardware (NVIDIA GPU -> CUDA PyTorch,
rem       otherwise CPU PyTorch), builds everything and creates Nivra.exe in
rem       the Nivra folder plus Desktop/Start menu shortcuts.
rem
rem  Usage:  Nivra-Setup.bat [install folder]
rem  Run it again later to update. Set NIVRA_SETUP_YES=1 to answer yes to
rem  every question (unattended install).
rem ======================================================================
setlocal EnableExtensions
set "NIVRA_REPO_URL=https://github.com/BenjiSyS/nivra.git"
if defined NIVRA_REPO set "NIVRA_REPO_URL=%NIVRA_REPO%"
set "NIVRA_BRANCH=main"
set "DEST=%~1"
if "%DEST%"=="" set "DEST=%USERPROFILE%\Nivra"
set "PSYES="
if "%NIVRA_SETUP_YES%"=="1" set "PSYES=-Yes"

echo.
echo   NIVRA SETUP
echo   Source : %NIVRA_REPO_URL% (%NIVRA_BRANCH%)
echo   Folder : %DEST%
echo.

where winget >nul 2>nul
if errorlevel 1 (
  echo !! winget was not found. Install "App Installer" from the Microsoft Store,
  echo    or install Git, Go, Node.js LTS and Python 3.12 yourself, then run this again.
  set "NOWINGET=1"
)

rem ---- Git ----
set "GIT=git"
where git >nul 2>nul
if errorlevel 1 (
  if exist "%ProgramFiles%\Git\cmd\git.exe" (
    set "GIT=%ProgramFiles%\Git\cmd\git.exe"
  ) else (
    if defined NOWINGET goto :nogit
    if not "%NIVRA_SETUP_YES%"=="1" (
      choice /C YN /M "Git is needed to download Nivra. Install Git with winget now"
      if errorlevel 2 goto :nogit
    )
    winget install --id Git.Git -e --source winget
    if exist "%ProgramFiles%\Git\cmd\git.exe" (
      set "GIT=%ProgramFiles%\Git\cmd\git.exe"
    ) else (
      goto :nogit
    )
  )
)

rem ---- Clone or update ----
if exist "%DEST%\.git" (
  echo ==^> Updating %DEST%
  "%GIT%" -C "%DEST%" pull --ff-only
  if errorlevel 1 (
    echo !! Could not update. Local changes in %DEST%? Fix them or move the folder, then run again.
    goto :fail
  )
) else (
  if exist "%DEST%\*" (
    dir /b "%DEST%" | findstr . >nul && (
      echo !! %DEST% exists and is not a Nivra download. Choose another folder:  Nivra-Setup.bat D:\Nivra
      goto :fail
    )
  )
  echo ==^> Downloading Nivra into %DEST%
  "%GIT%" clone --depth 1 --branch %NIVRA_BRANCH% "%NIVRA_REPO_URL%" "%DEST%"
  if errorlevel 1 (
    echo !! Download failed. Check the internet connection and that the repository is public.
    goto :fail
  )
)

rem ---- Install everything and build Nivra.exe ----
powershell -NoProfile -ExecutionPolicy Bypass -File "%DEST%\scripts\install.ps1" -Role app %PSYES%
if errorlevel 1 goto :fail

echo.
echo   Done. Open Nivra with the "Nivra" shortcut or %DEST%\Nivra.exe
echo.
if "%NIVRA_SETUP_YES%"=="1" exit /b 0
choice /C YN /M "Open Nivra now"
if errorlevel 2 goto :end
start "" "%DEST%\Nivra.exe"
goto :end

:nogit
echo !! Git is required. Install it from https://git-scm.com/download/win and run this file again.
:fail
echo.
echo   Setup did not finish. The messages above say why; run this file again after fixing it.
if not "%NIVRA_SETUP_YES%"=="1" pause
exit /b 1

:end
pause
exit /b 0
