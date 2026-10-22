@echo off
rem ======================================================================
rem  Nivra one-file setup for Windows 10/11 (64-bit). No admin rights needed
rem  except for what winget itself may ask when it installs a missing tool.
rem
rem  Send only this file. It:
rem    1. refuses to continue while Nivra.exe is running (close it first),
rem    2. downloads the latest tagged GitHub release, verifies its SHA-256
rem       against the SHA256SUMS of the same release and unpacks it into
rem       %USERPROFILE%\Nivra. No Git, Go or Node.js is needed on this path,
rem    3. only if no release is available (or it cannot be verified): installs
rem       Git if missing and builds from source (clone/update of the repository),
rem    4. runs scripts\install.ps1 -Role app: Python, PyTorch (CUDA when an
rem       NVIDIA GPU exists, otherwise CPU), ffmpeg, Desktop/Start menu shortcuts.
rem
rem  Usage:  Nivra-Setup.bat [install folder]
rem  Run it again later to update. Environment:
rem    NIVRA_REPO=owner/name  GitHub repository (default: the REPO line below)
rem    NIVRA_SETUP_YES=1      answer yes to every question (unattended)
rem    NIVRA_SETUP_SOURCE=1   skip the release download and build from source
rem ======================================================================
setlocal EnableExtensions

rem ---- Set this to the GitHub repository that publishes the releases (owner/name). ----
set "REPO=BenjiSyS/NIVRA"
if defined NIVRA_REPO set "REPO=%NIVRA_REPO%"
if /I "%REPO%"=="OWNER/nivra" (
  echo !! No repository configured. Run:  set NIVRA_REPO=owner/name ^& Nivra-Setup.bat
  echo    or edit the REPO line at the top of this file.
  goto :fail
)
set "NIVRA_REPO_URL=https://github.com/%REPO%.git"
set "NIVRA_BRANCH=main"
set "DEST=%~1"
if "%DEST%"=="" set "DEST=%USERPROFILE%\Nivra"
if "%DEST:~-1%"=="\" set "DEST=%DEST:~0,-1%"
set "PSYES="
if "%NIVRA_SETUP_YES%"=="1" set "PSYES=-Yes"
set "NIVRA_BAT=%~f0"
set "NIVRA_DEST=%DEST%"
set "NIVRA_REPO=%REPO%"

echo.
echo   NIVRA SETUP
echo   Source : %REPO% (latest release, or %NIVRA_BRANCH% as fallback)
echo   Folder : %DEST%
echo.

rem ---- Nothing is touched while Nivra is running from the install folder ----
set "NIVRA_STAGE=check"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$t=[IO.File]::ReadAllText($env:NIVRA_BAT); $i=$t.IndexOf('#'+'PS-BOOTSTRAP-BEGIN'); Invoke-Expression $t.Substring($i)"
if errorlevel 1 goto :fail

rem ---- Preferred path: verified release package ----
if "%NIVRA_SETUP_SOURCE%"=="1" goto :source
set "NIVRA_STAGE=release"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$t=[IO.File]::ReadAllText($env:NIVRA_BAT); $i=$t.IndexOf('#'+'PS-BOOTSTRAP-BEGIN'); Invoke-Expression $t.Substring($i)"
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" goto :done
if not "%RC%"=="10" goto :fail
echo.
echo ==^> No usable release package; building from source instead.

:source
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

rem ---- Clone or update (Nivra was already checked not to be running) ----
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
      echo !! %DEST% exists and is not a Nivra source download. Choose another folder:  Nivra-Setup.bat D:\Nivra
      goto :fail
    )
  )
  echo ==^> Downloading Nivra into %DEST%
  "%GIT%" clone --depth 1 --branch %NIVRA_BRANCH% "%NIVRA_REPO_URL%" "%DEST%"
  if errorlevel 1 (
    echo !! Download failed. Check the internet connection, a proxy or firewall, and that the repository is public.
    goto :fail
  )
)

rem ---- Install everything and build Nivra.exe ----
powershell -NoProfile -ExecutionPolicy Bypass -File "%DEST%\scripts\install.ps1" -Role app %PSYES%
if errorlevel 1 goto :fail

:done
echo.
echo   Done. Open Nivra with the "Nivra" shortcut or %DEST%\Nivra.exe
echo.
if "%NIVRA_SETUP_YES%"=="1" exit /b 0
choice /C YN /M "Open Nivra now"
if errorlevel 2 goto :end
start "" "%DEST%\Nivra.exe"
goto :end

:nogit
echo !! Git is required for the source build. Install it from https://git-scm.com/download/win and run this file again.
:fail
echo.
echo   Setup did not finish. The messages above say why; run this file again after fixing it.
if not "%NIVRA_SETUP_YES%"=="1" pause
exit /b 1

:end
pause
exit /b 0

rem ======================================================================
rem  Everything below is PowerShell, run by the lines above; cmd never gets here.
rem ======================================================================
#PS-BOOTSTRAP-BEGIN
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$dest = $env:NIVRA_DEST.TrimEnd('\')
$repo = $env:NIVRA_REPO
$yes = ($env:NIVRA_SETUP_YES -eq '1')

function Say($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "!!  $m" -ForegroundColor Yellow }

if ($env:NIVRA_STAGE -eq 'check') {
  $running = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -ieq 'nivra' -and $_.Path -and $_.Path.StartsWith($dest + '\', [System.StringComparison]::OrdinalIgnoreCase) })
  if ($running.Count -gt 0) {
    Write-Host ''
    Write-Host "!! Nivra is running from $dest ($($running[0].ProcessName))." -ForegroundColor Yellow
    Write-Host '   Close it first (Nivra window > This computer as a worker > Quit Nivra, or the tray icon > Quit),'
    Write-Host '   then run this file again. Nothing has been changed.'
    exit 20
  }
  exit 0
}

# ---- release stage: exit 0 = installed, 10 = fall back to source, other = failed ----
if (Test-Path (Join-Path $dest '.git')) {
  Say "$dest is a source install; updating it from git"
  exit 10
}
if ($repo -notmatch '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$') { Warn "invalid repository name: $repo"; exit 10 }

$tmp = Join-Path ([IO.Path]::GetTempPath()) ('nivra-setup-' + [guid]::NewGuid().ToString('N'))
try {
  [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
  New-Item -ItemType Directory -Force $tmp | Out-Null
  Say "Looking for the latest release of $repo"
  $hdr = @{ 'User-Agent' = 'Nivra-Setup'; 'Accept' = 'application/vnd.github+json' }
  $rel = Invoke-RestMethod -UseBasicParsing -Headers $hdr -Uri "https://api.github.com/repos/$repo/releases/latest" -TimeoutSec 30
  $tag = [string]$rel.tag_name
  if ($tag -notmatch '^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$') { throw "unexpected release tag '$tag'" }
  $zipAsset = @($rel.assets | Where-Object { $_.name -match '^nivra-[A-Za-z0-9._+-]+-windows-amd64\.zip$' }) | Select-Object -First 1
  $sumAsset = @($rel.assets | Where-Object { $_.name -eq 'SHA256SUMS' }) | Select-Object -First 1
  if (-not $zipAsset -or -not $sumAsset) { throw "release $tag has no Windows package or no SHA256SUMS" }
  $zipName = [string]$zipAsset.name
  # The download URL is built from the pinned tag, not taken from the response.
  $base = "https://github.com/$repo/releases/download/$tag"
  Say "Downloading $zipName ($tag, $([math]::Round($zipAsset.size / 1MB, 1)) MB)"
  $zipPath = Join-Path $tmp $zipName
  $sumPath = Join-Path $tmp 'SHA256SUMS'
  Invoke-WebRequest -UseBasicParsing -Headers @{ 'User-Agent' = 'Nivra-Setup' } -Uri "$base/SHA256SUMS" -OutFile $sumPath -TimeoutSec 60
  Invoke-WebRequest -UseBasicParsing -Headers @{ 'User-Agent' = 'Nivra-Setup' } -Uri "$base/$zipName" -OutFile $zipPath -TimeoutSec 1800

  Say 'Verifying SHA-256'
  $want = $null
  foreach ($line in Get-Content $sumPath) {
    if ($line -match '^([0-9A-Fa-f]{64})\s+\*?(.+?)\s*$' -and $Matches[2] -eq $zipName) { $want = $Matches[1].ToLowerInvariant() }
  }
  if (-not $want) { throw "SHA256SUMS of $tag has no entry for $zipName" }
  $got = (Get-FileHash -Algorithm SHA256 -Path $zipPath).Hash.ToLowerInvariant()
  if ($got -ne $want) { throw "CHECKSUM MISMATCH for $zipName (expected $want, got $got). The download is damaged or was tampered with." }
  Write-Host "    ok: $got"

  Say 'Unpacking'
  $x = Join-Path $tmp 'x'
  Expand-Archive -LiteralPath $zipPath -DestinationPath $x -Force
  foreach ($f in @('Nivra.exe', 'bin\nivra.exe', 'scripts\install.ps1', 'ai\nivra\__init__.py', 'VERSION')) {
    if (-not (Test-Path (Join-Path $x $f))) { throw "the package is incomplete (missing $f)" }
  }
  New-Item -ItemType Directory -Force $dest | Out-Null
  # Replace the Python package wholesale (no stale files); config, logs and data are not part of the package.
  if (Test-Path (Join-Path $dest 'ai')) { Remove-Item -Recurse -Force (Join-Path $dest 'ai') }
  Copy-Item -Path (Join-Path $x '*') -Destination $dest -Recurse -Force
  Say "Nivra $tag unpacked into $dest"
} catch {
  Write-Host ''
  if ("$($_.Exception.Message)" -match '\(404\)') {
    # No release published yet: not an error, the source build is the normal path then.
    Say "No release package published for $repo yet"
    if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue }
    exit 10
  }
  Warn "Release download failed: $($_.Exception.Message)"
  Write-Host '   If you are behind a proxy, set HTTPS_PROXY (for example  set HTTPS_PROXY=http://proxy:8080)  and run this again.'
  Write-Host '   Antivirus or a firewall can block the download or quarantine Nivra.exe: allow the file or the folder, then retry.'
  Write-Host '   If this is a checksum problem, do not use the file; wait for a new release.'
  if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue }
  exit 10
}
Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue

try {
  $inst = Join-Path $dest 'scripts\install.ps1'
  if ($yes) { & $inst -Role app -Yes } else { & $inst -Role app }
  if ($LASTEXITCODE -ne 0) { throw "install.ps1 exited with code $LASTEXITCODE" }
} catch {
  Write-Host ''
  Write-Host "!! $($_.Exception.Message)" -ForegroundColor Red
  exit 1
}
exit 0
