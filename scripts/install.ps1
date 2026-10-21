# Nivra installer logic for Windows (called by install.bat).
#   install.bat                  interactive menu
#   install.bat -Role worker [-WithAI] [-Yes]
#   install.bat -Role controller -DatabaseUrl postgres://...
param(
  [ValidateSet('', 'controller', 'worker', 'repair', 'doctor')][string]$Role = '',
  [switch]$WithAI,
  [switch]$Yes,
  [string]$DatabaseUrl = $env:NIVRA_DATABASE_URL
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$NivraHome = if ($env:NIVRA_HOME) { $env:NIVRA_HOME } else { Join-Path $env:LOCALAPPDATA 'Nivra' }

function Say($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "!!  $m" -ForegroundColor Yellow }
function Ask($q, $d) { if ($Yes) { return $d }; $r = Read-Host "$q [$d]"; if ([string]::IsNullOrWhiteSpace($r)) { $d } else { $r } }

function Preflight {
  Say 'Preflight checks'
  $os = [System.Environment]::OSVersion.Version
  if ($os.Major -lt 10) { throw 'Windows 10 or newer is required.' }
  if (-not [Environment]::Is64BitOperatingSystem) { throw '64-bit Windows is required.' }
  $mem = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1MB)
  Write-Host "    memory: $mem MiB"
  New-Item -ItemType Directory -Force $NivraHome | Out-Null
  $drive = (Get-Item $NivraHome).PSDrive
  $free = [math]::Round($drive.Free / 1GB, 1)
  Write-Host "    free disk on ${drive}: $free GB"
  if ($free -lt 5) { Warn 'less than 5 GB free disk space' }
  $gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA' } | Select-Object -First 1
  if ($gpu) { Write-Host "    NVIDIA GPU: $($gpu.Name) (driver $($gpu.DriverVersion))" } else { Write-Host '    no NVIDIA GPU detected (CPU-only worker)' }
}

function Find-Binary {
  foreach ($c in @('bin\nivra.exe', 'bin\nivra-windows-amd64.exe', 'dist\bin\nivra-windows-amd64.exe')) {
    $p = Join-Path $Root $c
    if (Test-Path $p) { return $p }
  }
  if (Get-Command go -ErrorAction SilentlyContinue) {
    Say 'Building Nivra from source'
    $bash = Get-Command bash -ErrorAction SilentlyContinue
    if ($bash) { & bash scripts/build.sh } else {
      if (Get-Command npm -ErrorAction SilentlyContinue) { Push-Location web; npm ci; npm run build; Pop-Location }
      $env:CGO_ENABLED = '0'; & go build -trimpath -o bin\nivra.exe .\cmd\nivra
    }
    return (Join-Path $Root 'bin\nivra.exe')
  }
  throw 'No prebuilt nivra.exe found and Go is not installed. Download a release package or install Go 1.27+.'
}

function Setup-Python {
  $py = Get-Command python -ErrorAction SilentlyContinue
  if (-not $py) { Warn 'Python 3.11+ not found (https://www.python.org/downloads/); AI jobs disabled.'; return '' }
  $venv = Join-Path $NivraHome 'venv'
  $vpy = Join-Path $venv 'Scripts\python.exe'
  if (-not (Test-Path $vpy)) { Say "Creating Python environment at $venv"; & python -m venv $venv }
  & $vpy -m pip install --quiet --upgrade pip
  & $vpy -c 'import torch' 2>$null
  if ($LASTEXITCODE -ne 0) {
    $hasNvidia = (Get-Command nvidia-smi -ErrorAction SilentlyContinue) -or (Test-Path 'C:\Windows\System32\nvidia-smi.exe')
    if ($hasNvidia) {
      Say 'Installing PyTorch with CUDA support (large download)'
      & $vpy -m pip install torch --index-url https://download.pytorch.org/whl/cu126
    } else {
      Say 'Installing CPU PyTorch (large download)'
      & $vpy -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    }
  }
  & $vpy -m pip install --quiet -r (Join-Path $Root 'ai\requirements.txt')
  $env:PYTHONPATH = Join-Path $Root 'ai'
  & $vpy -c "import nivra.cli, torch; print('    AI executor OK: torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  if ($LASTEXITCODE -ne 0) { Warn 'AI executor self-test failed; run diagnostics.' }
  Remove-Item Env:PYTHONPATH
  return $vpy
}

function Write-Config($path, $obj) {
  if (Test-Path $path) { Write-Host "    keeping existing $path"; return }
  New-Item -ItemType Directory -Force (Split-Path $path) | Out-Null
  $json = $obj | ConvertTo-Json
  [System.IO.File]::WriteAllText((Join-Path $Root $path), $json + "`n", (New-Object System.Text.UTF8Encoding $false))
  Write-Host "    wrote $path"
}

function Install-Worker {
  Preflight
  $bin = Find-Binary
  $py = ''
  if ($WithAI -or ((Ask 'Enable AI jobs (installs PyTorch, 1-3 GB download)? (y/n)' 'y') -eq 'y')) { $py = Setup-Python }
  Write-Config 'config\worker.json' ([ordered]@{
      ui_listen_addr = '127.0.0.1:8741'
      data_dir       = (Join-Path $NivraHome 'worker')
      python_path    = $py
      ai_package_dir = (Join-Path $Root 'ai')
    })
  Say 'Worker installed.'
  Write-Host '    Start it with start.bat, then enter your project code in the page that opens (http://127.0.0.1:8741/).'
}

function Install-Controller {
  Preflight
  $bin = Find-Binary
  if (-not $DatabaseUrl) {
    Warn 'The Windows controller needs an existing PostgreSQL 13+ database (the Raspberry Pi is the primary controller host).'
    $DatabaseUrl = Read-Host 'PostgreSQL URL (postgres://user:pass@host:port/db)'
  }
  $secrets = Join-Path $NivraHome 'controller\secrets'
  New-Item -ItemType Directory -Force $secrets | Out-Null
  $envFile = Join-Path $secrets 'database.env'
  if (-not (Test-Path $envFile)) {
    [System.IO.File]::WriteAllText($envFile, "NIVRA_DATABASE_URL=$DatabaseUrl`n")
    icacls $envFile /inheritance:r /grant:r "${env:USERNAME}:(R,W)" | Out-Null
  }
  $py = Setup-Python
  Write-Config 'config\controller.json' ([ordered]@{
      listen_addr    = '127.0.0.1:8740'
      data_dir       = (Join-Path $NivraHome 'controller')
      python_path    = $py
      ai_package_dir = (Join-Path $Root 'ai')
    })
  & $bin controller init --config config\controller.json
  if ($LASTEXITCODE -ne 0) { throw 'controller init failed' }
  $name = Ask 'Name of the first project (leave as-is if one exists)' 'Nivra'
  & $bin project create --name $name 2>$null
  if ((Ask 'Create an administrator account now? (y/n)' 'y') -eq 'y') {
    $u = Ask 'Administrator username' 'admin'
    & $bin admin create --username $u
  }
  Say 'Controller installed. Start it with: start.bat controller   then open http://127.0.0.1:8740/'
}

if (-not $Role) {
  Write-Host ''
  Write-Host '  NIVRA SETUP'
  Write-Host '  1 - Controller'
  Write-Host '  2 - Worker'
  Write-Host '  3 - Repair installation'
  Write-Host '  4 - System diagnostics'
  Write-Host '  5 - Exit'
  $c = Read-Host 'Choose [1-5]'
  $Role = @{ '1' = 'controller'; '2' = 'worker'; '3' = 'repair'; '4' = 'doctor' }[$c]
  if (-not $Role) { exit 0 }
}
switch ($Role) {
  'worker' { Install-Worker }
  'controller' { Install-Controller }
  'repair' { $bin = Find-Binary; & $bin repair; exit $LASTEXITCODE }
  'doctor' { Preflight; $bin = Find-Binary; & $bin doctor; exit $LASTEXITCODE }
}
