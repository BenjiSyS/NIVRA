# Nivra installer logic for Windows (called by install.bat).
#   install.bat                  interactive menu
#   install.bat -Role app        everything for Nivra.exe (used by Nivra-Setup.bat)
#   install.bat -Role worker [-WithAI] [-Yes]
#   install.bat -Role controller -DatabaseUrl postgres://...
param(
  [ValidateSet('', 'app', 'controller', 'worker', 'repair', 'doctor')][string]$Role = '',
  [switch]$WithAI,
  [switch]$Yes,
  [switch]$NoShortcuts,
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
  if ($gpu) { Write-Host "    NVIDIA GPU: $($gpu.Name) (driver $($gpu.DriverVersion))" } else {
    $other = Get-CimInstance Win32_VideoController | Select-Object -First 1
    Write-Host "    no NVIDIA GPU detected ($($other.Name)); Nivra will use the CPU on this computer"
  }
}

function Refresh-Path {
  $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
}

# Ensure-Tool installs a missing tool with winget (winget shows its own
# licence prompts) and makes it usable in this session.
function Ensure-Tool($cmd, $wingetId, $label) {
  if (Get-Command $cmd -ErrorAction SilentlyContinue) { return }
  if (-not (Get-Command winget -ErrorAction SilentlyContinue)) { throw "$label is required. Install it, then run setup again." }
  Say "Installing $label (winget)"
  & winget install --id $wingetId -e --source winget
  Refresh-Path
  if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) { throw "$label was installed but is not usable yet. Close this window and run setup again." }
}

# Find-Python returns a command (array) for Python 3.11-3.14, skipping the
# Microsoft Store placeholder that only opens the Store.
function Find-Python {
  $check = 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 14) else 1)'
  foreach ($c in @(@('py', '-3.12'), @('py', '-3.13'), @('py', '-3.14'), @('py', '-3.11'), @('python'))) {
    if (-not (Get-Command $c[0] -ErrorAction SilentlyContinue)) { continue }
    $a = @($c | Select-Object -Skip 1) + @('-c', $check)
    if (Test-Native { & $c[0] @a }) { return , $c }
  }
  return $null
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
  $py = Find-Python
  if (-not $py) { Warn 'Python 3.11-3.14 not found (https://www.python.org/downloads/); AI jobs disabled.'; return '' }
  $venv = Join-Path $NivraHome 'venv'
  $vpy = Join-Path $venv 'Scripts\python.exe'
  if (-not (Test-Path $vpy)) {
    Say "Creating Python environment at $venv"
    $a = @($py | Select-Object -Skip 1) + @('-m', 'venv', $venv)
    & $py[0] @a | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'could not create the Python environment' }
  }
  # Native output goes to the host, never into this function's return value.
  & $vpy -m pip install --quiet --upgrade pip | Out-Host
  if (-not (Test-Native { & $vpy -c 'import torch' })) {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi -and (Test-Path 'C:\Windows\System32\nvidia-smi.exe')) { $smi = 'C:\Windows\System32\nvidia-smi.exe' }
    if ($smi) {
      # Newer CUDA wheels cover Turing (RTX 20xx, compute 7.5) and later;
      # older NVIDIA GPUs get the previous CUDA build.
      $cap = 0.0
      $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
      try { $q = & $smi --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1 } finally { $ErrorActionPreference = $old }
      [void][double]::TryParse(("$q").Trim(), [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$cap)
      $cu = if ($cap -ge 7.5) { 'cu128' } else { 'cu126' }
      Say "Installing PyTorch with CUDA support ($cu, GPU compute capability $cap; large download)"
      & $vpy -m pip install torch --index-url "https://download.pytorch.org/whl/$cu" | Out-Host
    } else {
      Say 'Installing CPU PyTorch (large download)'
      & $vpy -m pip install torch --index-url https://download.pytorch.org/whl/cpu | Out-Host
    }
    if ($LASTEXITCODE -ne 0) { Warn 'PyTorch installation failed; AI jobs stay disabled until setup is run again.'; return '' }
  }
  & $vpy -m pip install --quiet -r (Join-Path $Root 'ai\requirements.txt') | Out-Host
  $env:PYTHONPATH = Join-Path $Root 'ai'
  & $vpy -c "import nivra.cli, torch; print('    AI executor OK: torch', torch.__version__, 'cuda', torch.cuda.is_available())" | Out-Host
  if ($LASTEXITCODE -ne 0) { Warn 'AI executor self-test failed; run diagnostics.' }
  Remove-Item Env:PYTHONPATH
  return $vpy
}

# Test-Native runs a native command and reports success. Windows PowerShell
# 5.1 turns native stderr into terminating errors under 'Stop', so relax it.
function Test-Native([scriptblock]$Block) {
  $old = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try { & $Block 2>&1 | Out-Null; return ($LASTEXITCODE -eq 0) } finally { $ErrorActionPreference = $old }
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

function Build-App {
  Say 'Building Nivra (web interface, nivra.exe, Nivra.exe)'
  $running = Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.StartsWith($Root, [System.StringComparison]::OrdinalIgnoreCase) }
  if ($running) { throw 'Nivra is running from this folder. Quit it (Nivra -> This computer as a worker -> Quit Nivra), then run setup again.' }
  Push-Location web
  try {
    & npm ci --no-audit --no-fund
    if ($LASTEXITCODE -ne 0) { throw 'npm ci failed' }
    & npm run build
    if ($LASTEXITCODE -ne 0) { throw 'web build failed' }
  } finally { Pop-Location }
  $version = (Get-Content (Join-Path $Root 'VERSION') -ErrorAction SilentlyContinue | Select-Object -First 1)
  if (-not $version) { $version = '0.1.0-dev' }
  $commit = 'unknown'
  if (Test-Native { & git rev-parse --short HEAD }) { $commit = (& git rev-parse --short HEAD) }
  $ld = "-s -w -X nivra/internal/shared/versioning.Version=$version -X nivra/internal/shared/versioning.Commit=$commit"
  $env:CGO_ENABLED = '0'
  & go build -trimpath -ldflags $ld -o bin\nivra.exe .\cmd\nivra
  if ($LASTEXITCODE -ne 0) { throw 'go build (nivra.exe) failed' }
  & go build -trimpath -ldflags "$ld -H=windowsgui -X main.defaultCommand=app" -o Nivra.exe .\cmd\nivra
  if ($LASTEXITCODE -ne 0) { throw 'go build (Nivra.exe) failed' }
}

function Add-Shortcuts {
  $exe = Join-Path $Root 'Nivra.exe'
  $ws = New-Object -ComObject WScript.Shell
  foreach ($dir in @([Environment]::GetFolderPath('Desktop'), (Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'))) {
    if (-not (Test-Path $dir)) { continue }
    $lnk = $ws.CreateShortcut((Join-Path $dir 'Nivra.lnk'))
    $lnk.TargetPath = $exe
    $lnk.WorkingDirectory = $Root
    $lnk.Description = 'Nivra: generate videos and lend this computer to your Nivra network'
    $lnk.Save()
  }
}

# Install-App sets up everything Nivra.exe needs on this computer.
function Install-App {
  Preflight
  Write-Host ''
  Write-Host '    Missing tools are installed with winget: Go and Node.js LTS (to build Nivra), Python 3.12 + PyTorch (to run the model).'
  if ((Ask 'Continue? (y/n)' 'y') -ne 'y') { exit 1 }
  Refresh-Path
  Ensure-Tool go 'GoLang.Go' 'Go'
  Ensure-Tool npm 'OpenJS.NodeJS.LTS' 'Node.js'
  if (-not (Find-Python)) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) { throw 'Python 3.12 is required. Install it, then run setup again.' }
    Say 'Installing Python 3.12 (winget)'
    & winget install --id Python.Python.3.12 -e --source winget
    Refresh-Path
  }
  Build-App
  $py = Setup-Python
  if (-not $py) { Warn 'Python is not usable yet; Nivra.exe still works for generating on other computers. Run setup again to let this computer run AI jobs.' }
  Write-Config 'config\worker.json' ([ordered]@{
      ui_listen_addr = '127.0.0.1:8741'
      data_dir       = (Join-Path $NivraHome 'worker')
      python_path    = $py
      ai_package_dir = (Join-Path $Root 'ai')
    })
  if ($py) {
    # A config from an older install may lack the Python path.
    $cfgPath = Join-Path $Root 'config\worker.json'
    $cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json
    if (-not $cfg.python_path) {
      $cfg | Add-Member -NotePropertyName python_path -NotePropertyValue $py -Force
      [System.IO.File]::WriteAllText($cfgPath, ($cfg | ConvertTo-Json) + "`n", (New-Object System.Text.UTF8Encoding $false))
    }
  }
  if (-not $NoShortcuts) { Add-Shortcuts }
  & (Join-Path $Root 'bin\nivra.exe') doctor --role worker
  Say "Nivra is installed: $(Join-Path $Root 'Nivra.exe') (shortcuts on the Desktop and in the Start menu)."
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
  'app' { Install-App }
  'worker' { Install-Worker }
  'controller' { Install-Controller }
  'repair' { $bin = Find-Binary; & $bin repair; exit $LASTEXITCODE }
  'doctor' { Preflight; $bin = Find-Binary; & $bin doctor; exit $LASTEXITCODE }
}
