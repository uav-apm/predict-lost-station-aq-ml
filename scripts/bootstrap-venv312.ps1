param(
    [string]$VenvPath = ".venv312",
    [switch]$Activate
)

$ErrorActionPreference = "Stop"

$workspaceRoot = Split-Path -Parent $PSScriptRoot
Set-Location $workspaceRoot

$gitCommand = Get-Command git -ErrorAction SilentlyContinue
if ($gitCommand) {
    $hooksPath = (& git config --get core.hooksPath 2>$null)
    if ([string]::IsNullOrWhiteSpace($hooksPath) -or $hooksPath.Trim() -ne ".githooks") {
        & git config core.hooksPath .githooks
        Write-Host "Configured git hooks path to .githooks"
    }
}

$venvFullPath = Join-Path $workspaceRoot $VenvPath
$activateScript = Join-Path $venvFullPath "Scripts\Activate.ps1"

if (-not (Test-Path $venvFullPath)) {
    $pyCommand = Get-Command py -ErrorAction SilentlyContinue
    if (-not $pyCommand) {
        Write-Error "Python launcher 'py' was not found. Install Python 3.12 and ensure 'py -3.12' works."
    }

    try {
        & py -3.12 -c "import sys; assert sys.version_info[:2] == (3, 12)"
    }
    catch {
        Write-Error "Python 3.12 is required to bootstrap the LSTM/TensorFlow venv. Install Python 3.12 and retry."
    }

    Write-Host "Creating virtual environment at $venvFullPath using Python 3.12..."
    & py -3.12 -m venv $venvFullPath
}

if ($Activate) {
    if (-not (Test-Path $activateScript)) {
        Write-Error "Activation script was not found at $activateScript."
    }

    if ($env:VIRTUAL_ENV -ne $venvFullPath) {
        . $activateScript
    }

    Write-Host "Active virtual environment: $env:VIRTUAL_ENV"
}
