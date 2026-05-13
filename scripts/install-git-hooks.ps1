param()

$ErrorActionPreference = 'Stop'

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
Push-Location $repoRoot
try {
    git config core.hooksPath .githooks
    Write-Host "Configured git hooks path to .githooks"
    Write-Host "Pre-commit hook will now refresh docs anchors automatically."
}
finally {
    Pop-Location
}
