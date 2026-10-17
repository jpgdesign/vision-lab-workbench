# Default is diagnosis only. Example: .\setup.ps1 --install --profile full
$ErrorActionPreference = 'Stop'
$taskSetup = Join-Path $PSScriptRoot 'setup.py'
if ($env:VISION_WORKBENCH_PYTHON) {
    & $env:VISION_WORKBENCH_PYTHON $taskSetup @args
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python $taskSetup @args
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 $taskSetup @args
} else {
    Write-Error 'Install Python 3.11 (supported: 3.10-3.12), then run setup.py again.'
    exit 1
}
exit $LASTEXITCODE
