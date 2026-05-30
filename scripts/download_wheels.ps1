param(
    [string]$PythonBin = $(if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }),
    [string]$WheelhouseDir = $(if ($env:WHEELHOUSE_DIR) { $env:WHEELHOUSE_DIR } else { "wheelhouse" })
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RootDir

New-Item -ItemType Directory -Force -Path $WheelhouseDir | Out-Null

Write-Host "Downloading Python wheels into: $WheelhouseDir"
Write-Host "Use this on a machine with internet, then copy the whole wheelhouse folder to the offline Windows machine."

& $PythonBin -m pip download --only-binary=:all: --dest $WheelhouseDir -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Error "Wheel download failed. Check internet connection, Python version, and pip."
}

Write-Host "Wheelhouse is ready: $WheelhouseDir"
