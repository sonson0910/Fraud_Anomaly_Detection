param(
    [string]$DataDir = $env:DATA_DIR,
    [string]$PythonBin = $env:PYTHON_BIN,
    [string]$VenvDir = $env:VENV_DIR,
    [int]$Port = $(if ($env:PORT) { [int]$env:PORT } else { 8501 }),
    [string]$NotebookPath = $env:NOTEBOOK_PATH,
    [string]$ForceRebuild = $env:FORCE_REBUILD,
    [string]$WheelhouseDir = $env:WHEELHOUSE_DIR,
    [string]$SkipInstall = $env:SKIP_INSTALL,
    [string]$UseSystemSitePackages = $env:USE_SYSTEM_SITE_PACKAGES
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RootDir

if ([string]::IsNullOrWhiteSpace($DataDir)) {
    $DataDir = "Processed_Data"
}
if ([string]::IsNullOrWhiteSpace($PythonBin)) {
    $PythonBin = "python"
}
if ([string]::IsNullOrWhiteSpace($VenvDir)) {
    $VenvDir = ".venv"
}
if ([string]::IsNullOrWhiteSpace($WheelhouseDir)) {
    $WheelhouseDir = "wheelhouse"
}
if ([string]::IsNullOrWhiteSpace($SkipInstall)) {
    $SkipInstall = "0"
}
if ([string]::IsNullOrWhiteSpace($UseSystemSitePackages)) {
    $UseSystemSitePackages = "0"
}
if ([string]::IsNullOrWhiteSpace($NotebookPath)) {
    $NotebookPath = "Vong_3_EAZII_2.ipynb"
}
if ([string]::IsNullOrWhiteSpace($ForceRebuild)) {
    $ForceRebuild = "0"
}

$VenvPython = Join-Path $VenvDir "Scripts/python.exe"
$VenvStreamlit = Join-Path $VenvDir "Scripts/streamlit.exe"

if (-not (Test-Path $VenvPython)) {
    if ($UseSystemSitePackages -eq "1") {
        & $PythonBin -m venv --system-site-packages $VenvDir
    } else {
        & $PythonBin -m venv $VenvDir
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Cannot create Python virtual environment. Check that Python is installed and available as '$PythonBin'."
    }
}

function Test-DemoDependencies {
    param([string]$PythonExe)
    & $PythonExe -c "import pandas,numpy,sklearn,matplotlib,seaborn,plotly,openpyxl,nbformat,nbconvert,xgboost,streamlit,shap; print('Python dependencies OK')"
    return ($LASTEXITCODE -eq 0)
}

function Install-DemoDependencies {
    param([string]$PythonExe, [string]$Wheelhouse)

    if (Test-Path $Wheelhouse) {
        Write-Host "Installing dependencies from local wheelhouse: $Wheelhouse"
        & $PythonExe -m pip install --no-index --find-links $Wheelhouse -r requirements.txt
        return ($LASTEXITCODE -eq 0)
    }

    Write-Host "Installing dependencies from PyPI. If this machine has no internet/DNS, create a wheelhouse first."
    & $PythonExe -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { return $false }
    & $PythonExe -m pip install -r requirements.txt
    return ($LASTEXITCODE -eq 0)
}

if ($SkipInstall -ne "1") {
    $InstallOk = Install-DemoDependencies -PythonExe $VenvPython -Wheelhouse $WheelhouseDir
    if (-not $InstallOk) {
        Write-Host ""
        Write-Host "Dependency installation failed. The log usually means this Windows machine cannot resolve pypi.org."
        Write-Host "Fix options:"
        Write-Host "  1. Connect to internet / fix DNS / disable blocking proxy, then rerun this script."
        Write-Host "  2. On a machine with internet, run: powershell -ExecutionPolicy Bypass -File .\scripts\download_wheels.ps1"
        Write-Host "     Copy the generated 'wheelhouse' folder into this repo, then rerun this script."
        Write-Host "  3. If dependencies are already installed globally, delete .venv and rerun with:"
        Write-Host "     `$env:USE_SYSTEM_SITE_PACKAGES='1'; powershell -ExecutionPolicy Bypass -File .\scripts\run_demo.ps1"
        Write-Error "Cannot continue because required Python packages are missing."
    }
}

if (-not (Test-DemoDependencies -PythonExe $VenvPython)) {
    Write-Error "Required Python packages are still missing. Install from PyPI or provide .\wheelhouse first."
}

$CleanedDir = $(if ($env:CLEANED_DIR) { $env:CLEANED_DIR } else { "outputs/vong3_2_cleaned" })
$FiguresDir = $(if ($env:FIGURES_DIR) { $env:FIGURES_DIR } else { "outputs/vong3_2_figures" })

New-Item -ItemType Directory -Force -Path $CleanedDir, $FiguresDir | Out-Null

if ($ForceRebuild -eq "1") {
    Remove-Item -Recurse -Force $CleanedDir, $FiguresDir -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $CleanedDir, $FiguresDir | Out-Null
}

$DemoLightPath = Join-Path $CleanedDir "Customer_360_Demo_Light.csv"
if (-not (Test-Path $DemoLightPath)) {
    if (-not (Test-Path $DataDir)) {
        Write-Error "Missing data folder: $DataDir. Usage: powershell -ExecutionPolicy Bypass -File scripts/run_demo.ps1 -DataDir C:\path\to\Processed_Data"
    }
    if (-not (Test-Path $NotebookPath)) {
        Write-Error "Missing exact Colab notebook source: $NotebookPath"
    }

    & $VenvPython src/run_colab_exact.py `
        --notebook $NotebookPath `
        --raw-dir $DataDir `
        --cleaned-dir $CleanedDir `
        --figures-dir $FiguresDir
}

$env:COLAB_CLEANED_DIR = $CleanedDir
$env:COLAB_FIGURES_DIR = $FiguresDir
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"

Write-Host "Starting Streamlit demo at http://localhost:$Port"
if (Test-Path $VenvStreamlit) {
    & $VenvStreamlit run streamlit_app.py --server.port $Port --server.headless true --browser.gatherUsageStats false
} else {
    & $VenvPython -m streamlit run streamlit_app.py --server.port $Port --server.headless true --browser.gatherUsageStats false
}
