param(
    [string]$DataDir = $env:DATA_DIR,
    [string]$PythonBin = $env:PYTHON_BIN,
    [string]$VenvDir = $env:VENV_DIR,
    [int]$Port = $(if ($env:PORT) { [int]$env:PORT } else { 8501 }),
    [string]$NotebookPath = $env:NOTEBOOK_PATH,
    [string]$ForceRebuild = $env:FORCE_REBUILD
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
if ([string]::IsNullOrWhiteSpace($NotebookPath)) {
    $NotebookPath = "Vòng_3_EAZII.ipynb"
}
if ([string]::IsNullOrWhiteSpace($ForceRebuild)) {
    $ForceRebuild = "0"
}

$VenvPython = Join-Path $VenvDir "Scripts/python.exe"
$VenvStreamlit = Join-Path $VenvDir "Scripts/streamlit.exe"

if (-not (Test-Path $VenvPython)) {
    & $PythonBin -m venv $VenvDir
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt

$CleanedDir = $(if ($env:CLEANED_DIR) { $env:CLEANED_DIR } else { "outputs/vong3_cleaned" })
$FiguresDir = $(if ($env:FIGURES_DIR) { $env:FIGURES_DIR } else { "outputs/vong3_figures" })

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
