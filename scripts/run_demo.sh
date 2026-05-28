#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
PORT="${PORT:-8501}"
DATA_DIR="${1:-${DATA_DIR:-Processed_Data}}"
NOTEBOOK_PATH="${NOTEBOOK_PATH:-notebooks/colab_exact_business.ipynb}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p outputs/colab_exact_cleaned outputs/colab_exact_figures

if [ "$FORCE_REBUILD" = "1" ]; then
  rm -rf outputs/colab_exact_cleaned outputs/colab_exact_figures
  mkdir -p outputs/colab_exact_cleaned outputs/colab_exact_figures
fi

if [ ! -f "outputs/colab_exact_cleaned/Customer_360_Demo_Light.csv" ]; then
  if [ ! -d "$DATA_DIR" ]; then
    echo "Missing data folder: $DATA_DIR" >&2
    echo "Usage: bash scripts/run_demo.sh /path/to/Processed_Data" >&2
    echo "Or place the real contest data in ./Processed_Data" >&2
    exit 1
  fi
  if [ ! -f "$NOTEBOOK_PATH" ]; then
    echo "Missing exact Colab notebook source: $NOTEBOOK_PATH" >&2
    exit 1
  fi
  python src/run_colab_exact.py \
    --notebook "$NOTEBOOK_PATH" \
    --raw-dir "$DATA_DIR" \
    --cleaned-dir outputs/colab_exact_cleaned \
    --figures-dir outputs/colab_exact_figures
fi

export COLAB_CLEANED_DIR="outputs/colab_exact_cleaned"
export COLAB_FIGURES_DIR="outputs/colab_exact_figures"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

echo "Starting Streamlit demo at http://localhost:${PORT}"
streamlit run streamlit_app.py --server.port "$PORT" --server.headless true --browser.gatherUsageStats false
