#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
PORT="${PORT:-8501}"
DATA_DIR="${1:-${DATA_DIR:-Processed_Data}}"
NOTEBOOK_PATH="${NOTEBOOK_PATH:-Vong_3_EAZII_2.ipynb}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
CLEANED_DIR="${CLEANED_DIR:-outputs/vong3_2_cleaned}"
FIGURES_DIR="${FIGURES_DIR:-outputs/vong3_2_figures}"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-wheelhouse}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
USE_SYSTEM_SITE_PACKAGES="${USE_SYSTEM_SITE_PACKAGES:-0}"

if [ ! -d "$VENV_DIR" ]; then
  if [ "$USE_SYSTEM_SITE_PACKAGES" = "1" ]; then
    "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
  else
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

test_dependencies() {
  python -c "import pandas,numpy,sklearn,matplotlib,seaborn,plotly,openpyxl,nbformat,nbconvert,xgboost,streamlit,shap; print('Python dependencies OK')"
}

install_dependencies() {
  if [ -d "$WHEELHOUSE_DIR" ]; then
    echo "Installing dependencies from local wheelhouse: $WHEELHOUSE_DIR"
    python -m pip install --no-index --find-links "$WHEELHOUSE_DIR" -r requirements.txt
  else
    echo "Installing dependencies from PyPI. If this machine has no internet/DNS, create a wheelhouse first."
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
  fi
}

if [ "$SKIP_INSTALL" != "1" ]; then
  if ! install_dependencies; then
    cat >&2 <<'EOF'

Dependency installation failed. The log usually means this machine cannot resolve pypi.org.
Fix options:
  1. Connect to internet / fix DNS / disable blocking proxy, then rerun this script.
  2. On a machine with internet, run: bash scripts/download_wheels.sh
     Copy the generated 'wheelhouse' folder into this repo, then rerun this script.
  3. If dependencies are already installed globally, delete .venv and rerun with:
     USE_SYSTEM_SITE_PACKAGES=1 bash scripts/run_demo.sh
EOF
    exit 1
  fi
fi

if ! test_dependencies; then
  echo "Required Python packages are still missing. Install from PyPI or provide ./wheelhouse first." >&2
  exit 1
fi

mkdir -p "$CLEANED_DIR" "$FIGURES_DIR"

if [ "$FORCE_REBUILD" = "1" ]; then
  rm -rf "$CLEANED_DIR" "$FIGURES_DIR"
  mkdir -p "$CLEANED_DIR" "$FIGURES_DIR"
fi

if [ ! -f "$CLEANED_DIR/Customer_360_Demo_Light.csv" ]; then
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
    --cleaned-dir "$CLEANED_DIR" \
    --figures-dir "$FIGURES_DIR"
fi

export COLAB_CLEANED_DIR="$CLEANED_DIR"
export COLAB_FIGURES_DIR="$FIGURES_DIR"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

echo "Starting Streamlit demo at http://localhost:${PORT}"
streamlit run streamlit_app.py --server.port "$PORT" --server.headless true --browser.gatherUsageStats false
