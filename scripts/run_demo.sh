#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
PORT="${PORT:-8501}"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p outputs

if [ -d "demo_data/colab_exact_cleaned" ] && [ ! -f "outputs/colab_exact_cleaned/Customer_360_Demo_Light.csv" ]; then
  mkdir -p outputs/colab_exact_cleaned outputs/colab_exact_figures
  cp -R demo_data/colab_exact_cleaned/. outputs/colab_exact_cleaned/
  if [ -d "demo_data/colab_exact_figures" ]; then
    cp -R demo_data/colab_exact_figures/. outputs/colab_exact_figures/
  fi
fi

if [ ! -f "outputs/colab_exact_cleaned/Customer_360_Demo_Light.csv" ]; then
  if [ -d "Processed_Data" ] && [ -f "Another copy of Welcome To Colab" ]; then
    python src/run_colab_exact.py --raw-dir Processed_Data --cleaned-dir outputs/colab_exact_cleaned --figures-dir outputs/colab_exact_figures
  else
    echo "Missing demo data and no local Processed_Data/ + Colab notebook were found." >&2
    echo "Clone the full repo including demo_data/, or place contest data in Processed_Data/." >&2
    exit 1
  fi
fi

export COLAB_CLEANED_DIR="outputs/colab_exact_cleaned"
export COLAB_FIGURES_DIR="outputs/colab_exact_figures"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

echo "Starting Streamlit demo at http://localhost:${PORT}"
streamlit run streamlit_app.py --server.port "$PORT" --server.headless true --browser.gatherUsageStats false
