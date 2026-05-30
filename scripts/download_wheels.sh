#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-wheelhouse}"

mkdir -p "$WHEELHOUSE_DIR"

echo "Downloading Python wheels into: $WHEELHOUSE_DIR"
echo "Use this on a machine with internet, then copy the whole wheelhouse folder to the offline machine."

"$PYTHON_BIN" -m pip download --only-binary=:all: --dest "$WHEELHOUSE_DIR" -r requirements.txt

echo "Wheelhouse is ready: $WHEELHOUSE_DIR"
