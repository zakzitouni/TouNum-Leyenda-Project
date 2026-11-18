#!/bin/bash
# Starts the Leyenda Project page on http://localhost:8000 (or the port given as argument), on Linux or macOS.
# The first run installs the Python packages into ~/.leyenda-venv.
set -e
cd "$(dirname "$0")"
VENV="$HOME/.leyenda-venv"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/python" -m pip install --disable-pip-version-check -r ../requirements.txt
fi
export TF_CPP_MIN_LOG_LEVEL=3
exec "$VENV/bin/python" server.py "${1:-8000}"
