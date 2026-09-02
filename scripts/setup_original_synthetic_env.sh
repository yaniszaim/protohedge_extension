#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${1:-$REPO_ROOT/.venv-original-synthetic}"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/dh/bin/python3.10}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python 3.10 was not found at $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN to a Python 3.10 executable and rerun." >&2
  exit 1
fi

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$ENV_DIR"
fi

"$ENV_DIR/bin/python" -m pip install --upgrade pip wheel
"$ENV_DIR/bin/python" -m pip install -r "$REPO_ROOT/requirements-original-synthetic.txt"
"$ENV_DIR/bin/python" -c \
  'import numpy, tensorflow as tf, tensorflow_probability as tfp; print("NumPy", numpy.__version__); print("TensorFlow", tf.__version__); print("TFP", tfp.__version__)'

echo "Original synthetic environment is ready: $ENV_DIR"
