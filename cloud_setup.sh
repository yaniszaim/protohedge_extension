#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${PACKAGE_ROOT}/.venv-cloud"

python3 -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r "${PACKAGE_ROOT}/requirements-cloud.txt"

cd "${PACKAGE_ROOT}/.."
python - <<'PY'
import torch
import deephedging
from deephedging.base_torch import resolve_torch_device

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
device = resolve_torch_device("cuda")
print("selected device:", device)
print("accelerator:", torch.cuda.get_device_name(device))
PY

echo "Cloud environment is ready: ${VENV_PATH}"
