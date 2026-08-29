#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DH_PYTHON="${DH_PYTHON:-/opt/anaconda3/envs/dh/bin/python}"
TF_PYTHON="${TF_PYTHON:-${REPO_ROOT}/.venv/original-synthetic/bin/python}"
DH_DEVICE="${DH_DEVICE:-auto}"
LOG_DIR="${REPO_ROOT}/paper/prewrite_compute_logs"

for executable in "${DH_PYTHON}" "${TF_PYTHON}"; do
  if [[ ! -x "${executable}" ]]; then
    echo "Required Python executable is missing: ${executable}" >&2
    exit 1
  fi
done

mkdir -p "${LOG_DIR}" /private/tmp/protohedge-mpl
export PYTHONPATH="$(dirname "${REPO_ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/protohedge-mpl}"

echo "[$(date -u +%FT%TZ)] Starting locked checkpoint sensitivity"
"${DH_PYTHON}" -u \
  "${REPO_ROOT}/scripts/run_checkpoint_selection_sensitivity.py" \
  --device "${DH_DEVICE}" \
  2>&1 | tee -a "${LOG_DIR}/checkpoint_selection_sensitivity.log"

echo "[$(date -u +%FT%TZ)] Starting original synthetic reproduction"
"${TF_PYTHON}" -u \
  "${REPO_ROOT}/scripts/reproduce_original_synthetic.py" \
  2>&1 | tee -a "${LOG_DIR}/original_synthetic_reproduction.log"

echo "[$(date -u +%FT%TZ)] All pre-writing compute stages completed"
