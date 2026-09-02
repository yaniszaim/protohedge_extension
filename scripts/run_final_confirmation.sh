#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTORCH_PYTHON="${PYTORCH_PYTHON:-/opt/anaconda3/envs/dh/bin/python}"
SYNTHETIC_PYTHON="${SYNTHETIC_PYTHON:-$REPO_ROOT/.venv-original-synthetic/bin/python}"
DEVICE="${DEVICE:-auto}"
RUN_BLACK_SCHOLES=0
STATUS_ONLY=0

usage() {
  cat <<'EOF'
Usage: scripts/run_final_confirmation.sh [options]

Options:
  --device DEVICE          PyTorch device: auto, cpu, cuda, or mps.
  --with-black-scholes     Also run the two-model notebook-faithful BS check.
  --status                 Show SPY completion status without training.
  -h, --help               Show this help.

Environment overrides:
  PYTORCH_PYTHON            Python for the real-data SPY check.
  SYNTHETIC_PYTHON          Python for the TensorFlow synthetic check.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --with-black-scholes)
      RUN_BLACK_SCHOLES=1
      shift
      ;;
    --status)
      STATUS_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$REPO_ROOT"
if [[ ! -x "$PYTORCH_PYTHON" ]]; then
  echo "PyTorch Python is not executable: $PYTORCH_PYTHON" >&2
  exit 1
fi

if [[ "$STATUS_ONLY" -eq 1 ]]; then
  "$PYTORCH_PYTHON" scripts/run_spy_softclip_confirmation.py \
    --device "$DEVICE" \
    --status-only
  exit 0
fi

echo "[1/3] Verifying the exact SoftClip implementation"
PYTHONPATH="$REPO_ROOT/.." "$PYTORCH_PYTHON" -m unittest \
  tests.test_softclip_torch \
  tests.test_spy_softclip_confirmation

echo "[2/3] Running two matched ProtoHedge fits on European SPY"
"$PYTORCH_PYTHON" -u scripts/run_spy_softclip_confirmation.py \
  --device "$DEVICE"

if [[ "$RUN_BLACK_SCHOLES" -eq 1 ]]; then
  if [[ ! -x "$SYNTHETIC_PYTHON" ]]; then
    echo "Synthetic environment is missing: $SYNTHETIC_PYTHON" >&2
    echo "Run: bash scripts/setup_original_synthetic_env.sh" >&2
    exit 1
  fi
  echo "[3/3] Running the notebook-faithful Black-Scholes DH/PH pair"
  "$SYNTHETIC_PYTHON" -u scripts/reproduce_original_synthetic.py \
    --profile notebook-black-scholes
else
  echo "[3/3] Black-Scholes check skipped (add --with-black-scholes to run it)"
fi

echo "Final confirmation workflow complete."
