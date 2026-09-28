#!/usr/bin/env bash
# MMS-LLaMA save/resume smoke: single-batch forward/backward, checkpoint save,
# then a resume run that must continue past the restored update count.
#
# Usage:
#   MANIFEST_DIR=<manifest> NGPUS=4 GPU_IDS=0,1,2,3 scripts/smoke_local.sh
#
# Optional: CONFIG_NAME, MAX_TOKENS, OUT_PATH, PHASE1_UPDATES, PHASE2_UPDATES.
#
# Two things this script guards against, both learned the hard way:
#   * a previous run must have fully released its GPUs before the next starts,
#     otherwise the resume OOMs while loading the checkpoint;
#   * a rank that dies leaves the others blocked in a collective, so a stale
#     process tree can hold ~30 GB per GPU indefinitely -- check and clean.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT_PATH=${OUT_PATH:-${ROOT}/exp/mms_smoke}
PHASE1_UPDATES=${PHASE1_UPDATES:-2}
PHASE2_UPDATES=${PHASE2_UPDATES:-4}
export MAX_TOKENS=${MAX_TOKENS:-1500}
export OUT_PATH

wait_for_gpus() {
  local free_after=${1:-3000}
  for _ in $(seq 1 60); do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
    if [[ ${used} -lt ${free_after} ]]; then
      return 0
    fi
    echo "waiting for GPUs to drain (used=${used} MiB)"
    sleep 15
  done
  echo "GPUs did not drain; a stale process tree may be holding memory:" >&2
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >&2
  return 1
}

echo "=== phase 1: fresh run, ${PHASE1_UPDATES} updates ==="
rm -rf "${OUT_PATH}"
wait_for_gpus
MAX_UPDATE="${PHASE1_UPDATES}" bash "${ROOT}/scripts/train_local.sh"
echo "=== phase 1 done ==="
ls -la "${OUT_PATH}/checkpoints/"

echo "=== phase 2: resume, ${PHASE2_UPDATES} updates ==="
wait_for_gpus
MAX_UPDATE="${PHASE2_UPDATES}" RESTORE="${OUT_PATH}/checkpoints/checkpoint_last.pt" \
  bash "${ROOT}/scripts/train_local.sh"
echo "=== phase 2 done ==="
ls -la "${OUT_PATH}/checkpoints/"
echo "=== SMOKE_OK ==="
