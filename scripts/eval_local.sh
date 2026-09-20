#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SRC_PTH=${ROOT}/src
GPU_ID=${GPU_ID:-1}
LLM_PATH=${LLM_PATH:-${ROOT}/pretrained_models/llama-3.2-3b}
WHISPER_PATH=${WHISPER_PATH:-${ROOT}/pretrained_models/hf/openai-whisper-medium.en}
QFORMER_CONFIG_PATH=${QFORMER_CONFIG_PATH:-${ROOT}/pretrained_models/hf/bert-large-uncased}
SR_PREDICTOR_PATH=${SR_PREDICTOR_PATH:-${ROOT}/pretrained_models/sr_predictor/checkpoint.pt}
AVHUBERT_PATH=${AVHUBERT_PATH:-${ROOT}/pretrained_models/avhubert/large_vox_iter5.pt}
PYTHON_BIN=${PYTHON_BIN:-/root/miniforge3/envs/mms-llama-repro/bin/python}
MANIFEST_DIR=${MANIFEST_DIR:-${ROOT}/manifest/433h.local}
MODEL_PATH=${MODEL_PATH:-${ROOT}/pretrained_models/mms_llama/1759h/checkpoint_best.pt}
NOISE_WAV=${NOISE_WAV:-${ROOT}/noise/babble_noise.wav}
NOISE_PROB=${NOISE_PROB:-0}
NOISE_SNR=${NOISE_SNR:-0}
OUT_PATH=${OUT_PATH:-${ROOT}/results/$(basename "$(dirname "${MODEL_PATH}")")_noise${NOISE_PROB}_snr${NOISE_SNR}}

[[ -x ${PYTHON_BIN} ]] || { echo "Python executable not found: ${PYTHON_BIN}" >&2; exit 1; }

export PYTHONPATH=${ROOT}/fairseq${PYTHONPATH:+:${PYTHONPATH}}
CUDA_VISIBLE_DEVICES=${GPU_ID} "${PYTHON_BIN}" -B "${SRC_PTH}/eval.py" \
  --config-dir "${SRC_PTH}/conf" \
  --config-name s2s_decode \
  dataset.gen_subset=test \
  common.user_dir="${SRC_PTH}" \
  generation.beam=5 \
  generation.temperature=0.3 \
  override.llm_path="${LLM_PATH}" \
  override.w2v_path="${AVHUBERT_PATH}" \
  override.whisper_path="${WHISPER_PATH}" \
  override.qformer_config_path="${QFORMER_CONFIG_PATH}" \
  override.sr_predictor_path="${SR_PREDICTOR_PATH}" \
  "override.modalities=['video','audio']" \
  common_eval.path="${MODEL_PATH}" \
  common_eval.results_path="${OUT_PATH}" \
  override.label_dir="${MANIFEST_DIR}" \
  override.data="${MANIFEST_DIR}" \
  override.noise_wav="${NOISE_WAV}" \
  override.noise_prob="${NOISE_PROB}" \
  override.noise_snr="${NOISE_SNR}"
