#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SRC_PTH=${ROOT}/src
LLM_PATH=${LLM_PATH:-meta-llama/Llama-3.2-3B}
WHISPER_PATH=${WHISPER_PATH:-${ROOT}/pretrained_models/hf/openai-whisper-medium.en}
QFORMER_CONFIG_PATH=${QFORMER_CONFIG_PATH:-${ROOT}/pretrained_models/hf/bert-large-uncased}
SR_PREDICTOR_PATH=${SR_PREDICTOR_PATH:-${ROOT}/pretrained_models/sr_predictor/checkpoint.pt}
MANIFEST_DIR=${MANIFEST_DIR:-${ROOT}/manifest/433h.local}
NOISE_WAV=${NOISE_WAV:-${ROOT}/noise/babble_noise.wav}
NGPUS=${NGPUS:-2}
GPU_IDS=${GPU_IDS:-0,1}
UPDATE_FREQ=${UPDATE_FREQ:-4}
MAX_TOKENS=${MAX_TOKENS:-1000}
OUT_PATH=${OUT_PATH:-${ROOT}/exp/mms-llama/433h_${NGPUS}gpu}

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=${ROOT}/fairseq${PYTHONPATH:+:${PYTHONPATH}}

CUDA_VISIBLE_DEVICES=${GPU_IDS} fairseq-hydra-train \
  --config-dir "${SRC_PTH}/conf" \
  --config-name mms-llama.yaml \
  task.data="${MANIFEST_DIR}" \
  task.label_dir="${MANIFEST_DIR}" \
  task.tokenizer_bpe_model=null \
  task.llm_path="${LLM_PATH}" \
  task.noise_prob=0.75 \
  task.noise_wav="${NOISE_WAV}" \
  hydra.run.dir="${OUT_PATH}" \
  common.user_dir="${SRC_PTH}" \
  common.seed=1 \
  common.empty_cache_freq=1000 \
  dataset.max_tokens="${MAX_TOKENS}" \
  model.w2v_path="${ROOT}/pretrained_models/avhubert/large_vox_iter5.pt" \
  model.whisper_path="${WHISPER_PATH}" \
  model.qformer_config_path="${QFORMER_CONFIG_PATH}" \
  model.sr_predictor_path="${SR_PREDICTOR_PATH}" \
  model.llm_path="${LLM_PATH}" \
  model.llama_embed_dim=3072 \
  model.queries_per_sec=3 \
  model.target_modules=q_proj.k_proj.v_proj.o_proj \
  model.modality_fuse=concat \
  model.lora_rank=16 \
  model.lora_alpha=32 \
  model.use_qformer=true \
  model.use_sr_predictor=true \
  "optimization.update_freq=[${UPDATE_FREQ}]" \
  'optimization.lr=[1e-4]' \
  optimization.max_update=30000 \
  lr_scheduler._name=cosine \
  lr_scheduler.warmup_updates=500 \
  distributed_training.distributed_world_size="${NGPUS}" \
  distributed_training.nprocs_per_node="${NGPUS}" \
  distributed_training.ddp_backend=legacy_ddp \
  distributed_training.find_unused_parameters=true
