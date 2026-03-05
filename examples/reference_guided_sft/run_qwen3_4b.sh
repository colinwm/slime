#!/bin/bash
#
# Reference-Guided On-Policy SFT — example launch script for Qwen3-4B.
#
# This demonstrates the hybrid SFT/RL training mode:
#   1. The model sees a reference solution in-context during generation.
#   2. Tools are executed (multi-turn), producing an on-policy trajectory.
#   3. The reference is stripped; training uses SFT loss on the clean prompt +
#      the model's own response.
#
# Prerequisites:
#   - A dataset (.jsonl or .parquet) with columns:
#       "prompt"     : the math problem (string)
#       "reference"  : a reference solution trace (string)
#       "label"      : (optional) ground-truth answer for eval
#   - Model checkpoint & torch_dist conversion (see retool README)
#   - Megatron-LM on PYTHONPATH
#
# Adapt paths below to your environment.
# ============================================================================

# --- cleanup from previous runs ---
pkill -9 sglang 2>/dev/null
sleep 3
ray stop --force 2>/dev/null
pkill -9 ray 2>/dev/null
pkill -9 python 2>/dev/null
sleep 3

set -ex

export PYTHONBUFFERED=16

# --- detect NVLink ---
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$( [ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0 )
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

# --- source model architecture args ---
source "${SLIME_DIR}/scripts/models/qwen3-4B.sh"

# ============================================================================
# CHANGE THESE PATHS to match your setup
# ============================================================================
HF_CHECKPOINT="/root/Qwen/Qwen3-4B-Instruct-2507"
REF_LOAD="/root/Qwen/Qwen3-4B-Instruct-2507_torch_dist"
SAVE_DIR="/root/Qwen/Qwen3-4B-Instruct-2507_ref_guided_sft/"
PROMPT_DATA="/root/data/reference_guided_sft_data.jsonl"

# ============================================================================
# Arguments
# ============================================================================

CKPT_ARGS=(
    --hf-checkpoint "${HF_CHECKPOINT}"
    --ref-load "${REF_LOAD}"
    --save "${SAVE_DIR}"
    --save-interval 100
    --rotary-base 5000000
)

# This is NOT --debug-train-only: we need the SGLang inference engine running
# because we do on-policy generation.  The key difference from normal RL is
# that we use sft_loss and disable advantage computation.
ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_DATA}"
    --input-key prompt
    --label-key reference
    --apply-chat-template
    --rollout-shuffle
    --num-rollout 3000
    --rollout-batch-size 32
    --n-samples-per-prompt 1
    --rollout-max-response-len 8192
    --rollout-temperature 0.7

    --global-batch-size 32
)

SFT_LOSS_ARGS=(
    --loss-type sft_loss
    --calculate-per-token-loss
    --disable-compute-advantages-and-returns
)

PERF_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1

    --use-dynamic-batch-size
    --max-tokens-per-gpu 9216
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-5
    --lr-decay-style cosine
    --min-lr 1e-6
    --lr-warmup-fraction 0.1
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project slime-ref-guided-sft
    --wandb-group qwen3-4B-ref-guided
    --wandb-key "${WANDB_KEY}"
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

# This is the key: we use the standard sglang_rollout (which starts inference
# engines), but plug in our custom generate function that injects the reference
# and then reconstructs the training sequence without it.
CUSTOM_ARGS=(
    --custom-generate-function-path examples.reference_guided_sft.generate.generate
)

# ============================================================================
# Launch
# ============================================================================

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 4 \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:${SLIME_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 4 \
    --colocate \
    ${MODEL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${ROLLOUT_ARGS[@]} \
    ${SFT_LOSS_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${WANDB_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${MISC_ARGS[@]} \
    ${CUSTOM_ARGS[@]}
