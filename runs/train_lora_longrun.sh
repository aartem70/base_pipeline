#!/usr/bin/env bash
# Launch a single LoRA long-run training.
#
# Usage:
#   bash train_lora_longrun.sh <tag> <max_steps> <teacher_gpu> <student_gpu> [--prompts_file <jsonl>]
#
# Examples:
#   # E1: random ClimbMix, 2000 steps, GPUs 0,1
#   bash train_lora_longrun.sh e1_random 2000 0 1
#
#   # E2-forum: forum JSONL, 1000 steps, GPUs 2,3
#   bash train_lora_longrun.sh e2_forum 1000 2 3 --prompts_file \
#       /root/base_pipeline/caches/climbmix_per_cat/cat_forum.jsonl

set -euo pipefail

TAG="${1:?tag required (e.g. e1_random, e2_forum)}"
MAX_STEPS="${2:?max_steps required}"
TGPU="${3:?teacher_gpu required}"
SGPU="${4:?student_gpu required}"
shift 4
EXTRA=("$@")  # e.g. --prompts_file /path/to.jsonl

# Fixed config: LoRA r=8 attn-only, LR=5e-7 (Day 7 Exp 7.4 safe recipe).
LR=5e-7
BATCH=4
PPS=60
SEQ=1024
SAVE_EVERY=200

OUT_DIR="/root/checkpoints/exp_longlora_${TAG}_lr${LR}_steps${MAX_STEPS}"
LOG_FILE="/root/base_pipeline/logs/train_longlora_${TAG}.log"
mkdir -p "$(dirname "$LOG_FILE")"

source /root/base_pipeline/.venv/bin/activate
export HF_HOME=/root/.hf_home

echo "=== launching LoRA long run: $TAG ==="
echo "  max_steps:  $MAX_STEPS  (save_every=$SAVE_EVERY)"
echo "  teacher_gpu=$TGPU  student_gpu=$SGPU"
echo "  out: $OUT_DIR"
echo "  extra args: ${EXTRA[*]:-<none>}"

python /root/base_pipeline/train.py \
    --student kharchevnykov/distil \
    --teacher Qwen/Qwen3.5-35B-A3B \
    --teacher_gpu "$TGPU" --student_gpu "$SGPU" \
    --lr "$LR" --warmup_steps 20 --weight_decay 0.0 \
    --max_seq_len "$SEQ" --batch_size "$BATCH" --max_steps "$MAX_STEPS" \
    --prompts_per_step "$PPS" --resample_every 1 --seed 42 \
    --lora --lora_rank 8 --lora_alpha 16 \
    --lora_target "q_proj,k_proj,v_proj,o_proj" \
    --output_dir "$OUT_DIR" \
    --save_every "$SAVE_EVERY" --plot_every 50 --no_wandb \
    "${EXTRA[@]}" \
    2>&1 | tee "$LOG_FILE"
