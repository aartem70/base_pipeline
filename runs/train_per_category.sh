#!/usr/bin/env bash
# Train distil on a single category's JSONL.
#
# Usage:
#   bash train_per_category.sh <category> [MAX_STEPS] [LR]
# Example:
#   bash train_per_category.sh academic 200 5e-7

set -euo pipefail

CATEGORY="${1:?category name required (academic|news|forum|encyclopedic|product_desc)}"
MAX_STEPS="${2:-200}"
LR="${3:-5e-7}"
BATCH="${4:-4}"
PROMPTS_PER_STEP="${5:-60}"
TEACHER_GPU="${6:-0}"
STUDENT_GPU="${7:-1}"

PROMPTS_FILE="/root/base_pipeline/caches/climbmix_per_cat/cat_${CATEGORY}.jsonl"
# fall back to relaxed-length dir if strict version missing
if [ ! -f "$PROMPTS_FILE" ]; then
    PROMPTS_FILE="/root/base_pipeline/caches/climbmix_per_cat_1024/cat_${CATEGORY}.jsonl"
fi
if [ ! -f "$PROMPTS_FILE" ]; then
    echo "ERROR: No JSONL for category '$CATEGORY'"
    echo "  tried /root/base_pipeline/caches/climbmix_per_cat/cat_${CATEGORY}.jsonl"
    echo "  tried /root/base_pipeline/caches/climbmix_per_cat_1024/cat_${CATEGORY}.jsonl"
    exit 1
fi

OUT_DIR="/root/checkpoints/exp_percat_${CATEGORY}_lr${LR}_steps${MAX_STEPS}"
LOG_FILE="/root/base_pipeline/logs/train_percat_${CATEGORY}_lr${LR}_steps${MAX_STEPS}.log"

source /root/base_pipeline/.venv/bin/activate
export HF_HOME=/root/.hf_home

echo "=== training distil on category: $CATEGORY ==="
echo "  prompts: $PROMPTS_FILE ($(wc -l < $PROMPTS_FILE) lines)"
echo "  output:  $OUT_DIR"
echo "  log:     $LOG_FILE"
echo "  LR=$LR  max_steps=$MAX_STEPS  batch=$BATCH  prompts/step=$PROMPTS_PER_STEP"

python /root/base_pipeline/train.py \
    --student kharchevnykov/distil \
    --teacher Qwen/Qwen3.5-35B-A3B \
    --teacher_gpu "$TEACHER_GPU" --student_gpu "$STUDENT_GPU" \
    --prompts_file "$PROMPTS_FILE" \
    --lr "$LR" --warmup_steps 10 --weight_decay 0.0 \
    --max_seq_len 1024 --batch_size "$BATCH" --max_steps "$MAX_STEPS" \
    --prompts_per_step "$PROMPTS_PER_STEP" --resample_every 1 --seed 42 \
    --output_dir "$OUT_DIR" \
    --save_every 50 --plot_every 20 --no_wandb \
    2>&1 | tee "$LOG_FILE"
