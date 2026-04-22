#!/usr/bin/env bash
# Evaluate a checkpoint on the seed-42 teacher cache.
# Usage: eval_per_cat.sh <checkpoint_dir> <gpu_id> [out_json]

set -euo pipefail

CKPT="${1:?checkpoint dir required}"
GPU="${2:?gpu id required}"
OUT="${3:-/root/base_pipeline/logs/pp_$(echo $CKPT | tr / _).json}"

CACHE=/root/base_pipeline/caches/teacher_cache_60.pt

source /root/base_pipeline/.venv/bin/activate
export HF_HOME=/root/.hf_home

echo "=== eval: $CKPT on GPU $GPU ==="
echo "  cache: $CACHE"
echo "  out:   $OUT"

CUDA_VISIBLE_DEVICES="$GPU" python /root/base_pipeline/runs/eval_bootstrap.py \
    --model-repo "$CKPT" \
    --cache "$CACHE" \
    --out "$OUT" \
    --device cuda:0
