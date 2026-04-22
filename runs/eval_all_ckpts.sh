#!/usr/bin/env bash
# Evaluate every checkpoint under a run dir + paired-bootstrap vs baseline.
# Usage: eval_all_ckpts.sh <run_dir>
#    Produces /ephemeral/logs/pp_<exp>_<ckpt>.json and paired comparison prints.

set -euo pipefail

RUN_DIR="${1:?run dir required}"
NAME=$(basename "$RUN_DIR")
CACHE=/ephemeral/teacher_cache_60.pt
BASELINE=/ephemeral/logs/pp_distil_baseline.json

cd /home/shadeform/base_pipeline
source .venv/bin/activate
export HF_HOME=/ephemeral/.hf_home

for ckpt in "$RUN_DIR"/step_* "$RUN_DIR"/final "$RUN_DIR"/best_train_loss; do
    [ -d "$ckpt" ] || continue
    ck=$(basename "$ckpt")
    out=/ephemeral/logs/pp_${NAME}_${ck}.json
    echo "=== [$NAME/$ck] ==="
    CUDA_VISIBLE_DEVICES=0 python runs/eval_bootstrap.py \
        --model-repo "$ckpt" \
        --cache "$CACHE" \
        --out "$out"
    echo
    echo "--- paired bootstrap vs baseline ---"
    python runs/compare_pp.py --baseline "$BASELINE" --candidate "$out"
    echo
done
