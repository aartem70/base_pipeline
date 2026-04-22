#!/usr/bin/env bash
# Long-running seq-level distillation pipeline.
#
# Stages:
#   1. Eval Exp 8.1b (LR=1e-8, 50 steps) - already trained, just score.
#   2. LR sweep (each 300 steps): seq-CE at LR = {5e-8, 3e-8, 1e-7, 5e-8 x 1000 steps if winner}.
#   3. Norm/bias-only fine-tune at LR=1e-5, 500 steps.
#   4. Evaluate every final/ + paired bootstrap vs kharchevnykov/distil baseline.
#   5. Print summary table.

set -u
cd /home/shadeform/base_pipeline
source .venv/bin/activate
export HF_HOME=/ephemeral/.hf_home
export PYTHONUNBUFFERED=1
mkdir -p /ephemeral/logs

CACHE_EVAL=/ephemeral/teacher_cache_60.pt
CACHE_TRAIN=/ephemeral/cont_cache_600_seqs.pt
BASELINE_JSON=/ephemeral/logs/pp_distil_baseline.json
SUMMARY=/ephemeral/logs/pipeline_summary.txt
echo "=== Pipeline start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee "$SUMMARY"

eval_ckpt () {
    local ckpt=$1
    local tag=$2
    local out=/ephemeral/logs/pp_${tag}.json
    echo "" | tee -a "$SUMMARY"
    echo "--- eval $tag ($ckpt) ---" | tee -a "$SUMMARY"
    CUDA_VISIBLE_DEVICES=0 python runs/eval_bootstrap.py \
        --model-repo "$ckpt" \
        --cache "$CACHE_EVAL" \
        --out "$out" 2>&1 | tail -10 | tee -a "$SUMMARY"
    echo "   paired bootstrap vs baseline:" | tee -a "$SUMMARY"
    python runs/compare_pp.py --baseline "$BASELINE_JSON" --candidate "$out" 2>&1 | tee -a "$SUMMARY"
}

run_train () {
    local name=$1
    local lr=$2
    local steps=$3
    local warmup=$4
    local extra_args=${5:-}
    local outdir=/ephemeral/runs/exp_${name}
    echo "" | tee -a "$SUMMARY"
    echo "=== training ${name} (LR=$lr, steps=$steps) ===" | tee -a "$SUMMARY"
    rm -rf "$outdir"
    CUDA_VISIBLE_DEVICES=0 python runs/train_seqkd.py \
        --student kharchevnykov/distil \
        --cache "$CACHE_TRAIN" \
        --output_dir "$outdir" \
        --max_steps "$steps" \
        --lr "$lr" \
        --warmup_steps "$warmup" \
        --batch_size 4 \
        --save_every $((steps / 2)) \
        --seed 42 \
        $extra_args \
        > /ephemeral/logs/train_${name}.log 2>&1
    # summarize train curve
    python3 <<EOF | tee -a "$SUMMARY"
import json
rows = [json.loads(l) for l in open('$outdir/train_metrics.jsonl')]
losses = [r['loss'] for r in rows]
n = len(losses)
print(f'   train loss: steps={n} first5={sum(losses[:5])/5:.3f} last20={sum(losses[-20:])/min(20,n):.3f} min={min(losses):.3f} max={max(losses):.3f}')
EOF
}

# ---------------- stage 1: eval 8.1b (already trained) ----------------
eval_ckpt /ephemeral/runs/exp_8.1b_seqkd_lr1e8/final exp_8.1b_final

# ---------------- stage 2: LR sweep ----------------
run_train 8.1c_lr5e8 5e-8 300 10
eval_ckpt /ephemeral/runs/exp_8.1c_lr5e8/final exp_8.1c_final

run_train 8.1d_lr3e8 3e-8 300 10
eval_ckpt /ephemeral/runs/exp_8.1d_lr3e8/final exp_8.1d_final

run_train 8.1e_lr1e7 1e-7 300 10
eval_ckpt /ephemeral/runs/exp_8.1e_lr1e7/final exp_8.1e_final

# ---------------- stage 3: long horizon of best seq-CE ----------------
# Find winner among 8.1{b,c,d,e} based on mean paired delta. Simple heuristic:
# if any ran with last20 avg train loss within [0.2, 0.5] AND eval improved, extend.
# For now: always run an 8.1f with LR=5e-8 at 1000 steps (long horizon).
run_train 8.1f_lr5e8_1000 5e-8 1000 20
eval_ckpt /ephemeral/runs/exp_8.1f_lr5e8_1000/final exp_8.1f_final

# ---------------- stage 4: different mechanism (norm/bias-only FT) ----------------
# Requires a code variant of train_seqkd.py with freeze_except_norm_bias.
# We'll inject via a small wrapper script instead of modifying train_seqkd.py.
# See train_seqkd_normbias.py (sibling script).
run_train_normbias () {
    local name=$1
    local lr=$2
    local steps=$3
    local outdir=/ephemeral/runs/exp_${name}
    echo "" | tee -a "$SUMMARY"
    echo "=== training ${name} norm/bias-only (LR=$lr, steps=$steps) ===" | tee -a "$SUMMARY"
    rm -rf "$outdir"
    CUDA_VISIBLE_DEVICES=0 python runs/train_seqkd_normbias.py \
        --student kharchevnykov/distil \
        --cache "$CACHE_TRAIN" \
        --output_dir "$outdir" \
        --max_steps "$steps" \
        --lr "$lr" \
        --warmup_steps 20 \
        --batch_size 4 \
        --save_every $((steps / 2)) \
        --seed 42 \
        > /ephemeral/logs/train_${name}.log 2>&1
    python3 <<EOF | tee -a "$SUMMARY"
import json
rows = [json.loads(l) for l in open('$outdir/train_metrics.jsonl')]
losses = [r['loss'] for r in rows]
n = len(losses)
print(f'   train loss: steps={n} first5={sum(losses[:5])/5:.3f} last20={sum(losses[-20:])/min(20,n):.3f} min={min(losses):.3f} max={max(losses):.3f}')
EOF
}

run_train_normbias 8.2_normbias_lr1e5 1e-5 500
eval_ckpt /ephemeral/runs/exp_8.2_normbias_lr1e5/final exp_8.2_final

# ---------------- stage 5: print summary ----------------
echo "" | tee -a "$SUMMARY"
echo "=== ALL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "Summary of paired-bootstrap deltas (vs kharchevnykov/distil, negative = better):" | tee -a "$SUMMARY"
for tag in exp_8.1b_final exp_8.1c_final exp_8.1d_final exp_8.1e_final exp_8.1f_final exp_8.2_final; do
    f=/ephemeral/logs/pp_${tag}.json
    [ -f "$f" ] || continue
    python3 <<EOF | tee -a "$SUMMARY"
import json, random, statistics
a = json.load(open('${BASELINE_JSON}'))
b = json.load(open('${f}'))
pa, pb = a['per_prompt_kl'], b['per_prompt_kl']
n = len(pa)
deltas = [bi - ai for ai, bi in zip(pa, pb)]
mean_b = sum(pb)/n
md = sum(deltas)/n
rng = random.Random(42)
boots = [sum([deltas[rng.randrange(n)] for _ in range(n)])/n for _ in range(10000)]
boots.sort()
print(f'  ${tag}: mean_kl={mean_b:.6f} delta={md:+.6f} CI=[{boots[250]:+.6f},{boots[9750]:+.6f}]')
EOF
done
