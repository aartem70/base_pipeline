#!/bin/bash
# Launch 8-GPU parallel labeling on ClimbMix.
# Each GPU labels 1/8 of the sample.
#
# Usage:
#   bash launch_label_8gpu.sh <name> <n_total>
# e.g.:
#   bash launch_label_8gpu.sh pilot_2k 2000
#   bash launch_label_8gpu.sh bulk_200k 200000

set -e
NAME="${1:-pilot_2k}"
N_TOTAL="${2:-2000}"
BATCH="${3:-64}"
SHARDS="${4:-0-11}"

OUT_DIR="/root/base_pipeline/caches/climbmix_inspect"
LOG_DIR="/root/base_pipeline/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

source /root/base_pipeline/.venv/bin/activate
export HF_HOME=/root/.hf_home

echo "=== launching 8-GPU labeling: $NAME  N=$N_TOTAL  batch=$BATCH ==="
pids=()
for g in 0 1 2 3 4 5 6 7; do
    python /root/base_pipeline/runs/label_climbmix.py \
        --shards "$SHARDS" --n "$N_TOTAL" --batch "$BATCH" \
        --gpu "$g" --num-shards 8 --shard-id "$g" \
        --out "${OUT_DIR}/${NAME}_g${g}.parquet" \
        > "${LOG_DIR}/label_${NAME}_g${g}.log" 2>&1 &
    pids+=($!)
    echo "  gpu $g: PID=${pids[-1]}"
done

echo "waiting for all 8 workers..."
for pid in "${pids[@]}"; do
    wait "$pid" || echo "  worker PID=$pid failed with exit $?"
done

echo "=== merging per-GPU parquet files ==="
python - <<PY
import glob, pandas as pd
files = sorted(glob.glob("${OUT_DIR}/${NAME}_g*.parquet"))
print(f"merging {len(files)} files")
dfs = [pd.read_parquet(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
df.to_parquet("${OUT_DIR}/${NAME}.parquet", index=False)
print(f"merged {len(df)} rows -> ${OUT_DIR}/${NAME}.parquet")
print("\n=== global label distribution ===")
dist = df["category"].value_counts()
for name, c in dist.items():
    print(f"  {name:15s}  {c:>7d}  ({100*c/len(df):5.1f}%)")
print(f"\nconf mean={df['confidence'].mean():.3f} median={df['confidence'].median():.3f}")
print(f"margin mean={df['margin'].mean():.3f} median={df['margin'].median():.3f}")
PY
echo "=== done ==="
