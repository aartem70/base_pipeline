"""Spot-check LLM labels: show N samples per category + flag low-confidence ones."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path("/root/base_pipeline/caches/climbmix_raw")
IDX_DIR = Path("/root/base_pipeline/caches/climbmix_inspect")


def read_doc(shard: int, idx: int, fp_cache: dict, off_cache: dict, max_chars: int = 400):
    if shard not in fp_cache:
        fp_cache[shard] = (RAW_DIR / f"part_{shard}.jsonl").open("rb")
        off_cache[shard] = np.load(IDX_DIR / f"part_{shard}.idx.npy")
    fp = fp_cache[shard]
    off = int(off_cache[shard][idx])
    fp.seek(off)
    line = fp.readline().decode("utf-8", errors="replace")
    try:
        return json.loads(line).get("text", "")[:max_chars]
    except Exception:
        return "<PARSE FAIL>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--per-cat", type=int, default=5)
    ap.add_argument("--low-conf", action="store_true", help="also show lowest-confidence labels")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    print(f"loaded {len(df)} labeled docs")
    fp_cache, off_cache = {}, {}

    for cat in sorted(df["category"].unique()):
        sub = df[df["category"] == cat]
        sample = sub.sample(min(args.per_cat, len(sub)), random_state=42)
        print(f"\n=== {cat} ({len(sub)} docs, conf mean={sub['confidence'].mean():.2f}) ===")
        for _, row in sample.iterrows():
            text = read_doc(int(row["shard"]), int(row["idx"]), fp_cache, off_cache, 300)
            text = text.replace("\n", " | ")
            print(f"  [s{row['shard']} #{row['idx']}] conf={row['confidence']:.2f} margin={row['margin']:.2f} top2={row['top2']}")
            print(f"    {text[:280]}")

    if args.low_conf:
        print("\n\n=== 10 LOWEST-CONFIDENCE LABELS ===")
        low = df.nsmallest(10, "confidence")
        for _, row in low.iterrows():
            text = read_doc(int(row["shard"]), int(row["idx"]), fp_cache, off_cache, 300)
            text = text.replace("\n", " | ")
            print(f"  [s{row['shard']} #{row['idx']}] {row['category']:13s} conf={row['confidence']:.2f} margin={row['margin']:.2f} top2={row['top2']}")
            print(f"    {text[:280]}")

    for fp in fp_cache.values():
        fp.close()


if __name__ == "__main__":
    main()
