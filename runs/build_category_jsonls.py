"""
Build per-category training JSONLs from bulk LLM labels.

For each category, filter labels by confidence, shuffle, and materialize a JSONL
file where each line is {"text": "<doc text>"}. Skips docs too short to be useful
as training prompts (< min_chars for the teacher context).

Usage:
  python build_category_jsonls.py --parquet bulk_200k.parquet \
     --conf-min 0.5 --margin-min 0.1 --min-chars 2560 \
     --out-dir /root/base_pipeline/caches/climbmix_per_cat
"""
import argparse, json, os
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path("/root/base_pipeline/caches/climbmix_raw")
IDX_DIR = Path("/root/base_pipeline/caches/climbmix_inspect")


def read_doc(shard: int, idx: int, fp_cache: dict, off_cache: dict) -> str:
    if shard not in fp_cache:
        fp_cache[shard] = (RAW_DIR / f"part_{shard}.jsonl").open("rb")
        off_cache[shard] = np.load(IDX_DIR / f"part_{shard}.idx.npy")
    fp = fp_cache[shard]
    off = int(off_cache[shard][idx])
    fp.seek(off)
    line = fp.readline().decode("utf-8", errors="replace")
    try:
        return json.loads(line).get("text", "")
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--conf-min", type=float, default=0.5)
    ap.add_argument("--margin-min", type=float, default=0.1)
    ap.add_argument("--min-chars", type=int, default=2560, help="skip docs shorter than this")
    ap.add_argument("--max-chars", type=int, default=10000)
    ap.add_argument("--out-dir", default="/root/base_pipeline/caches/climbmix_per_cat")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-per-cat", type=int, default=500,
                    help="only write JSONL for categories with >=N usable docs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    total = len(df)
    print(f"loaded {total} labeled docs from {args.parquet}")

    # confidence + margin filter
    kept = df[(df["confidence"] >= args.conf_min) & (df["margin"] >= args.margin_min)]
    print(f"after conf>={args.conf_min} margin>={args.margin_min}: {len(kept)} docs ({100*len(kept)/total:.1f}%)")

    rng = np.random.default_rng(args.seed)
    fp_cache, off_cache = {}, {}
    summary = []

    for cat in sorted(kept["category"].unique()):
        sub = kept[kept["category"] == cat].copy()
        # shuffle
        sub = sub.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

        out_path = out_dir / f"cat_{cat}.jsonl"
        written = 0
        skipped_short = 0
        skipped_err = 0
        with out_path.open("w", encoding="utf-8") as f:
            for _, row in sub.iterrows():
                text = read_doc(int(row["shard"]), int(row["idx"]), fp_cache, off_cache)
                if not text:
                    skipped_err += 1
                    continue
                if len(text) < args.min_chars:
                    skipped_short += 1
                    continue
                if len(text) > args.max_chars:
                    text = text[:args.max_chars]
                    ls = text.rfind(" ")
                    if ls > args.max_chars // 2:
                        text = text[:ls]
                f.write(json.dumps({"text": text, "shard": int(row["shard"]), "idx": int(row["idx"]),
                                     "category": cat, "confidence": float(row["confidence"])}) + "\n")
                written += 1

        status = "OK" if written >= args.min_per_cat else "TOO FEW"
        summary.append((cat, written, skipped_short, skipped_err, status))
        print(f"  {cat:15s}  wrote {written:>6d}  short-skip {skipped_short:>5d}  err {skipped_err:>4d}  [{status}]")

        if written < args.min_per_cat:
            # delete tiny file so downstream scripts don't accidentally pick it
            out_path.unlink(missing_ok=True)

    for fp in fp_cache.values():
        fp.close()

    # write summary
    summary_path = out_dir / "_summary.txt"
    with summary_path.open("w") as f:
        f.write(f"source: {args.parquet}\nconf_min: {args.conf_min}  margin_min: {args.margin_min}\n")
        f.write(f"min_chars: {args.min_chars}  max_chars: {args.max_chars}\n\n")
        for cat, w, s, e, st in summary:
            f.write(f"{cat:15s}  wrote {w:>6d}  short-skip {s:>5d}  err {e:>4d}  [{st}]\n")
    print(f"\nwrote summary to {summary_path}")

    usable = [(cat, w) for cat, w, _, _, st in summary if st == "OK"]
    print(f"\ncategories with >={args.min_per_cat} usable docs: {len(usable)}")
    for cat, w in sorted(usable, key=lambda x: -x[1]):
        print(f"  {cat:15s}  {w:,} docs")


if __name__ == "__main__":
    main()
