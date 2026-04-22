"""
Fast byte-offset indexer using numpy vectorized newline scan.
~100x faster than pure-Python byte iteration.

Usage: python index_shards.py --shards 2-11
"""
import argparse, os, time
from pathlib import Path
import numpy as np

RAW_DIR = Path("/root/base_pipeline/caches/climbmix_raw")
IDX_DIR = Path("/root/base_pipeline/caches/climbmix_inspect")
IDX_DIR.mkdir(parents=True, exist_ok=True)

CHUNK = 64 * 1024 * 1024  # 64 MB


def index_shard(shard: int) -> int:
    src = RAW_DIR / f"part_{shard}.jsonl"
    dst = IDX_DIR / f"part_{shard}.idx.npy"
    offsets = [0]  # first line starts at 0
    pos = 0
    with src.open("rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            # vectorized search for '\n' bytes
            arr = np.frombuffer(chunk, dtype=np.uint8)
            nl_local = np.flatnonzero(arr == 0x0A)
            # offsets of next-line starts = pos + nl_index + 1
            offsets.append(nl_local + (pos + 1))
            pos += len(chunk)
    # concat: first entry is scalar 0, rest are arrays
    parts = []
    for e in offsets:
        if np.isscalar(e):
            parts.append(np.asarray([e], dtype=np.int64))
        else:
            parts.append(e.astype(np.int64))
    all_offsets = np.concatenate(parts)
    # if file ends with '\n', last offset == pos (past-end) - remove
    if all_offsets.size and all_offsets[-1] >= pos:
        all_offsets = all_offsets[:-1]
    np.save(dst, all_offsets)
    return int(all_offsets.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=str, default="2-11")
    args = ap.parse_args()
    shards = []
    for p in args.shards.split(","):
        if "-" in p:
            a, b = p.split("-")
            shards.extend(range(int(a), int(b) + 1))
        else:
            shards.append(int(p))
    total = 0
    for s in shards:
        t0 = time.time()
        n = index_shard(s)
        dt = time.time() - t0
        print(f"shard {s:>2d}: {n:>10d} lines  in {dt:5.1f}s  ({(os.path.getsize(RAW_DIR/f'part_{s}.jsonl')/1e9)/dt:.2f} GB/s)", flush=True)
        total += n
    print(f"total: {total} lines across {len(shards)} shards", flush=True)


if __name__ == "__main__":
    main()
