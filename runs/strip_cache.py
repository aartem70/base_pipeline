"""
Strip teacher_logits from cont_cache_600.pt to avoid OOM during training.

Leaves sequences + prompt_lens. Saves to a new small cache.
"""
import argparse
import os
import gc
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[load] {args.inp}  (mmap=True so we don't blow RAM)", flush=True)
    d = torch.load(args.inp, map_location="cpu", weights_only=False, mmap=True)
    small = {
        "sequences": [s.clone() for s in d["sequences"]],
        "prompt_lens": list(d["prompt_lens"]),
    }
    print(f"  n={len(small['sequences'])}", flush=True)
    del d
    gc.collect()

    print(f"[save] {args.out}", flush=True)
    torch.save(small, args.out)
    sz = os.path.getsize(args.out) / 1e6
    print(f"  size={sz:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
