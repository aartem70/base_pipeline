"""
Filter a continuation cache by teacher-output n-gram repetition score.

For each sample, computes the fraction of n-grams in the teacher's continuation
that are repeats of an earlier n-gram. Drops samples above --threshold.

Input cache format: {sequences, prompt_lens, teacher_logits}.
Output preserves the same format with only clean samples.

Usage:
    python runs/filter_cache_by_repetition.py \
        --cache /ephemeral/cont_cache_600_multishard.pt \
        --out /ephemeral/cont_cache_600_multishard_clean.pt \
        --threshold 0.30 --n 8
"""
import argparse
import os
from collections import Counter

import torch


def repetition_score(token_ids, n=8):
    if len(token_ids) < 2 * n:
        return 0.0
    ngrams = [tuple(token_ids[i:i+n]) for i in range(len(token_ids)-n+1)]
    c = Counter(ngrams)
    repeated = sum(c[g] - 1 for g in c if c[g] > 1)
    return repeated / len(ngrams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--n", type=int, default=8, help="n-gram size")
    args = ap.parse_args()

    print(f"[load] {args.cache}", flush=True)
    c = torch.load(args.cache, map_location="cpu", weights_only=False, mmap=True)
    seqs = c["sequences"]
    plens = c["prompt_lens"]
    tlogits = c["teacher_logits"]
    n_total = len(seqs)

    keep = []
    scores = []
    for i in range(n_total):
        cont = seqs[i][plens[i]:].tolist()
        r = repetition_score(cont, n=args.n)
        scores.append(r)
        if r <= args.threshold:
            keep.append(i)

    n_keep = len(keep)
    print(f"[filter] threshold={args.threshold}  n-gram={args.n}", flush=True)
    print(f"  kept {n_keep}/{n_total} ({100*n_keep/n_total:.1f}%)", flush=True)
    import statistics
    print(f"  rep score — kept mean={statistics.mean([scores[i] for i in keep]):.3f}  "
          f"dropped mean={statistics.mean([s for i,s in enumerate(scores) if i not in set(keep)]):.3f}",
          flush=True)

    new_seqs = [seqs[i] for i in keep]
    new_plens = [plens[i] for i in keep]
    # must .clone() the teacher_logits slice to force read from mmap
    new_tlogits = [tlogits[i].clone() for i in keep]

    print(f"[save] {args.out}", flush=True)
    tmp = args.out + ".tmp"
    torch.save({
        "sequences": new_seqs,
        "prompt_lens": new_plens,
        "teacher_logits": new_tlogits,
    }, tmp)
    os.replace(tmp, args.out)
    print(f"[done] {n_keep} clean samples", flush=True)


if __name__ == "__main__":
    main()
