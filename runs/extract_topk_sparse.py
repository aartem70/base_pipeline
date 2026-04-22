"""
Read a Phase-1 SGLang cache checkpoint (sequences only) and run the teacher
forward pass to extract SPARSE top-K logits per position. Stored as:
  full_sequences     list[LongTensor[1, L]]
  prompt_lens        list[int]
  teacher_topk_vals  list[FloatTensor[L, K]]    top-K teacher logits
  teacher_topk_idx   list[LongTensor[L, K]]     matching token indices

This matches the live eval's sparse top-K format (stored as raw logits; the
eval code handles renormalization). ~500 MB for 60 prompts × 8k tokens × 128.
"""
import argparse
import os
import time

import torch
from transformers import AutoModelForCausalLM

TEACHER = "Qwen/Qwen3.5-35B-A3B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Phase 1 checkpoint .pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print(f"[load ckpt] {args.ckpt}", flush=True)
    data = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    seqs = data["sequences"]
    plens = data["prompt_lens"]
    n = len(seqs)
    print(f"  n={n} sequences", flush=True)

    print(f"[load teacher] {TEACHER}", flush=True)
    t0 = time.time()
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER, dtype=torch.bfloat16, device_map={"": args.device},
        trust_remote_code=True,
    )
    teacher.eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    full_sequences = []
    out_plens = []
    topk_vals = []
    topk_idx = []

    t0 = time.time()
    with torch.no_grad():
        for i, (seq, plen) in enumerate(zip(seqs, plens)):
            s = seq.unsqueeze(0).to(args.device)  # [1, L]
            logits = teacher(s).logits[0]         # [L, V] bf16
            vals, idx = logits.topk(args.topk, dim=-1)
            topk_vals.append(vals.float().cpu())  # [L, K] fp32
            topk_idx.append(idx.cpu())            # [L, K] int64
            full_sequences.append(seq.unsqueeze(0).cpu())
            out_plens.append(int(plen))
            del logits, vals, idx
            if (i + 1) % 5 == 0 or i == n - 1:
                print(f"  [{i+1}/{n}] L={s.shape[1]}  elapsed={time.time()-t0:.0f}s",
                      flush=True)

    print(f"[save] {args.out}", flush=True)
    tmp = args.out + ".tmp"
    torch.save({
        "full_sequences": full_sequences,
        "prompt_lens": out_plens,
        "teacher_topk_vals": topk_vals,
        "teacher_topk_idx": topk_idx,
        "topk": args.topk,
    }, tmp)
    os.replace(tmp, args.out)
    print(f"[done] {n} prompts", flush=True)


if __name__ == "__main__":
    main()
