"""
Rebuild a teacher cache storing FULL-sequence teacher logits (prompt + continuation)
instead of continuation-only. Reuses the sequences + prompt_lens from an existing
eval cache; only reruns the teacher forward pass.

Usage:
    python runs/build_fullseq_teacher_cache.py \
        --in /ephemeral/teacher_cache_60.pt \
        --out /ephemeral/teacher_cache_60_fullseq.pt
"""
import argparse
import os
import time

import torch
from transformers import AutoModelForCausalLM

TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[load] {args.inp}", flush=True)
    cache = torch.load(args.inp, map_location="cpu", weights_only=False)
    seqs = cache["full_sequences"]
    plens = cache["prompt_lens"]
    n = len(seqs)
    print(f"  n={n} sequences", flush=True)

    print(f"[load teacher] {TEACHER_MODEL}", flush=True)
    t0 = time.time()
    try:
        teacher = AutoModelForCausalLM.from_pretrained(
            TEACHER_MODEL, dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, attn_implementation="flash_attention_2",
        )
    except Exception:
        teacher = AutoModelForCausalLM.from_pretrained(
            TEACHER_MODEL, dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )
    teacher.eval()
    print(f"  loaded in {time.time()-t0:.1f}s  VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB",
          flush=True)

    full_logits = []
    t0 = time.time()
    with torch.no_grad():
        for i, seq in enumerate(seqs):
            s = seq.to(teacher.device)
            out = teacher(s)
            # Full-sequence logits (including prompt positions)
            logits = out.logits[0].float().cpu()  # [L, V]
            full_logits.append(logits)
            del out
            if (i + 1) % 5 == 0 or i == n - 1:
                print(f"  [{i+1}/{n}] L={seq.shape[1]}  elapsed={time.time()-t0:.0f}s",
                      flush=True)

    print(f"[save] {args.out}", flush=True)
    tmp = args.out + ".tmp"
    torch.save({
        "full_sequences": seqs,
        "prompt_lens": plens,
        "teacher_logits_full": full_logits,
        # also retain continuation-only slice for backwards compat
        "teacher_logits": [l[p-1:-1, :].unsqueeze(0) if (seqs[i].shape[1] - p) > 0
                           else l[p-1:p, :].unsqueeze(0)
                           for i, (l, p) in enumerate(zip(full_logits, plens))],
    }, tmp)
    os.replace(tmp, args.out)
    print(f"[done] {n} prompts, total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
