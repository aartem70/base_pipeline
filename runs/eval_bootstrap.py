"""
Bootstrap eval: replicate evaluate.py CHECK 10 scoring on a cached teacher dump,
save per-prompt KL scores, and report mean + normal CI + bootstrap CI.

Per-prompt results dumped to JSON so we can paired-bootstrap two models later.

Usage:
    python eval_bootstrap.py \
        --model-repo kharchevnykov/distil \
        --cache /ephemeral/teacher_cache_60.pt \
        --out /ephemeral/logs/pp_distil_baseline.json
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

MIN_COMPLETION_TOKENS = 64
EVAL_BATCH_SIZE = 4
BOOTSTRAP_N = 10_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-repo", required=True,
                    help="HF repo or local dir of student")
    ap.add_argument("--cache", required=True, help="Path to teacher_cache_*.pt")
    ap.add_argument("--out", required=True, help="JSON output path")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--min-completion-tokens", type=int, default=MIN_COMPLETION_TOKENS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[load cache] {args.cache}", flush=True)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    full_sequences = cache["full_sequences"]
    teacher_logits_list = cache["teacher_logits"]
    prompt_lens = cache["prompt_lens"]
    print(f"  {len(full_sequences)} prompts in cache", flush=True)

    # Apply min_completion_tokens filter (same as evaluate.py)
    kept_idx = []
    for i, (seq, plen) in enumerate(zip(full_sequences, prompt_lens)):
        cont_len = seq.shape[1] - plen
        if cont_len >= args.min_completion_tokens:
            kept_idx.append(i)
    print(f"  {len(kept_idx)} prompts after min_completion_tokens={args.min_completion_tokens}", flush=True)
    full_sequences = [full_sequences[i] for i in kept_idx]
    teacher_logits_list = [teacher_logits_list[i] for i in kept_idx]
    prompt_lens = [prompt_lens[i] for i in kept_idx]

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"[load student] {args.model_repo}", flush=True)
    t0 = time.time()
    student = AutoModelForCausalLM.from_pretrained(
        args.model_repo,
        dtype=dtype,
        trust_remote_code=True,
    ).to(args.device)
    student.eval()
    print(f"  loaded in {time.time() - t0:.1f}s, VRAM={torch.cuda.memory_allocated(args.device) / 1e9:.1f}GB", flush=True)

    n = len(full_sequences)
    per_prompt_kl = []

    full_sequences = [s.to(args.device) for s in full_sequences]

    print(f"[score] {n} prompts, batch={EVAL_BATCH_SIZE}", flush=True)
    t0 = time.time()
    for batch_start in range(0, n, EVAL_BATCH_SIZE):
        batch_end = min(batch_start + EVAL_BATCH_SIZE, n)
        batch_seqs = full_sequences[batch_start:batch_end]
        batch_plens = prompt_lens[batch_start:batch_end]
        batch_t = teacher_logits_list[batch_start:batch_end]

        max_len = max(s.shape[1] for s in batch_seqs)
        padded = torch.zeros(len(batch_seqs), max_len,
                             dtype=batch_seqs[0].dtype, device=args.device)
        attn = torch.zeros(len(batch_seqs), max_len, dtype=torch.long, device=args.device)
        for j, seq in enumerate(batch_seqs):
            L = seq.shape[1]
            padded[j, :L] = seq[0]
            attn[j, :L] = 1

        with torch.no_grad():
            s_logits = student(padded, attention_mask=attn).logits.float()

        for j in range(len(batch_seqs)):
            plen = batch_plens[j]
            L = batch_seqs[j].shape[1]
            t_logits = batch_t[j].to(args.device).float()
            t_lp = F.log_softmax(t_logits, dim=-1)
            s_cont = s_logits[j:j+1, plen - 1:L - 1, :]
            m = min(s_cont.shape[1], t_lp.shape[1])
            s_lp = F.log_softmax(s_cont[:, :m, :], dim=-1)
            t_lp_s = t_lp[:, :m, :]
            kl = F.kl_div(s_lp, t_lp_s, log_target=True, reduction="none").sum(dim=-1).mean().item()
            per_prompt_kl.append(kl)

        del s_logits, padded, attn
        if (batch_end) % 20 == 0 or batch_end == n:
            print(f"  {batch_end}/{n} | mean={sum(per_prompt_kl)/len(per_prompt_kl):.6f} | "
                  f"{time.time() - t0:.1f}s", flush=True)

    kl_mean = sum(per_prompt_kl) / len(per_prompt_kl)
    kl_std = statistics.stdev(per_prompt_kl) if len(per_prompt_kl) > 1 else 0.0
    kl_se = kl_std / (len(per_prompt_kl) ** 0.5)
    normal_ci_low = kl_mean - 1.96 * kl_se
    normal_ci_high = kl_mean + 1.96 * kl_se

    # Bootstrap CI
    rng = random.Random(args.seed)
    boots = []
    for _ in range(BOOTSTRAP_N):
        sample = [per_prompt_kl[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    boot_ci_low = boots[int(0.025 * BOOTSTRAP_N)]
    boot_ci_high = boots[int(0.975 * BOOTSTRAP_N)]
    boot_std = statistics.stdev(boots)

    print("")
    print(f"  KL mean     : {kl_mean:.6f}")
    print(f"  std (prompts): {kl_std:.6f}  (se = {kl_se:.6f})")
    print(f"  95% normal   : [{normal_ci_low:.6f}, {normal_ci_high:.6f}]")
    print(f"  95% bootstrap: [{boot_ci_low:.6f}, {boot_ci_high:.6f}]  (bootstrap sd={boot_std:.6f})")
    print(f"  n = {n}")

    out = {
        "model": args.model_repo,
        "n_prompts": n,
        "kl_mean": kl_mean,
        "kl_std_prompts": kl_std,
        "kl_se": kl_se,
        "normal_ci_95": [normal_ci_low, normal_ci_high],
        "bootstrap_ci_95": [boot_ci_low, boot_ci_high],
        "bootstrap_sd": boot_std,
        "bootstrap_n": BOOTSTRAP_N,
        "per_prompt_kl": per_prompt_kl,
        "cache": args.cache,
        "min_completion_tokens": args.min_completion_tokens,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[write] {args.out}", flush=True)


if __name__ == "__main__":
    main()
