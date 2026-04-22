"""
Prod-validator-matching eval: top-128 renorm KL on continuation positions,
MIN_COMPLETION_TOKENS=10.

Mirrors `compute_kl_from_sparse` in unarbos/distil/scripts/pod_eval_vllm.py
exactly, and aggregates per-prompt then globally the same way that file does.

Equation (per-position):
    t_log_p        = log_softmax_k( V_t[i,p,:] )                                # renorm teacher's top-K
    s_log_p_full   = log_softmax_v( student_logits[i,p,:] )                     # full-vocab student
    s_log_p_k      = s_log_p_full.gather(-1, I_t[i,p,:])
    s_log_p_k_norm = s_log_p_k - logsumexp_k( s_log_p_k[i,p,:] )                # renorm student over same K
    KL[i,p]        = sum_k exp(t_log_p[k]) * ( t_log_p[k] - s_log_p_k_norm[k] )

Per-prompt:  kl_mean_i     = mean_p KL[i,p]
Global:      kl_global_avg = mean_i kl_mean_i     (arithmetic, not weighted by gen_len)

Cache format must be the top-128 sparse version produced by
runs/build_eval_cache.py .

Usage:
    python runs/eval_bootstrap.py \\
        --model-repo kharchevnykov/distil \\
        --cache /root/base_pipeline/caches/teacher_cache_60_top128.pt \\
        --out /root/base_pipeline/logs/per_cat_evals/distil_baseline_top128.json
"""

import argparse
import json
import os
import random
import statistics
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

MIN_COMPLETION_TOKENS = 10     # prod: pod_eval_vllm.py line 77
BOOTSTRAP_N = 10_000
KL_CHUNK_SIZE = 128             # prod: pod_eval_vllm.py compute_kl_from_sparse chunk size


def compute_kl_from_sparse(t_indices, t_values_logprobs, student_logits):
    """Exact port of pod_eval_vllm.compute_kl_from_sparse (values_are_logprobs=True).

    t_indices       : [1, gen_len, K] int64        — teacher's top-K vocab indices
    t_values_logprobs: [1, gen_len, K] float32     — teacher's top-K log-probs
    student_logits  : [1, gen_len, V] float32      — student raw logits on same positions

    Returns kl_per_pos : [gen_len]   (squeezed to 1-D as prod does after .squeeze(0))
    """
    device = student_logits.device
    t_idx = t_indices.to(device)
    t_vals = t_values_logprobs.to(device).float()

    # Teacher: renormalize top-K log-probs to a proper dist over K
    t_log_p = t_vals - t_vals.logsumexp(dim=-1, keepdim=True)

    # Student: full-vocab log-softmax, gather at teacher's K indices, renorm over K
    s_log_p_full = F.log_softmax(student_logits.float(), dim=-1)
    s_log_p_k = s_log_p_full.gather(-1, t_idx)
    s_log_p_k_norm = s_log_p_k - s_log_p_k.logsumexp(dim=-1, keepdim=True)
    del s_log_p_full

    # Chunked KL over positions (matches prod)
    B, n_pos, _K = t_log_p.shape
    kl_per_pos = torch.empty(B, n_pos, device=device)
    for i in range(0, n_pos, KL_CHUNK_SIZE):
        j = min(i + KL_CHUNK_SIZE, n_pos)
        kl_per_pos[:, i:j] = F.kl_div(
            s_log_p_k_norm[:, i:j, :], t_log_p[:, i:j, :],
            log_target=True, reduction="none",
        ).sum(dim=-1)
    return kl_per_pos.squeeze(0)   # [gen_len]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-repo", required=True)
    ap.add_argument("--cache", required=True, help="top-128 sparse cache (see build_eval_cache.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--min-completion-tokens", type=int, default=MIN_COMPLETION_TOKENS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[load cache] {args.cache}", flush=True)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    entries = cache["entries"]
    K = cache.get("logprobs_k", 128)
    print(f"  {len(entries)} prompts, K={K}", flush=True)

    # Filter by min_completion_tokens (prod filters prompts, not positions)
    kept = [e for e in entries if e.get("gen_len", 0) >= args.min_completion_tokens
            and e.get("teacher_topk_indices") is not None]
    print(f"  {len(kept)}/{len(entries)} prompts after min_completion_tokens={args.min_completion_tokens}",
          flush=True)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"[load student] {args.model_repo}", flush=True)
    t0 = time.time()
    student = AutoModelForCausalLM.from_pretrained(
        args.model_repo, dtype=dtype, trust_remote_code=True,
    ).to(args.device)
    student.eval()
    print(f"  loaded in {time.time()-t0:.1f}s, VRAM={torch.cuda.memory_allocated(args.device)/1e9:.1f}GB",
          flush=True)

    # Score sequentially (matches prod's per-prompt loop at lines 3372-3449).
    # Each prompt has its own full_seq length, so no benefit from batching.
    per_prompt_kl = []
    t0 = time.time()
    with torch.no_grad():
        for i, e in enumerate(kept):
            full_seq = e["full_ids"].to(args.device)   # [1, L]
            plen = int(e["prompt_len"])

            s_logits = student(full_seq).logits.float()
            cont_s = s_logits[:, plen - 1:-1, :]         # [1, gen_len, V]

            t_idx = e["teacher_topk_indices"]            # [1, gen_len, K]
            t_lp = e["teacher_topk_logprobs"]            # [1, gen_len, K]
            min_len = min(cont_s.shape[1], t_idx.shape[1])
            if min_len == 0:
                per_prompt_kl.append(0.0)
                del s_logits, cont_s
                continue

            kl_per_pos = compute_kl_from_sparse(
                t_idx[:, :min_len, :], t_lp[:, :min_len, :], cont_s[:, :min_len, :],
            )
            kl_mean_i = kl_per_pos.mean().item()
            per_prompt_kl.append(kl_mean_i)

            del s_logits, cont_s, t_idx, t_lp, kl_per_pos
            if (i + 1) % 10 == 0 or (i + 1) == len(kept):
                running = sum(per_prompt_kl) / len(per_prompt_kl)
                print(f"  [{i+1}/{len(kept)}] KL={kl_mean_i:.6f}  running={running:.6f}  "
                      f"{time.time()-t0:.1f}s", flush=True)

    # Global aggregate (matches prod line 3528: arithmetic mean over prompts)
    n = len(per_prompt_kl)
    kl_mean = sum(per_prompt_kl) / n
    kl_std = statistics.stdev(per_prompt_kl) if n > 1 else 0.0
    kl_se = kl_std / (n ** 0.5) if n > 0 else 0.0
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
    print(f"  kl_global_avg  : {kl_mean:.6f}   (prod: kl_global_avg)")
    print(f"  std / se       : {kl_std:.6f} / {kl_se:.6f}")
    print(f"  95% normal     : [{normal_ci_low:.6f}, {normal_ci_high:.6f}]")
    print(f"  95% bootstrap  : [{boot_ci_low:.6f}, {boot_ci_high:.6f}]")
    print(f"  n_prompts_kept : {n}")

    out = {
        "model": args.model_repo,
        "scoring": "top128_renorm_kl",
        "logprobs_k": K,
        "min_completion_tokens": args.min_completion_tokens,
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
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[write] {args.out}", flush=True)


if __name__ == "__main__":
    main()
