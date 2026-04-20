"""
Region-selectable eval: compute KL(teacher || student) over a chosen span
within the cached sequences. Needs a cache that stores FULL-sequence teacher
logits (built via runs/build_fullseq_teacher_cache.py).

Regions:
  cont    — continuation positions only (match validator / current metric)
  prompt  — prompt positions only (what the model "knows" given context,
            excludes generated garbage)
  all     — every non-pad position
  answer  — positions inside the prompt that follow an "Answer:" marker
            (captures the spans where the model is predicting a canonical
            answer given a prior question)

Outputs a JSON per-prompt KL dump (same shape as eval_bootstrap.py) so
runs/compare_pp.py works as-is.

Usage:
    python runs/eval_bootstrap_region.py --model-repo MODEL \
        --cache /ephemeral/teacher_cache_60_fullseq.pt \
        --region cont \
        --out /ephemeral/logs/pp_MODEL_cont.json
"""
import argparse
import json
import os
import random
import statistics
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

EVAL_BATCH_SIZE_DEFAULT = 4
BOOTSTRAP_N = 10_000
MIN_COMPLETION_TOKENS = 64


def build_answer_mask(tokens, tokenizer, window=20):
    """Locate positions in the prompt that follow an 'Answer:' marker.
    Returns a bool mask over token positions (True = this is an answer position).
    """
    text_ids = tokens.tolist()
    mask = [False] * len(text_ids)
    # Find all 'Answer:' occurrences — the token 'Answer' may encode as one or
    # two BPE pieces depending on preceding char. We match by decoding windows.
    for i in range(len(text_ids) - 2):
        chunk = tokenizer.decode(text_ids[i:i+2], skip_special_tokens=False)
        if "Answer:" in chunk or "answer:" in chunk:
            # mark up to `window` tokens after the match (bounded by sequence)
            start = i + 2
            end = min(len(text_ids), start + window)
            # stop at the next newline for cleaner answer-span
            for j in range(start, end):
                tok_txt = tokenizer.decode([text_ids[j]], skip_special_tokens=False)
                if "\n" in tok_txt:
                    break
                mask[j] = True
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-repo", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--region", required=True,
                    choices=["cont", "prompt", "all", "answer", "cont_top128"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--min-completion-tokens", type=int, default=MIN_COMPLETION_TOKENS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--student-temp", type=float, default=1.0,
                    help="Scale student logits by 1/temp before softmax (inference-time). "
                         "temp<1 sharpens, temp>1 flattens.")
    ap.add_argument("--topk", type=int, default=128,
                    help="Top-k for sparse KL (only used with region=cont_top128).")
    ap.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE_DEFAULT)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-35B-A3B",
                    help="Tokenizer for answer-span detection (shared Qwen3.5 vocab)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[load cache] {args.cache}", flush=True)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False, mmap=True)
    full_sequences = cache["full_sequences"]
    prompt_lens = cache["prompt_lens"]
    # Prefer full-seq logits if available
    if "teacher_logits_full" in cache:
        teacher_logits_full = cache["teacher_logits_full"]
        print(f"  using full-sequence teacher logits", flush=True)
    else:
        raise RuntimeError("Cache does not contain teacher_logits_full; "
                           "rebuild with build_fullseq_teacher_cache.py")
    print(f"  {len(full_sequences)} prompts", flush=True)

    # Apply same min-completion filter as eval_bootstrap.py
    kept = []
    for i, (seq, plen) in enumerate(zip(full_sequences, prompt_lens)):
        cont_len = seq.shape[1] - plen
        if cont_len >= args.min_completion_tokens:
            kept.append(i)
    print(f"  {len(kept)} after min_completion_tokens={args.min_completion_tokens}",
          flush=True)
    full_sequences = [full_sequences[i] for i in kept]
    prompt_lens = [prompt_lens[i] for i in kept]
    teacher_logits_full = [teacher_logits_full[i] for i in kept]

    # Tokenizer for answer-span mode
    tokenizer = None
    if args.region == "answer":
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print(f"[load student] {args.model_repo}", flush=True)
    t0 = time.time()
    student = AutoModelForCausalLM.from_pretrained(
        args.model_repo, dtype=dtype, trust_remote_code=True,
    ).to(args.device).to(dtype)
    student.eval()
    print(f"  loaded in {time.time()-t0:.1f}s VRAM={torch.cuda.memory_allocated(args.device)/1e9:.1f}GB",
          flush=True)

    n = len(full_sequences)
    per_prompt_kl = []
    per_prompt_n_positions = []

    full_sequences_gpu = [s.to(args.device) for s in full_sequences]

    print(f"[score region={args.region}] {n} prompts, batch={args.batch_size}", flush=True)
    t0 = time.time()
    for batch_start in range(0, n, args.batch_size):
        batch_end = min(batch_start + args.batch_size, n)
        batch_seqs = full_sequences_gpu[batch_start:batch_end]
        batch_plens = prompt_lens[batch_start:batch_end]
        batch_t = teacher_logits_full[batch_start:batch_end]

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
            if args.student_temp != 1.0:
                s_logits.div_(args.student_temp)

        for j in range(len(batch_seqs)):
            plen = batch_plens[j]
            L = batch_seqs[j].shape[1]
            t_full = batch_t[j].to(args.device).float()  # [L, V]

            # Decide which positions to score. We always score "predict token k
            # given context up to k-1", using logits[k-1]. So the score
            # positions in logit-space are (start-1, end-1).
            if args.region == "cont":
                start = plen - 1
                end = L - 1
            elif args.region == "cont_top128":
                # Validator-style sparse top-K KL over continuation positions.
                # Teacher: softmax over its own top-K logits.
                # Student: full-vocab log_softmax, gather at teacher top-K indices,
                # then renormalize over the same K support.
                start = plen - 1
                end = L - 1
                if end <= start:
                    continue
                K = args.topk
                total_kl = 0.0
                n_pos = end - start
                CHUNK = 256
                for c0 in range(start, end, CHUNK):
                    c1 = min(c0 + CHUNK, end)
                    t_slice = t_full[c0:c1, :]
                    s_slice = s_logits[j, c0:c1, :]
                    t_topk_vals, t_topk_idx = t_slice.topk(K, dim=-1)
                    t_lp_k = F.log_softmax(t_topk_vals, dim=-1)
                    s_lp_full = F.log_softmax(s_slice, dim=-1)
                    s_lp_k = s_lp_full.gather(-1, t_topk_idx)
                    s_lp_k = s_lp_k - s_lp_k.logsumexp(-1, keepdim=True)
                    chunk_kl = F.kl_div(s_lp_k, t_lp_k, log_target=True,
                                         reduction="none").sum(-1).sum().item()
                    total_kl += chunk_kl
                    del t_slice, s_slice, t_topk_vals, t_topk_idx, t_lp_k, s_lp_full, s_lp_k
                per_prompt_kl.append(total_kl / n_pos)
                per_prompt_n_positions.append(n_pos)
                continue
            elif args.region == "prompt":
                start = 0
                end = plen - 1
            elif args.region == "all":
                start = 0
                end = L - 1
            elif args.region == "answer":
                seq_tokens = batch_seqs[j][0]
                ans_mask = build_answer_mask(seq_tokens, tokenizer, window=20)
                # positions we want to score: positions k where ans_mask[k]=True
                # convert to logit indices k-1
                score_ix = [k - 1 for k in range(1, L) if ans_mask[k]]
                if not score_ix:
                    continue
                s_lp = F.log_softmax(s_logits[j:j+1, score_ix, :], dim=-1)
                t_lp = F.log_softmax(t_full[score_ix, :].unsqueeze(0), dim=-1)
                kl = F.kl_div(s_lp, t_lp, log_target=True, reduction="none").sum(-1).mean().item()
                per_prompt_kl.append(kl)
                per_prompt_n_positions.append(len(score_ix))
                continue

            if end <= start:
                continue
            # span-based, chunked over positions to avoid OOM on long prompts
            total_kl = 0.0
            n_pos = end - start
            CHUNK = 256
            for c0 in range(start, end, CHUNK):
                c1 = min(c0 + CHUNK, end)
                t_slice = t_full[c0:c1, :]
                s_slice = s_logits[j, c0:c1, :]
                t_lp = F.log_softmax(t_slice, dim=-1)
                s_lp = F.log_softmax(s_slice, dim=-1)
                chunk_kl = F.kl_div(s_lp, t_lp, log_target=True,
                                     reduction="none").sum(-1).sum().item()
                total_kl += chunk_kl
                del t_slice, s_slice, t_lp, s_lp
            per_prompt_kl.append(total_kl / n_pos)
            per_prompt_n_positions.append(n_pos)

        del s_logits, padded, attn
        if batch_end % 20 == 0 or batch_end == n:
            print(f"  {batch_end}/{n} | mean={sum(per_prompt_kl)/len(per_prompt_kl):.6f} "
                  f"| {time.time()-t0:.1f}s", flush=True)

    if not per_prompt_kl:
        print("[abort] no positions scored", flush=True)
        return

    kl_mean = sum(per_prompt_kl) / len(per_prompt_kl)
    kl_std = statistics.stdev(per_prompt_kl) if len(per_prompt_kl) > 1 else 0.0
    kl_se = kl_std / (len(per_prompt_kl) ** 0.5)
    normal_lo = kl_mean - 1.96 * kl_se
    normal_hi = kl_mean + 1.96 * kl_se
    rng = random.Random(args.seed)
    boots = []
    for _ in range(BOOTSTRAP_N):
        sample = [per_prompt_kl[rng.randrange(len(per_prompt_kl))] for _ in range(len(per_prompt_kl))]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    boot_lo = boots[int(0.025 * BOOTSTRAP_N)]
    boot_hi = boots[int(0.975 * BOOTSTRAP_N)]

    print(f"\n  KL mean:     {kl_mean:.6f}")
    print(f"  std:         {kl_std:.6f} (se={kl_se:.6f})")
    print(f"  95% normal:  [{normal_lo:.6f}, {normal_hi:.6f}]")
    print(f"  95% boot:    [{boot_lo:.6f}, {boot_hi:.6f}]")
    print(f"  n = {len(per_prompt_kl)}  avg positions/prompt = "
          f"{sum(per_prompt_n_positions)/len(per_prompt_n_positions):.1f}")

    out = {
        "model": args.model_repo,
        "region": args.region,
        "n_prompts": len(per_prompt_kl),
        "kl_mean": kl_mean,
        "kl_std_prompts": kl_std,
        "kl_se": kl_se,
        "normal_ci_95": [normal_lo, normal_hi],
        "bootstrap_ci_95": [boot_lo, boot_hi],
        "per_prompt_kl": per_prompt_kl,
        "per_prompt_n_positions": per_prompt_n_positions,
        "cache": args.cache,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[write] {args.out}", flush=True)


if __name__ == "__main__":
    main()
