"""
Build a prod-validator-matching teacher cache at seed S with N prompts.

Matches exactly what unarbos/distil/scripts/pod_eval_vllm.py stores as the
sparse "teacher_logits_list" entry:

    - full_ids        : [1, prompt_len + gen_len] token ids
    - prompt_len      : int
    - gen_len         : int
    - teacher_topk_indices : [1, gen_len, K]  dtype=int64
    - teacher_topk_logprobs: [1, gen_len, K]  dtype=float32   (log-probs, negative)

K defaults to 128 (prod default; --logprobs-k 128 in pod_eval_vllm.py).

Generation config matches prod greedy path (block_seed is None):
    temperature=0.0, top_p=1.0, do_sample=False, max_new_tokens=512

Cache file is ~20 MB for N=60 (vs 16 GB for full-vocab cache).

Usage:
    python runs/build_eval_cache.py --seed 42 --n 60 \
        --output /root/base_pipeline/caches/teacher_cache_60_top128.pt
"""

import argparse
import os
import random
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"
CLIMBMIX_DATASET = "karpathy/climbmix-400b-shuffle"
CLIMBMIX_NUM_SHARDS = 6542
VALIDATOR_MAX_NEW_TOKENS = 512
LOGPROBS_K = 128  # prod default: pod_eval_vllm.py --logprobs-k 128


def sample_random_prompts(n, seed, min_chars=500, max_chars=10000):
    from datasets import load_dataset
    rng = random.Random(seed)
    shard_idx = rng.randint(0, CLIMBMIX_NUM_SHARDS - 1)
    shard_file = f"shard_{shard_idx:05d}.parquet"
    print(f"[dataset] seed={seed} → shard={shard_idx}/{CLIMBMIX_NUM_SHARDS}", flush=True)
    ds = load_dataset(CLIMBMIX_DATASET, data_files=shard_file, split="train")
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    prompts = []
    for idx in indices:
        text = ds[idx].get("text", "")
        if not text or len(text) < min_chars:
            continue
        if len(text) > max_chars:
            text = text[:max_chars]
            last_space = text.rfind(" ")
            if last_space > max_chars // 2:
                text = text[:last_space]
        prompts.append(text)
        if len(prompts) >= n:
            break
    print(f"[dataset] got {len(prompts)} prompts from shard {shard_idx}", flush=True)
    return prompts, shard_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=VALIDATOR_MAX_NEW_TOKENS)
    ap.add_argument("--logprobs_k", type=int, default=LOGPROBS_K,
                    help="Top-K log-probs to store per generated position (default 128, matches prod).")
    args = ap.parse_args()

    prompts, shard_idx = sample_random_prompts(args.n, args.seed)
    if len(prompts) < args.n:
        raise RuntimeError(f"only got {len(prompts)}/{args.n} prompts")

    print(f"[load teacher] {TEACHER_MODEL}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
    try:
        teacher = AutoModelForCausalLM.from_pretrained(
            TEACHER_MODEL, dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, attn_implementation="flash_attention_2",
        )
    except Exception as e:
        print(f"[warn] flash_attn failed ({e}); falling back", flush=True)
        teacher = AutoModelForCausalLM.from_pretrained(
            TEACHER_MODEL, dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )
    teacher.eval()
    print(f"  loaded in {time.time()-t0:.1f}s  VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB",
          flush=True)

    cache_entries = []
    t0 = time.time()
    with torch.no_grad():
        for i, text in enumerate(prompts):
            ids = tok(text, return_tensors="pt", truncation=False).input_ids.to(teacher.device)
            plen = ids.shape[1]

            # prod path: greedy generation when block_seed is None
            out_ids = teacher.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,          # greedy (temperature=0, top_p=1)
                use_cache=True,
            )
            gen_len = out_ids.shape[1] - plen
            if gen_len == 0:
                cache_entries.append({
                    "full_ids": out_ids.cpu(),
                    "prompt_len": int(plen),
                    "gen_len": 0,
                    "teacher_topk_indices": None,
                    "teacher_topk_logprobs": None,
                })
                print(f"  [{i+1}/{len(prompts)}] plen={plen} gen=0  (skipped by min_completion filter at eval)",
                      flush=True)
                continue

            # forward pass on full sequence, take continuation slice only
            logits = teacher(out_ids).logits.float()
            # logits[p] predicts token p+1, so continuation logits live at
            # positions [plen-1 : -1]  (length = gen_len)
            cont_logits = logits[:, plen - 1:-1, :]   # [1, gen_len, V]

            # full-vocab log_softmax, then top-K
            cont_logp = F.log_softmax(cont_logits, dim=-1)
            topk_vals, topk_idx = cont_logp.topk(args.logprobs_k, dim=-1)   # [1, gen_len, K]

            cache_entries.append({
                "full_ids": out_ids.cpu(),
                "prompt_len": int(plen),
                "gen_len": int(gen_len),
                "teacher_topk_indices": topk_idx.cpu(),               # int64
                "teacher_topk_logprobs": topk_vals.cpu(),             # float32, log-probs (negative)
            })
            del logits, cont_logits, cont_logp, topk_vals, topk_idx
            print(f"  [{i+1}/{len(prompts)}] plen={plen} gen={gen_len}  elapsed={time.time()-t0:.0f}s",
                  flush=True)

    print(f"[save] {args.output}", flush=True)
    tmp = args.output + ".tmp"
    torch.save({
        "entries": cache_entries,
        "shard_idx": shard_idx,
        "seed": args.seed,
        "logprobs_k": args.logprobs_k,
        "max_new_tokens": args.max_new_tokens,
        "teacher_model": TEACHER_MODEL,
    }, tmp)
    os.replace(tmp, args.output)
    print(f"[done] {len(cache_entries)} prompts, shard {shard_idx}, K={args.logprobs_k}, "
          f"total {time.time()-t0:.0f}s, file={os.path.getsize(args.output)/1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
