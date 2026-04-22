"""
Build a teacher_cache_60-style eval cache for a given climbmix seed/shard.

Mirrors evaluate.py's cache-building path exactly (VALIDATOR_MAX_NEW_TOKENS=512,
greedy, full-vocab continuation logits).

Usage:
    python runs/build_eval_cache.py --seed 7 --n 60 \
        --output /ephemeral/teacher_cache_60_seed7.pt
"""
import argparse
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"
CLIMBMIX_DATASET = "karpathy/climbmix-400b-shuffle"
CLIMBMIX_NUM_SHARDS = 6542
VALIDATOR_MAX_NEW_TOKENS = 512


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

    full_sequences, teacher_logits_list, prompt_lens = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i, text in enumerate(prompts):
            ids = tok(text, return_tensors="pt", truncation=False).input_ids.to(teacher.device)
            plen = ids.shape[1]
            out_ids = teacher.generate(
                ids, max_new_tokens=args.max_new_tokens,
                do_sample=False, use_cache=True,
            )
            gen_len = out_ids.shape[1] - plen
            logits = teacher(out_ids).logits.float()
            cont = logits[:, plen - 1:-1, :].cpu()
            full_sequences.append(out_ids.cpu())
            teacher_logits_list.append(cont)
            prompt_lens.append(plen)
            del logits, cont
            print(f"  [{i+1}/{len(prompts)}] plen={plen} gen={gen_len}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"[save] {args.output}", flush=True)
    tmp = args.output + ".tmp"
    torch.save({
        "full_sequences": full_sequences,
        "teacher_logits": teacher_logits_list,
        "prompt_lens": prompt_lens,
        "shard_idx": shard_idx,
        "seed": args.seed,
    }, tmp)
    os.replace(tmp, args.output)
    print(f"[done] {len(full_sequences)} prompts, shard {shard_idx}, "
          f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
