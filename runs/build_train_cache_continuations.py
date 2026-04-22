"""
Phase 2: Build a CONTINUATION training cache — teacher greedy-generates its own
continuations for training prompts, stores full_ids + prompt_len + gen_len +
teacher's top-128 indices/log-probs on the generated continuation positions.

Mirrors build_eval_cache.py exactly (same generation params, same storage
format) but for training: N is larger (default 5000), seed differs from eval.

Parallelizable: pass --num-workers W and --worker-id I to split the N prompts
across W workers, each producing cache_part_<I>.pt. Merge at end with
merge_train_cache.py (or just read all parts and concat).

Usage (single worker on tensor-parallel GPUs 0,1):
    CUDA_VISIBLE_DEVICES=0,1 python runs/build_train_cache_continuations.py \\
        --seed 7 --n 5000 \\
        --output /root/base_pipeline/caches/train_cache_continuations_5k.pt

Usage (4 parallel workers, 2 GPUs each = 8 GPUs):
    for i in 0 1 2 3; do
      gpus=$((i*2)),$((i*2+1))
      CUDA_VISIBLE_DEVICES=$gpus python runs/build_train_cache_continuations.py \\
          --seed 7 --n 5000 --num-workers 4 --worker-id $i \\
          --output /root/base_pipeline/caches/train_cont_part_${i}.pt &
    done
    wait
"""

import argparse, os, random, time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"
CLIMBMIX_DATASET = "karpathy/climbmix-400b-shuffle"
CLIMBMIX_NUM_SHARDS = 6542
MAX_NEW_TOKENS = 512
LOGPROBS_K = 128


def sample_prompts(n, seed, min_chars=500, max_chars=10000):
    from datasets import load_dataset
    rng = random.Random(seed)
    # Use a DIFFERENT shard than the eval (eval uses seed=42 → shard 5238).
    shard_idx = rng.randint(0, CLIMBMIX_NUM_SHARDS - 1)
    shard_file = f"shard_{shard_idx:05d}.parquet"
    print(f"[dataset] seed={seed} → train shard={shard_idx}", flush=True)
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
            ls = text.rfind(" ")
            if ls > max_chars // 2:
                text = text[:ls]
        prompts.append(text)
        if len(prompts) >= n:
            break
    return prompts, shard_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=5000, help="total prompts to cache")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--logprobs_k", type=int, default=LOGPROBS_K)
    ap.add_argument("--num-workers", type=int, default=1,
                    help="number of parallel workers total (split prompts across workers)")
    ap.add_argument("--worker-id", type=int, default=0,
                    help="this worker's index in 0..num_workers-1")
    args = ap.parse_args()

    prompts, shard_idx = sample_prompts(args.n, args.seed)
    if len(prompts) < args.n:
        print(f"[warn] only got {len(prompts)}/{args.n} prompts; proceeding", flush=True)

    # split for this worker
    my = [p for i, p in enumerate(prompts) if i % args.num_workers == args.worker_id]
    print(f"[worker {args.worker_id}/{args.num_workers}] {len(my)}/{len(prompts)} prompts", flush=True)

    print(f"[load teacher] {TEACHER_MODEL}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
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

    entries = []
    t0 = time.time()
    with torch.no_grad():
        for i, text in enumerate(my):
            ids = tok(text, return_tensors="pt", truncation=False).input_ids.to(teacher.device)
            plen = ids.shape[1]
            out_ids = teacher.generate(
                ids, max_new_tokens=args.max_new_tokens,
                do_sample=False, use_cache=True,
            )
            gen_len = out_ids.shape[1] - plen
            if gen_len == 0:
                continue
            logits = teacher(out_ids).logits.float()
            cont_logits = logits[:, plen - 1:-1, :]
            cont_logp = F.log_softmax(cont_logits, dim=-1)
            topk_vals, topk_idx = cont_logp.topk(args.logprobs_k, dim=-1)
            entries.append({
                "full_ids": out_ids.cpu(),
                "prompt_len": int(plen),
                "gen_len": int(gen_len),
                "teacher_topk_indices": topk_idx.cpu(),
                "teacher_topk_logprobs": topk_vals.cpu(),
            })
            del logits, cont_logits, cont_logp, topk_vals, topk_idx
            if (i + 1) % 25 == 0 or (i + 1) == len(my):
                rate = (i + 1) / (time.time() - t0)
                eta = (len(my) - (i + 1)) / max(rate, 1e-6)
                print(f"  [{i+1}/{len(my)}] plen={plen} gen={gen_len}  "
                      f"rate={rate:.2f}/s ETA={eta:.0f}s", flush=True)

    print(f"[save] {args.output}", flush=True)
    tmp = args.output + ".tmp"
    torch.save({
        "entries": entries,
        "shard_idx": shard_idx,
        "seed": args.seed,
        "logprobs_k": args.logprobs_k,
        "max_new_tokens": args.max_new_tokens,
        "teacher_model": TEACHER_MODEL,
        "num_workers": args.num_workers,
        "worker_id": args.worker_id,
    }, tmp)
    os.replace(tmp, args.output)
    print(f"[done] {len(entries)} entries, {time.time()-t0:.0f}s, "
          f"file={os.path.getsize(args.output)/1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
