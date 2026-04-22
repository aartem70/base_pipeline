"""
Phase 2 cache builder via SGLang — single-shot generation + top-K logprobs.

SGLang's /generate endpoint can return per-token top-K logprobs in the same
call as the greedy continuation. This eliminates the slow HF forward pass.

Storage format matches the prod cache (top-128 sparse):
    full_ids               : [1, prompt_len + gen_len] token ids (int64)
    prompt_len             : int
    gen_len                : int
    teacher_topk_indices   : [1, gen_len, K] int64
    teacher_topk_logprobs  : [1, gen_len, K] float32  (log-probs, negative)

Prereqs:
  - SGLang server running on http://127.0.0.1:30000
  - Server launched with --return-token-ids implicit (default in modern sglang).

Usage:
    python runs/build_train_cache_sglang.py --seed 7 --n 800 \\
        --output /root/base_pipeline/caches/train_continuations/cache.pt
"""

import argparse, json, os, random, time
import urllib.request, urllib.error
import concurrent.futures

import torch
from transformers import AutoTokenizer

TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"
CLIMBMIX_DATASET = "karpathy/climbmix-400b-shuffle"
CLIMBMIX_NUM_SHARDS = 6542
MAX_NEW_TOKENS = 512
LOGPROBS_K = 128
SGLANG_URL = "http://127.0.0.1:30000"


def sample_prompts(n, seed, min_chars=500, max_chars=10000):
    from datasets import load_dataset
    rng = random.Random(seed)
    shard_idx = rng.randint(0, CLIMBMIX_NUM_SHARDS - 1)
    shard_file = f"shard_{shard_idx:05d}.parquet"
    print(f"[dataset] seed={seed} → shard={shard_idx}", flush=True)
    ds = load_dataset(CLIMBMIX_DATASET, data_files=shard_file, split="train")
    indices = list(range(len(ds))); rng.shuffle(indices)
    out = []
    for idx in indices:
        text = ds[idx].get("text", "")
        if not text or len(text) < min_chars: continue
        if len(text) > max_chars:
            text = text[:max_chars]
            ls = text.rfind(" ")
            if ls > max_chars // 2: text = text[:ls]
        out.append(text)
        if len(out) >= n: break
    return out, shard_idx


def sglang_generate_with_logprobs(prompt: str, max_new_tokens: int, logprobs_k: int,
                                   base_url: str = SGLANG_URL, timeout: int = 600):
    """One-shot: SGLang greedy gen + per-token top-K input/output logprobs.

    Returns dict with:
        text          : generated text
        output_ids    : list of int (generated token ids)
        topk_indices  : list of [K] int (per generated position)
        topk_logprobs : list of [K] float (per generated position)
    """
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "return_logprob": True,
        "top_logprobs_num": logprobs_k,
        "logprob_start_len": -1,   # -1 = only continuation positions, not the prompt
    }
    req = urllib.request.Request(
        f"{base_url}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode())

    text = result.get("text", "")
    meta = result.get("meta_info", {})
    # output_top_logprobs format (sglang): list[list[(logprob, token_id, decoded_str)]]
    # one entry per generated position; each entry is K (logprob, token_id, str) tuples
    out_lp = meta.get("output_top_logprobs", [])
    out_ids_lp = meta.get("output_token_logprobs", [])

    return {
        "text": text,
        "output_ids": [tup[1] for tup in out_ids_lp] if out_ids_lp else [],
        "topk_indices": [[tup[1] for tup in pos] for pos in out_lp],
        "topk_logprobs": [[tup[0] for tup in pos] for pos in out_lp],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--logprobs_k", type=int, default=LOGPROBS_K)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # health check
    print(f"[sglang] probing {SGLANG_URL}/health", flush=True)
    try:
        urllib.request.urlopen(f"{SGLANG_URL}/health", timeout=5).read()
    except Exception as e:
        raise SystemExit(f"SGLang server not reachable: {e}")
    print("  ok", flush=True)

    print(f"[tokenizer] {TEACHER_MODEL}", flush=True)
    tok = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)

    prompts, shard_idx = sample_prompts(args.n, args.seed)
    if len(prompts) < args.n:
        print(f"[warn] only got {len(prompts)}/{args.n} prompts; continuing", flush=True)

    print(f"[gen] {len(prompts)} prompts × max_new={args.max_new_tokens} via sglang "
          f"(concurrency={args.concurrency}, K={args.logprobs_k})", flush=True)

    results = [None] * len(prompts)

    def _one(idx_text):
        idx, text = idx_text
        try:
            r = sglang_generate_with_logprobs(text, args.max_new_tokens, args.logprobs_k)
            return idx, text, r
        except Exception as e:
            print(f"  [error] prompt {idx}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            return idx, text, None

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(_one, (i, p)) for i, p in enumerate(prompts)]
        done = 0
        for f in concurrent.futures.as_completed(futures):
            idx, text, r = f.result()
            results[idx] = (text, r)
            done += 1
            if done % 50 == 0 or done == len(prompts):
                rate = done / (time.time() - t0)
                eta = (len(prompts) - done) / max(rate, 1e-6)
                print(f"  [{done}/{len(prompts)}] rate={rate:.2f}/s ETA={eta:.0f}s", flush=True)
    print(f"[gen done] {time.time()-t0:.1f}s", flush=True)

    # Build entries
    print("[assemble] tokenizing prompts and packing entries", flush=True)
    entries = []
    for idx, item in enumerate(results):
        if item is None:
            continue
        text, r = item
        if r is None or not r["topk_indices"]:
            continue
        prompt_ids = tok(text, return_tensors="pt", truncation=False).input_ids
        plen = prompt_ids.shape[1]
        gen_ids = r["output_ids"]
        if not gen_ids:
            continue
        full_ids = torch.cat([prompt_ids, torch.tensor(gen_ids, dtype=prompt_ids.dtype).unsqueeze(0)], dim=1)

        topk_indices = torch.tensor(r["topk_indices"], dtype=torch.int64).unsqueeze(0)   # [1, gen_len, K]
        topk_logprobs = torch.tensor(r["topk_logprobs"], dtype=torch.float32).unsqueeze(0)
        entries.append({
            "full_ids": full_ids,
            "prompt_len": int(plen),
            "gen_len": int(topk_indices.shape[1]),
            "teacher_topk_indices": topk_indices,
            "teacher_topk_logprobs": topk_logprobs,
        })

    print(f"[save] {len(entries)} entries → {args.output}", flush=True)
    tmp = args.output + ".tmp"
    torch.save({
        "entries": entries,
        "shard_idx": shard_idx, "seed": args.seed,
        "logprobs_k": args.logprobs_k,
        "max_new_tokens": args.max_new_tokens,
        "teacher_model": TEACHER_MODEL,
        "source": "sglang_single_shot",
    }, tmp)
    os.replace(tmp, args.output)
    print(f"[done] {os.path.getsize(args.output)/1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
