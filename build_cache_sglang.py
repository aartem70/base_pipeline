#!/usr/bin/env python3
"""
Fast cache builder using SGLang for teacher generation + HF for full-vocab logits.

Architecture:
  1. SGLang server (separate venv) generates continuation TEXT fast (~2-5s/prompt)
  2. This script (main venv) collects generated text, then runs HF forward pass
     to extract full-vocab teacher logits (~0.5-1s/prompt)
  3. Saves cache in the exact format train.py expects

Prerequisites:
  - SGLang server running: bash setup_sglang.sh serve 0
  - Main venv active with transformers>=5, torch, datasets

Usage:
  # Terminal 1: Start SGLang server
  bash setup_sglang.sh serve 0

  # Terminal 2: Build cache (main venv)
  python build_cache_sglang.py --n 5000 --output /ephemeral/cache_5000.pt

  # Use cache for training
  python train.py --student Qwen/Qwen3.5-4B --continuation \
      --cache_continuations /ephemeral/cache_5000.pt ...
"""

import argparse
import concurrent.futures
import gc
import json
import logging
import os
import random
import time
import urllib.request
import urllib.error

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
CLIMBMIX_DATASET = "karpathy/climbmix-400b-shuffle"
CLIMBMIX_NUM_SHARDS = 6542
DEFAULT_TEACHER = "Qwen/Qwen3.5-35B-A3B"


# ---------------------------------------------------------------------------
# Prompt sampler (same as train.py)
# ---------------------------------------------------------------------------
class RandomPromptSampler:
    def __init__(self, seed=42, min_chars=512, max_chars=10000):
        self._rng = random.Random(seed)
        self._min_chars = min_chars
        self._max_chars = max_chars
        self._current_shard = None
        self._current_indices = []
        self._index_pos = 0
        self._total = 0

    def _load_shard(self):
        from datasets import load_dataset
        shard_idx = self._rng.randint(0, CLIMBMIX_NUM_SHARDS - 1)
        shard_file = f"shard_{shard_idx:05d}.parquet"
        log.info(f"Loading shard {shard_idx}/{CLIMBMIX_NUM_SHARDS}...")
        ds = load_dataset(CLIMBMIX_DATASET, data_files=shard_file, split="train")
        indices = list(range(len(ds)))
        self._rng.shuffle(indices)
        self._current_shard = ds
        self._current_indices = indices
        self._index_pos = 0
        log.info(f"  Shard loaded: {len(ds)} rows")

    def sample(self, n):
        texts = []
        while len(texts) < n:
            if self._current_shard is None or self._index_pos >= len(self._current_indices):
                self._load_shard()
            while len(texts) < n and self._index_pos < len(self._current_indices):
                idx = self._current_indices[self._index_pos]
                self._index_pos += 1
                text = self._current_shard[idx].get("text", "")
                if not text or len(text) < self._min_chars:
                    continue
                if len(text) > self._max_chars:
                    text = text[:self._max_chars]
                    last_space = text.rfind(' ')
                    if last_space > self._max_chars // 2:
                        text = text[:last_space]
                texts.append(text)
            if len(texts) < n:
                self._current_shard = None
        self._total += len(texts)
        return texts


# ---------------------------------------------------------------------------
# SGLang client
# ---------------------------------------------------------------------------
def sglang_health_check(base_url):
    """Check if SGLang server is running."""
    try:
        req = urllib.request.Request(f"{base_url}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def sglang_generate(base_url, prompt_tokens, max_new_tokens, tokenizer,
                    temperature=0.0, top_p=1.0, seed=None):
    """Generate continuation using SGLang server. Returns list of generated token IDs."""
    sampling_params = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    # Note: SGLang 0.5.10 doesn't support seed in sampling_params. Diversity
    # still comes from server-internal randomness per request.
    payload = {
        "input_ids": prompt_tokens,
        "sampling_params": sampling_params,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    # SGLang returns generated text - tokenize it to get IDs
    generated_text = result.get("text", "")
    if not generated_text:
        return []
    gen_ids = tokenizer.encode(generated_text, add_special_tokens=False)
    return gen_ids


def sglang_generate_batch(base_url, batch_prompt_tokens, max_new_tokens, tokenizer):
    """Generate continuations for a batch using SGLang. Returns list of generated token ID lists."""
    # Use the batch endpoint for efficiency
    payload = []
    for prompt_tokens in batch_prompt_tokens:
        payload.append({
            "input_ids": prompt_tokens,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
            },
        })
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        results = json.loads(resp.read().decode("utf-8"))

    # Handle both single and batch responses
    if isinstance(results, dict):
        results = [results]

    gen_ids_list = []
    for result in results:
        text = result.get("text", "")
        if text:
            gen_ids_list.append(tokenizer.encode(text, add_special_tokens=False))
        else:
            gen_ids_list.append([])
    return gen_ids_list


# ---------------------------------------------------------------------------
# Phase 1: Generate continuations via SGLang
# ---------------------------------------------------------------------------
def generate_continuations(args, tokenizer):
    """Use SGLang server to generate continuations. Returns list of (prompt_ids, full_ids) pairs."""
    base_url = args.sglang_url
    log.info(f"Connecting to SGLang server at {base_url}...")
    if not sglang_health_check(base_url):
        raise RuntimeError(
            f"SGLang server not reachable at {base_url}. "
            f"Start it first: bash setup_sglang.sh serve 0"
        )
    log.info("  SGLang server is healthy.")

    sampler = RandomPromptSampler(seed=args.seed, min_chars=512, max_chars=10000)
    results = []  # list of (prompt_ids_tensor, full_seq_tensor)
    n_generated = 0
    n_skipped = 0
    t_start = time.time()

    while n_generated < args.n:
        # Sample a batch of texts
        batch_size = min(args.gen_batch, args.n - n_generated)
        texts = sampler.sample(batch_size * 2)  # oversample to handle skips

        # Tokenize and truncate to prompt_len
        batch_prompt_tokens = []
        batch_prompt_tensors = []
        for text in texts:
            if len(batch_prompt_tokens) >= batch_size:
                break
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_seq_len)
            ids = enc["input_ids"][0]
            if ids.shape[0] < args.prompt_len + 10:
                n_skipped += 1
                continue
            prompt = ids[:args.prompt_len]
            batch_prompt_tokens.append(prompt.tolist())
            batch_prompt_tensors.append(prompt)

        if not batch_prompt_tokens:
            continue

        # Concurrent SGLang requests — SGLang does continuous batching server-side.
        concurrency = max(1, args.concurrency)
        def _one(prompt_tokens):
            try:
                gen_ids = sglang_generate(
                    base_url, prompt_tokens, args.max_new_tokens, tokenizer,
                    temperature=args.temperature, top_p=args.top_p, seed=None,
                )
                return gen_ids
            except Exception as e:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {
                ex.submit(_one, pt): pt_tensor
                for pt, pt_tensor in zip(batch_prompt_tokens, batch_prompt_tensors)
            }
            for fut in concurrent.futures.as_completed(futures):
                if n_generated >= args.n:
                    break
                prompt_tensor = futures[fut]
                gen_ids = fut.result()
                if gen_ids is None or len(gen_ids) < 10:
                    n_skipped += 1
                    continue
                gen_tensor = torch.tensor(gen_ids, dtype=prompt_tensor.dtype)
                full_seq = torch.cat([prompt_tensor, gen_tensor])
                prompt_len = prompt_tensor.shape[0]
                results.append((prompt_len, full_seq))
                n_generated += 1
                if n_generated % 25 == 0 or n_generated == args.n:
                    elapsed = time.time() - t_start
                    rate = n_generated / elapsed if elapsed > 0 else 0
                    eta = (args.n - n_generated) / rate if rate > 0 else 0
                    log.info(
                        f"  Generated {n_generated}/{args.n} "
                        f"({rate:.2f}/s, ETA {eta / 60:.1f}min, skipped {n_skipped})"
                    )

    elapsed = time.time() - t_start
    log.info(f"Phase 1 done: {n_generated} continuations in {elapsed / 60:.1f}min")
    return results


# ---------------------------------------------------------------------------
# Phase 2: HF forward pass for full-vocab logits
# ---------------------------------------------------------------------------
def extract_logits(args, results):
    """Load HF teacher and run forward pass to get full-vocab logits."""
    log.info(f"Loading HF teacher for logit extraction: {args.teacher}")
    device = torch.device(f"cuda:{args.logit_gpu}")

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    teacher.eval()
    log.info(f"  Teacher loaded on GPU {args.logit_gpu}")

    all_sequences = []
    all_prompt_lens = []
    all_teacher_logits = []

    t_start = time.time()
    with torch.no_grad():
        for i, (prompt_len, full_seq) in enumerate(results):
            full_input = full_seq.unsqueeze(0).to(device)
            out = teacher(full_input)
            t_logits = out.logits[0].cpu()  # [seq_len, vocab_size]
            del out

            all_sequences.append(full_seq.cpu())
            all_prompt_lens.append(prompt_len)
            all_teacher_logits.append(t_logits)

            if (i + 1) % 50 == 0 or i == len(results) - 1:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta = (len(results) - i - 1) / rate if rate > 0 else 0
                log.info(
                    f"  Logits [{i + 1}/{len(results)}] "
                    f"({rate:.1f}/s, ETA {eta / 60:.0f}min)"
                )

            # Periodic memory cleanup
            if (i + 1) % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    log.info(f"Phase 2 done: {len(results)} logit extractions in {elapsed / 60:.1f}min")

    return {
        "sequences": all_sequences,
        "prompt_lens": all_prompt_lens,
        "teacher_logits": all_teacher_logits,
    }


# ---------------------------------------------------------------------------
# Phase 3: Save + optional resume
# ---------------------------------------------------------------------------
def save_cache(cache_data, output_path):
    """Save cache in train.py-compatible format."""
    log.info(f"Saving cache to {output_path}...")
    # Save to tmp first then rename (atomic)
    tmp_path = output_path + ".tmp"
    torch.save(cache_data, tmp_path)
    os.replace(tmp_path, output_path)

    size_gb = os.path.getsize(output_path) / (1024 ** 3)
    log.info(f"  Cache saved: {len(cache_data['sequences'])} samples, {size_gb:.1f} GB")


def load_partial(checkpoint_path):
    """Load partial results from a checkpoint (for resume)."""
    if os.path.isfile(checkpoint_path):
        log.info(f"Resuming from checkpoint: {checkpoint_path}")
        data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        log.info(f"  Loaded {data['n_generated']} previously generated continuations")
        return data
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build teacher continuation cache using SGLang")
    parser.add_argument("--n", type=int, default=5000, help="Number of samples to generate")
    parser.add_argument("--output", type=str, required=True, help="Output cache path (.pt)")
    parser.add_argument("--teacher", type=str, default=DEFAULT_TEACHER, help="Teacher model for HF logit extraction")
    parser.add_argument("--sglang_url", type=str, default="http://localhost:30000", help="SGLang server URL")
    parser.add_argument("--prompt_len", type=int, default=128, help="Prompt length in tokens")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max continuation tokens")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Max sequence length for tokenization")
    parser.add_argument("--logit_gpu", type=int, default=0, help="GPU for HF teacher logit extraction")
    parser.add_argument("--gen_batch", type=int, default=20, help="Prompts to sample per batch")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_every", type=int, default=500, help="Save checkpoint every N samples")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Concurrent SGLang requests in flight")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 = greedy)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top_p")
    parser.add_argument("--gen_seed", type=int, default=None,
                        help="Per-prompt generation seed base (seed=gen_seed+idx). None = unseeded")
    parser.add_argument("--eval_format", action="store_true",
                        help="Save cache in eval_bootstrap_region.py format: "
                             "full_sequences (B=1 batch dim), teacher_logits_full")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("SGLang Cache Builder")
    log.info("=" * 60)
    log.info(f"  Target: {args.n} samples")
    log.info(f"  Output: {args.output}")
    log.info(f"  Teacher: {args.teacher}")
    log.info(f"  SGLang: {args.sglang_url}")
    log.info(f"  prompt_len={args.prompt_len}, max_new_tokens={args.max_new_tokens}")
    log.info(f"  Logit GPU: {args.logit_gpu}")
    log.info("")

    # Set HF cache
    if os.path.isdir("/ephemeral"):
        os.environ["HF_HOME"] = "/ephemeral/hf_cache"

    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)

    # Check for resume checkpoint
    ckpt_path = args.output + ".ckpt"
    partial = load_partial(ckpt_path)

    if partial and partial["n_generated"] >= args.n:
        log.info("Checkpoint already has enough generations. Skipping to logit extraction.")
        phase1_results = list(zip(partial["prompt_lens"], partial["sequences"]))
    elif partial:
        # Resume phase 1
        existing = list(zip(partial["prompt_lens"], partial["sequences"]))
        n_existing = len(existing)
        log.info(f"Resuming from {n_existing} existing generations...")
        args_remaining = argparse.Namespace(**vars(args))
        args_remaining.n = args.n - n_existing
        new_results = generate_continuations(args_remaining, tokenizer)
        phase1_results = existing + new_results
    else:
        # Phase 1: Generate continuations via SGLang
        log.info("=" * 60)
        log.info("Phase 1: Generating continuations via SGLang")
        log.info("=" * 60)
        phase1_results = generate_continuations(args, tokenizer)

    # Save phase 1 checkpoint
    log.info("Saving phase 1 checkpoint...")
    torch.save({
        "sequences": [r[1] for r in phase1_results],
        "prompt_lens": [r[0] for r in phase1_results],
        "n_generated": len(phase1_results),
    }, ckpt_path)

    # Phase 2: HF forward pass for full-vocab logits
    log.info("")
    log.info("=" * 60)
    log.info("Phase 2: Extracting full-vocab teacher logits via HF")
    log.info("=" * 60)
    cache_data = extract_logits(args, phase1_results)

    # Phase 3: Save final cache
    log.info("")
    log.info("=" * 60)
    log.info("Phase 3: Saving cache")
    log.info("=" * 60)
    if args.eval_format:
        cache_data = {
            "full_sequences": [s.unsqueeze(0) for s in cache_data["sequences"]],
            "prompt_lens": cache_data["prompt_lens"],
            "teacher_logits_full": cache_data["teacher_logits"],
        }
    save_cache(cache_data, args.output)

    # Cleanup checkpoint
    if os.path.isfile(ckpt_path):
        os.remove(ckpt_path)
        log.info("  Checkpoint cleaned up.")

    log.info("")
    log.info("Done! Use with:")
    log.info(f"  python train.py --continuation --cache_continuations {args.output} ...")


if __name__ == "__main__":
    main()
