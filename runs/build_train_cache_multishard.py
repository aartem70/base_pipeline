"""
Multi-shard replacement for cont_cache_600.pt.

Samples N prompts per shard across many distinct climbmix shards, so the
training distribution covers what the validator actually tests (new shard per
block). Output is in the same format train_kld_normbias.py expects:

    {sequences: list[LongTensor(L)],
     prompt_lens: list[int]=128,
     teacher_logits: list[BF16Tensor(L, V)]}

Usage:
    python runs/build_train_cache_multishard.py \
        --shards 30 --per_shard 20 \
        --output /ephemeral/cont_cache_600_multishard.pt
"""
import argparse
import gc
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"
CLIMBMIX_DATASET = "karpathy/climbmix-400b-shuffle"
CLIMBMIX_NUM_SHARDS = 6542


def pick_shards(n, seed):
    rng = random.Random(seed)
    # Deduplicate; rng.sample avoids collisions
    return rng.sample(range(CLIMBMIX_NUM_SHARDS), n)


def load_shard(shard_idx):
    from datasets import load_dataset
    shard_file = f"shard_{shard_idx:05d}.parquet"
    ds = load_dataset(CLIMBMIX_DATASET, data_files=shard_file, split="train")
    return ds


def sample_texts_from_shard(ds, n, rng, min_chars=512, max_chars=10000):
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    texts = []
    for idx in indices:
        t = ds[idx].get("text", "")
        if not t or len(t) < min_chars:
            continue
        if len(t) > max_chars:
            t = t[:max_chars]
            sp = t.rfind(" ")
            if sp > max_chars // 2:
                t = t[:sp]
        texts.append(t)
        if len(texts) >= n:
            break
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=30, help="number of distinct shards")
    ap.add_argument("--per_shard", type=int, default=20, help="prompts per shard")
    ap.add_argument("--shard_seed", type=int, default=1234, help="seed for shard selection")
    ap.add_argument("--prompt_len", type=int, default=128)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8, help="batch size for teacher inference")
    ap.add_argument("--output", required=True)
    ap.add_argument("--save_every", type=int, default=60, help="checkpoint every N prompts")
    args = ap.parse_args()

    shards = pick_shards(args.shards, args.shard_seed)
    target_n = args.shards * args.per_shard
    print(f"[plan] {args.shards} shards × {args.per_shard} prompts = {target_n} total", flush=True)
    print(f"[plan] shards: {shards[:10]}...{shards[-5:]}", flush=True)
    print(f"[plan] prompt_len={args.prompt_len} max_new_tokens={args.max_new_tokens}", flush=True)

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

    all_sequences = []
    all_prompt_lens = []
    all_teacher_logits = []
    ckpt_path = args.output + ".ckpt"

    # Resume
    if os.path.isfile(ckpt_path):
        print(f"[resume] loading {ckpt_path}", flush=True)
        prev = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        all_sequences = prev["sequences"]
        all_prompt_lens = prev["prompt_lens"]
        all_teacher_logits = prev["teacher_logits"]
        print(f"  resumed with {len(all_sequences)} existing prompts", flush=True)

    done_count = len(all_sequences)
    shards_done = done_count // args.per_shard
    t_start = time.time()

    for shard_i, shard_idx in enumerate(shards[shards_done:], start=shards_done):
        print(f"\n[shard {shard_i+1}/{len(shards)}] idx={shard_idx}", flush=True)
        t_sh = time.time()
        try:
            ds = load_shard(shard_idx)
        except Exception as e:
            print(f"  load failed: {e}; skipping", flush=True)
            continue
        rng = random.Random(args.shard_seed * 1000 + shard_idx)
        texts = sample_texts_from_shard(ds, args.per_shard * 3, rng)  # oversample
        print(f"  shard loaded in {time.time()-t_sh:.0f}s, {len(texts)} candidate texts", flush=True)

        # Pre-tokenize all valid prompts for this shard up front
        shard_prompts = []
        for text in texts:
            if len(shard_prompts) >= args.per_shard:
                break
            enc = tok(text, return_tensors="pt", truncation=True,
                      max_length=args.prompt_len + 4)
            ids = enc["input_ids"][0]
            if ids.shape[0] < args.prompt_len:
                continue
            shard_prompts.append(ids[:args.prompt_len])

        kept = 0
        with torch.no_grad():
            for bs_start in range(0, len(shard_prompts), args.batch_size):
                batch = shard_prompts[bs_start:bs_start + args.batch_size]
                prompt_batch = torch.stack(batch, dim=0).to(teacher.device)  # [B, plen]
                attn = torch.ones_like(prompt_batch)

                # Generate continuation for whole batch in one call
                out_ids = teacher.generate(
                    prompt_batch, attention_mask=attn,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False, use_cache=True,
                    pad_token_id=tok.eos_token_id,
                )
                # Forward pass on full batch for logits (attend everywhere)
                logits_batch = teacher(out_ids).logits.to(torch.bfloat16).cpu()

                B = out_ids.shape[0]
                for b in range(B):
                    full_seq = out_ids[b].cpu()
                    L = full_seq.shape[0]
                    # Drop samples with tiny continuations
                    if L - args.prompt_len < 10:
                        continue
                    all_sequences.append(full_seq)
                    all_prompt_lens.append(args.prompt_len)
                    all_teacher_logits.append(logits_batch[b, :L, :].clone())
                    kept += 1

                del logits_batch, out_ids

                el = time.time() - t_start
                done = len(all_sequences) - done_count
                rate = done / el if el > 0 else 0
                remain = target_n - len(all_sequences)
                eta = (remain / rate / 60) if rate > 0 else -1
                print(f"    [{len(all_sequences)}/{target_n}] shard_kept={kept}/{args.per_shard} "
                      f"rate={rate:.2f}/s  eta={eta:.1f}min", flush=True)

        del ds
        gc.collect()
        torch.cuda.empty_cache()

        # Periodic save
        if (shard_i + 1) % max(1, args.save_every // args.per_shard) == 0:
            print(f"  [save-ckpt] {ckpt_path}", flush=True)
            torch.save({
                "sequences": all_sequences,
                "prompt_lens": all_prompt_lens,
                "teacher_logits": all_teacher_logits,
            }, ckpt_path)

    print(f"\n[save final] {args.output}", flush=True)
    tmp = args.output + ".tmp"
    torch.save({
        "sequences": all_sequences,
        "prompt_lens": all_prompt_lens,
        "teacher_logits": all_teacher_logits,
    }, tmp)
    os.replace(tmp, args.output)
    if os.path.isfile(ckpt_path):
        os.remove(ckpt_path)
    print(f"[done] {len(all_sequences)} prompts across {len(shards)} shards in "
          f"{(time.time()-t_start)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
