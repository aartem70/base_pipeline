"""
Forward-KL distillation with PEFT adapters (LoRA variants never tested on distil).

Mirrors train_kld_normbias.py but swaps the norm/bias subspace for a PEFT-wrapped
adapter. After training, merge adapter into base model and save a standalone
checkpoint so eval_bootstrap.py can load it directly.

Supported --method values:
    rslora  — LoRA with rank-stabilized scaling (use_rslora=True)
    ia3     — IA^3 multiplicative vectors on K, V, down_proj
    oft     — Orthogonal Fine-Tuning (block-diagonal rotations)
    vera    — Vector-based Random Adaptation (shared random + tiny vectors)

Usage:
    python runs/train_kld_peft.py --method rslora --student kharchevnykov/distil \
        --cache /ephemeral/cont_cache_600.pt \
        --output_dir /ephemeral/runs/exp_10.1_rslora --lr 3e-6 --max_steps 500
"""
import argparse
import gc
import json
import os
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
import peft


ATTN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
IA3_TARGETS = ["k_proj", "v_proj", "down_proj"]
IA3_FF_TARGETS = ["down_proj"]  # feedforward-only, rescaled


def build_peft_config(method: str, rank: int, alpha: int):
    if method == "rslora":
        return peft.LoraConfig(
            r=rank, lora_alpha=alpha, target_modules=ATTN_TARGETS,
            lora_dropout=0.0, bias="none", use_rslora=True,
            task_type="CAUSAL_LM",
        )
    if method == "ia3":
        return peft.IA3Config(
            target_modules=IA3_TARGETS,
            feedforward_modules=IA3_FF_TARGETS,
            task_type="CAUSAL_LM",
        )
    if method == "oft":
        # OFT requires r XOR oft_block_size (product == in_features). Use r only.
        return peft.OFTConfig(
            r=rank, oft_block_size=0, target_modules=ATTN_TARGETS,
            module_dropout=0.0, task_type="CAUSAL_LM",
        )
    if method == "vera":
        return peft.VeraConfig(
            r=rank, target_modules=ATTN_TARGETS,
            vera_dropout=0.0, task_type="CAUSAL_LM",
        )
    raise ValueError(f"unknown method {method}")


def pad_batch(seqs, pad_id=0):
    max_len = max(s.shape[0] for s in seqs)
    ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        L = s.shape[0]
        ids[i, :L] = s
        attn[i, :L] = 1
    return ids, attn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--method", required=True, choices=["rslora", "ia3", "oft", "vera"])
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"[load cache mmap] {args.cache}", flush=True)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False, mmap=True)
    seqs = cache["sequences"]
    plens = cache["prompt_lens"]
    t_logits_list = cache["teacher_logits"]
    n = len(seqs)
    print(f"  n={n}", flush=True)

    print(f"[load student] {args.student}", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=dtype, trust_remote_code=True,
    ).to(args.device)
    # Freeze base; PEFT will add trainable adapter params
    for p in base.parameters():
        p.requires_grad_(False)

    cfg = build_peft_config(args.method, args.rank, args.alpha)
    print(f"[peft config] {cfg.__class__.__name__}: {cfg}", flush=True)
    model = peft.get_peft_model(base, cfg)
    model.print_trainable_parameters()

    model.train()
    try:
        model.enable_input_require_grads()
    except Exception:
        pass
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    print(f"  loaded in {time.time()-t0:.1f}s, "
          f"VRAM={torch.cuda.memory_allocated(args.device)/1e9:.1f}GB", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_steps, max(args.max_steps, args.warmup_steps + 1)
    )

    out_abs = os.path.abspath(args.output_dir)
    if os.path.isdir(out_abs):
        shutil.rmtree(out_abs)
    os.makedirs(out_abs, exist_ok=True)
    with open(os.path.join(out_abs, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    metrics_path = os.path.join(out_abs, "train_metrics.jsonl")

    print(f"[train] method={args.method} max_steps={args.max_steps} lr={args.lr} batch={args.batch_size}",
          flush=True)
    idx_all = list(range(n))
    rng = random.Random(args.seed)
    rng.shuffle(idx_all)
    ptr = 0

    best_loss = float("inf")
    t_train = time.time()
    losses = []

    for step in range(1, args.max_steps + 1):
        batch_idx = []
        while len(batch_idx) < args.batch_size:
            if ptr >= n:
                rng.shuffle(idx_all)
                ptr = 0
            batch_idx.append(idx_all[ptr])
            ptr += 1

        batch_seqs = [seqs[i][:args.max_seq_len] for i in batch_idx]
        batch_plens = [min(plens[i], args.max_seq_len - 1) for i in batch_idx]
        batch_tlogits = [t_logits_list[i][:args.max_seq_len].clone() for i in batch_idx]

        input_ids, attn = pad_batch(batch_seqs, pad_id=tokenizer.pad_token_id or 0)
        input_ids = input_ids.to(args.device)
        attn = attn.to(args.device)

        optimizer.zero_grad()
        out = model(input_ids, attention_mask=attn)
        s_logits = out.logits

        B, L, V = s_logits.shape
        total_kl = 0.0
        total_tokens = 0
        for b in range(B):
            seq_len = batch_seqs[b].shape[0]
            plen = batch_plens[b]
            t = batch_tlogits[b].to(args.device).float()
            s = s_logits[b, :seq_len, :].float()
            start = plen - 1
            end = seq_len - 1
            if end <= start:
                continue
            t_slice = t[start:end]
            s_slice = s[start:end]
            t_lp = F.log_softmax(t_slice, dim=-1)
            s_lp = F.log_softmax(s_slice, dim=-1)
            kl = F.kl_div(s_lp, t_lp, log_target=True, reduction="none").sum(dim=-1).sum()
            total_kl = total_kl + kl
            total_tokens += (end - start)

        if total_tokens == 0:
            continue
        loss = total_kl / total_tokens

        if not torch.isfinite(loss):
            print(f"[non-finite loss at step {step}], stopping.", flush=True)
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        scheduler.step()

        l = loss.item()
        losses.append(l)
        lr = scheduler.get_last_lr()[0]

        if step == 1 or step % 50 == 0 or step == args.max_steps:
            rate = step / (time.time() - t_train)
            recent = sum(losses[-20:]) / min(20, len(losses))
            print(f"step {step} | KL={l:.4f} (last20={recent:.4f}) | lr={lr:.2e} | "
                  f"{rate:.2f} step/s", flush=True)

        with open(metrics_path, "a") as f:
            f.write(json.dumps({"step": step, "loss": l, "lr": lr}) + "\n")

        if l < best_loss:
            best_loss = l

        del out, s_logits, loss

    # Merge adapter into base model and save as standalone
    print("[merge] merging PEFT adapter into base and saving standalone model", flush=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(out_abs, safe_serialization=True)
    tokenizer.save_pretrained(out_abs)
    with open(os.path.join(out_abs, "train_state.json"), "w") as f:
        json.dump({"method": args.method, "rank": args.rank, "alpha": args.alpha,
                   "seed": args.seed, "max_steps": args.max_steps,
                   "best_train_kl": best_loss}, f, indent=2)
    print(f"[save] {out_abs}", flush=True)
    print(f"[done] best_train_kl={best_loss:.4f}, total={time.time()-t_train:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
