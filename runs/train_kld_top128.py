"""
Norm/bias-only top-K renormalized KL distillation on cached teacher logits.

Matches the validator's loss exactly:
  - Teacher: softmax over its own top-K logits at each position.
  - Student: full-vocab log_softmax, gather at teacher's top-K indices,
    then renormalize over the same K support.
  - KL is computed between two K-support distributions, not full vocab.

Only norm/bias parameters are trained; training runs in fp32 (required for
large-magnitude norm weights where bf16 Adam steps round to zero).
"""
import argparse
import gc
import json
import math
import os
import random
import shutil
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup


def is_norm_or_bias(name, param):
    n = name.lower()
    if n.endswith(".bias"):
        return True
    if any(k in n for k in (
        "layernorm", "layer_norm", "rmsnorm", "rms_norm", "norm.weight",
        "ln_f", "ln_1", "ln_2", ".ln.", "norm_f",
    )):
        return True
    return False


def pad_batch(seqs, pad_id=0):
    max_len = max(s.shape[0] for s in seqs)
    out_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        L = s.shape[0]
        out_ids[i, :L] = s
        attn[i, :L] = 1
    return out_ids, attn


def save_ckpt(student, tokenizer, args, step, name, extra=None):
    d = os.path.join(args.output_dir, name)
    os.makedirs(d, exist_ok=True)
    # Save at the model's current dtype (no cast). If training ran in fp32,
    # the saved file preserves fp32 precision; bf16 training saves bf16.
    student.save_pretrained(d, safe_serialization=True)
    tokenizer.save_pretrained(d)
    state = {"global_step": step, "name": name, "seed": args.seed}
    if extra:
        state.update(extra)
    with open(os.path.join(d, "train_state.json"), "w") as f:
        json.dump(state, f, indent=2)
    print(f"[save] {d}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--cache", required=True,
                    help="full cont_cache_600.pt with teacher_logits")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup_steps", type=int, default=20)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--kl_continuation_only", action="store_true", default=True,
                    help="mask KL to continuation positions (skip prompt)")
    ap.add_argument("--topk", type=int, default=128,
                    help="Top-K support for renormalized KL (validator uses 128).")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"[load cache mmap] {args.cache}", flush=True)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False, mmap=True)
    seqs = cache["sequences"]
    plens = cache["prompt_lens"]
    t_logits_list = cache["teacher_logits"]  # list of bf16 tensors, mmap'd
    n = len(seqs)
    print(f"  n={n} (mmap: teacher_logits paged in per batch)", flush=True)

    print(f"[load student] {args.student}", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    student = AutoModelForCausalLM.from_pretrained(
        args.student,
        dtype=dtype,
        trust_remote_code=True,
    ).to(args.device).to(dtype)  # force dtype even if config says otherwise
    print(f"  actual loaded dtype: {next(student.parameters()).dtype}", flush=True)

    n_total = sum(p.numel() for p in student.parameters())
    n_trainable = 0
    for name, p in student.named_parameters():
        if is_norm_or_bias(name, p):
            p.requires_grad_(True)
            n_trainable += p.numel()
        else:
            p.requires_grad_(False)
    print(f"  total params = {n_total/1e9:.3f}B; trainable = {n_trainable/1e6:.2f}M "
          f"({n_trainable/n_total*100:.2f}%)", flush=True)

    student.train()
    try:
        student.enable_input_require_grads()
    except Exception:
        pass
    try:
        student.gradient_checkpointing_enable()
    except Exception:
        pass

    print(f"  loaded in {time.time() - t0:.1f}s, "
          f"VRAM={torch.cuda.memory_allocated(args.device) / 1e9:.1f}GB", flush=True)

    params = [p for p in student.parameters() if p.requires_grad]
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

    print(f"[train] max_steps={args.max_steps}, lr={args.lr}, batch={args.batch_size}",
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
        # teacher logits shape per sample: [L, V]
        batch_tlogits = [t_logits_list[i][:args.max_seq_len].clone()  # force copy from mmap
                         for i in batch_idx]

        input_ids, attn = pad_batch(batch_seqs, pad_id=tokenizer.pad_token_id or 0)
        input_ids = input_ids.to(args.device)
        attn = attn.to(args.device)

        optimizer.zero_grad()
        out = student(input_ids, attention_mask=attn)
        s_logits = out.logits  # [B, L, V]

        # KL loss, continuation-only, masked to real sequence positions
        B, L, V = s_logits.shape
        total_kl = 0.0
        total_tokens = 0
        for b in range(B):
            seq_len = batch_seqs[b].shape[0]
            plen = batch_plens[b]
            t = batch_tlogits[b].to(args.device).float()  # [seq_len, V]
            s = s_logits[b, :seq_len, :].float()
            # predict token t from position t-1: teacher_logits[t-1] is the dist to predict
            #   teacher_logits are stored unshifted: t[t] = dist at position t
            # For match:
            #   student_logits[b, t, :] = student's dist at position t (predicting t+1)
            # Both should match at positions [plen-1 ... seq_len-2] (inclusive)
            # so the pair is (s_logits[plen-1:seq_len-1], t_logits[plen-1:seq_len-1])
            start = plen - 1 if args.kl_continuation_only else 0
            end = seq_len - 1
            if end <= start:
                continue
            t_slice = t[start:end]  # [k, V]
            s_slice = s[start:end]  # [k, V]
            # Validator-style sparse top-K KL:
            # Teacher = softmax over its own top-K logits.
            # Student = full-vocab log_softmax, gather at teacher top-K indices,
            # then renormalize over the same K support.
            t_topk_vals, t_topk_idx = t_slice.topk(args.topk, dim=-1)
            t_lp_k = F.log_softmax(t_topk_vals, dim=-1)
            s_lp_full = F.log_softmax(s_slice, dim=-1)
            s_lp_k = s_lp_full.gather(-1, t_topk_idx)
            s_lp_k = s_lp_k - s_lp_k.logsumexp(-1, keepdim=True)
            kl = F.kl_div(s_lp_k, t_lp_k, log_target=True, reduction="none").sum(dim=-1).sum()
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

        if step == 1 or step % 5 == 0 or step == args.max_steps:
            rate = step / (time.time() - t_train)
            recent = sum(losses[-10:]) / min(10, len(losses))
            print(f"step {step} | KL={l:.4f} (last10={recent:.4f}) | lr={lr:.2e} | "
                  f"{rate:.2f} step/s", flush=True)

        with open(metrics_path, "a") as f:
            f.write(json.dumps({"step": step, "loss": l, "lr": lr}) + "\n")

        if l < best_loss:
            best_loss = l

        if step % args.save_every == 0:
            save_ckpt(student, tokenizer, args, step, f"step_{step}")

        del out, s_logits, loss

    save_ckpt(student, tokenizer, args, args.max_steps, "final")
    print(f"[done] best_train_kl={best_loss:.4f}, total={time.time() - t_train:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
