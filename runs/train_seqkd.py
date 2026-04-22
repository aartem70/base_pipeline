"""
Sequence-level distillation (pure CE on teacher-generated continuations).

Gradient signal differs from forward-KL: updates push mass onto the teacher's
sampled token, not match the teacher's full distribution. May sidestep the
sharp-minimum finding (D3) that rules out every weight-space perturbation
along the distil <-> Qwen direction under forward-KL.

Data: cont_cache_600.pt -- dict with keys
    sequences    : list[LongTensor of shape [L]]  (prompt + teacher_continuation)
    prompt_lens  : list[int]                      (length of prompt portion)
    teacher_logits : not used here

Usage:
    python train_seqkd.py \
        --student kharchevnykov/distil \
        --cache /ephemeral/cont_cache_600.pt \
        --output_dir /ephemeral/runs/exp_8.1_seqkd \
        --max_steps 200 --lr 5e-7 --batch_size 4 --seed 42
"""
import argparse
import copy
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


def pad_batch(seqs, pad_id=0):
    """Pad a list of 1-D LongTensors to the same length."""
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
    ap.add_argument("--cache", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--warmup_steps", type=int, default=10)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--save_best", action="store_true", default=False,
                    help="Save best_train_loss/ whenever loss improves")
    ap.add_argument("--save_best_after", type=int, default=20,
                    help="Only consider save_best after this many steps")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ce_over_continuation_only", action="store_true", default=True,
                    help="Mask CE to continuation tokens only (skip prompt)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    # --- cache ---
    print(f"[load cache] {args.cache}", flush=True)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    seqs = cache["sequences"]
    plens = cache["prompt_lens"]
    # Drop teacher_logits to save memory (we don't use them for CE)
    if "teacher_logits" in cache:
        del cache["teacher_logits"]
    gc.collect()
    n = len(seqs)
    print(f"  n={n}, avg_cont_len={sum(s.shape[0] - p for s, p in zip(seqs, plens)) / n:.1f}",
          flush=True)

    # --- model ---
    print(f"[load student] {args.student}", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    student = AutoModelForCausalLM.from_pretrained(
        args.student,
        dtype=dtype,
        trust_remote_code=True,
    ).to(args.device)
    student.train()
    try:
        student.gradient_checkpointing_enable()
        print("  gradient_checkpointing=on", flush=True)
    except Exception:
        pass
    n_params = sum(p.numel() for p in student.parameters())
    print(f"  loaded in {time.time() - t0:.1f}s, params={n_params/1e9:.2f}B, "
          f"VRAM={torch.cuda.memory_allocated(args.device) / 1e9:.1f}GB", flush=True)

    # --- optimizer ---
    optimizer = AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_steps, max(args.max_steps, args.warmup_steps + 1)
    )

    # --- output dir ---
    out_abs = os.path.abspath(args.output_dir)
    if os.path.isdir(out_abs):
        shutil.rmtree(out_abs)
    os.makedirs(out_abs, exist_ok=True)
    with open(os.path.join(out_abs, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    metrics_path = os.path.join(out_abs, "train_metrics.jsonl")

    # --- training loop ---
    print(f"[train] max_steps={args.max_steps}, lr={args.lr}, batch={args.batch_size}", flush=True)
    idx_all = list(range(n))
    rng = random.Random(args.seed)
    rng.shuffle(idx_all)
    ptr = 0

    best_loss = float("inf")
    t_train = time.time()
    losses = []

    for step in range(1, args.max_steps + 1):
        # sample batch
        batch_idx = []
        while len(batch_idx) < args.batch_size:
            if ptr >= n:
                rng.shuffle(idx_all)
                ptr = 0
            batch_idx.append(idx_all[ptr])
            ptr += 1

        batch_seqs = [seqs[i][:args.max_seq_len] for i in batch_idx]
        batch_plens = [min(plens[i], args.max_seq_len - 1) for i in batch_idx]

        input_ids, attn = pad_batch(batch_seqs, pad_id=tokenizer.pad_token_id or 0)
        input_ids = input_ids.to(args.device)
        attn = attn.to(args.device)

        # build CE target mask: only continuation positions count
        # shift-by-one CE: predict token t from position t-1
        # so valid target positions = [plen, L-1], and we compute logits at [plen-1, L-2]
        labels = input_ids.clone()
        if args.ce_over_continuation_only:
            for i, plen in enumerate(batch_plens):
                labels[i, :plen] = -100  # don't score prompt tokens
        # also mask padding
        labels[attn == 0] = -100

        optimizer.zero_grad()
        out = student(input_ids, attention_mask=attn)
        logits = out.logits  # [B, L, V]

        # shift: predict token t (label[:, 1:]) from logits[:, :-1]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="mean",
        )

        if not torch.isfinite(loss):
            print(f"[non-finite loss at step {step}], stopping.", flush=True)
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        l = loss.item()
        losses.append(l)
        lr = scheduler.get_last_lr()[0]

        # log
        if step == 1 or step % 5 == 0 or step == args.max_steps:
            rate = step / (time.time() - t_train)
            recent = sum(losses[-10:]) / min(10, len(losses))
            print(f"step {step} | loss={l:.4f} (last10={recent:.4f}) | lr={lr:.2e} | "
                  f"{rate:.2f} step/s", flush=True)

        with open(metrics_path, "a") as f:
            f.write(json.dumps({"step": step, "loss": l, "lr": lr}) + "\n")

        # save best (gated + only after warmup so we don't thrash early steps)
        if l < best_loss:
            best_loss = l
            if args.save_best and step >= args.save_best_after:
                save_ckpt(student, tokenizer, args, step, "best_train_loss",
                          extra={"best_loss": best_loss})

        if step % args.save_every == 0:
            save_ckpt(student, tokenizer, args, step, f"step_{step}")

        del out, logits, shift_logits, shift_labels, loss

    save_ckpt(student, tokenizer, args, args.max_steps, "final")
    print(f"[done] best_train_loss={best_loss:.4f}, total={time.time() - t_train:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
