"""
On-policy training from a cached teacher-continuation file.

Loads a top-128 cached continuation file (built by build_train_cache_sglang.py
or build_train_cache_continuations.py), then trains the student to match the
cached teacher distribution on the continuation positions ONLY — exactly the
positions the prod validator scores. No teacher forward at runtime; the
student only needs 1 GPU.

Loss matches prod validator's compute_kl_from_sparse exactly:
    t_log_p_k    = teacher_topk_logprobs - logsumexp(teacher_topk_logprobs)
    s_log_p_full = log_softmax(student_logits[:, prompt_len-1:-1, :])
    s_log_p_k    = s_log_p_full.gather(-1, teacher_topk_indices)
    s_log_p_k_n  = s_log_p_k - logsumexp(s_log_p_k)
    KL[b,p]      = sum_k exp(t_log_p_k) * (t_log_p_k - s_log_p_k_n)

Per-step loss = mean over (batch, continuation positions).

Usage:
    python runs/train_oncache.py \\
        --student kharchevnykov/distil \\
        --cache /root/base_pipeline/caches/train_continuations/sglang_800.pt \\
        --gpu 0 --lr 5e-7 --max_steps 600 --batch_size 1 \\
        --lora --lora_rank 8 \\
        --output_dir /root/checkpoints/exp_oncache_lr5e-7_r8
"""

import argparse, copy, json, math, os, random, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


def kl_top128_renorm(t_topk_idx, t_topk_logp, s_logits):
    """Compute prod-matching top-K renorm KL on a single sample.

    t_topk_idx   : [1, gen_len, K] int64 on student device
    t_topk_logp  : [1, gen_len, K] float on student device
    s_logits     : [1, gen_len, V] float on student device  (continuation slice)

    Returns scalar loss (mean over positions).
    """
    t_log_p = t_topk_logp - t_topk_logp.logsumexp(dim=-1, keepdim=True)
    s_log_p_full = F.log_softmax(s_logits.float(), dim=-1)
    s_log_p_k = s_log_p_full.gather(-1, t_topk_idx)
    s_log_p_k_norm = s_log_p_k - s_log_p_k.logsumexp(dim=-1, keepdim=True)
    kl_per_pos = (t_log_p.exp() * (t_log_p - s_log_p_k_norm)).sum(dim=-1)   # [1, gen_len]
    return kl_per_pos.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--cache", required=True, help="continuation cache .pt file")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--warmup_steps", type=int, default=20)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=1, help="entries per optimizer step (each entry = 1 prompt+continuation, kept separate due to varying lengths)")
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", required=True)
    # LoRA
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_target", default="q_proj,k_proj,v_proj,o_proj")
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    device = f"cuda:{args.gpu}"
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train.log")
    log_f = open(log_path, "a")
    def log(msg):
        s = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(s, flush=True); log_f.write(s + "\n"); log_f.flush()

    log(f"loading cache: {args.cache}")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    entries = cache["entries"]
    log(f"  {len(entries)} entries (K={cache.get('logprobs_k', 128)})")

    log(f"loading student: {args.student}")
    t0 = time.time()
    student = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    if args.lora:
        from peft import LoraConfig, get_peft_model
        targets = [t.strip() for t in args.lora_target.split(",") if t.strip()]
        student = get_peft_model(student, LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
            target_modules=targets, bias="none", task_type="CAUSAL_LM",
        ))
        if hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()
        log(f"  LoRA r={args.lora_rank} α={args.lora_alpha} targets={targets}")
    student.train()
    student.gradient_checkpointing_enable()
    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    log(f"  loaded in {time.time()-t0:.1f}s   trainable={n_train:,}   "
        f"VRAM={torch.cuda.memory_allocated(device)/1e9:.1f}GB")

    optimizer = AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay, fused=True,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, 100_000)
    log(f"  optimizer: AdamW(fused=True), LR={args.lr}, warmup={args.warmup_steps}")

    rng = random.Random(args.seed)
    indices = list(range(len(entries))); rng.shuffle(indices)
    pos = 0
    metrics = []
    best_kl = float("inf")
    t_start = time.time()

    for step in range(1, args.max_steps + 1):
        # accumulate gradient over batch_size entries (each varying length, processed solo)
        optimizer.zero_grad()
        step_loss = 0.0
        n_microbatches = args.batch_size
        for _ in range(n_microbatches):
            if pos >= len(indices):
                rng.shuffle(indices); pos = 0
            entry = entries[indices[pos]]; pos += 1
            full_ids = entry["full_ids"].to(device)
            plen = int(entry["prompt_len"])
            t_idx = entry["teacher_topk_indices"].to(device)
            t_lp = entry["teacher_topk_logprobs"].to(device)
            s_full = student(full_ids).logits
            s_cont = s_full[:, plen - 1:-1, :]
            min_len = min(s_cont.shape[1], t_idx.shape[1])
            if min_len == 0:
                continue
            loss = kl_top128_renorm(
                t_idx[:, :min_len, :], t_lp[:, :min_len, :], s_cont[:, :min_len, :],
            )
            (loss / n_microbatches).backward()
            step_loss += loss.item()
            del s_full, s_cont, t_idx, t_lp, loss
        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        scheduler.step()

        avg_loss = step_loss / n_microbatches
        metrics.append({"step": step, "kl": avg_loss, "lr": scheduler.get_last_lr()[0]})
        if avg_loss < best_kl: best_kl = avg_loss
        if step == 1 or step % 10 == 0 or step == args.max_steps:
            elapsed = time.time() - t_start
            rate = step / elapsed
            eta = (args.max_steps - step) / rate if rate > 0 else 0
            log(f"step {step:>4d}/{args.max_steps}  KL={avg_loss:.4f}  "
                f"LR={scheduler.get_last_lr()[0]:.2e}  ({elapsed:.0f}s, ETA {eta:.0f}s)")

        if step % args.save_every == 0 or step == args.max_steps:
            ck_dir = os.path.join(args.output_dir, f"step_{step}" if step != args.max_steps else "final")
            os.makedirs(ck_dir, exist_ok=True)
            log(f"  saving checkpoint → {ck_dir}")
            try:
                from peft import PeftModel
                is_peft = isinstance(student, PeftModel)
            except Exception:
                is_peft = False
            if is_peft:
                merged = copy.deepcopy(student).to("cpu").merge_and_unload()
                merged.save_pretrained(ck_dir, safe_serialization=True)
                del merged
            else:
                student.save_pretrained(ck_dir, safe_serialization=True)
            tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
            tok.save_pretrained(ck_dir)

    # save metrics
    with open(os.path.join(args.output_dir, "train_metrics.json"), "w") as f:
        json.dump({"args": vars(args), "metrics": metrics, "best_kl": best_kl}, f, indent=2)
    log(f"DONE  best_kl={best_kl:.4f}  total={time.time()-t_start:.0f}s")
    log_f.close()


if __name__ == "__main__":
    main()
