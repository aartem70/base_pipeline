"""
Phase 4: 8-GPU DDP on-policy training from a cached teacher-continuation file.

Same loss as runs/train_oncache.py (prod-matching top-128 renorm KL on
continuation positions only), but data-parallel across N GPUs via HF
Accelerate. Each GPU processes 1 entry per step, gradients all-reduced →
effective batch = N. Avoids padding waste on variable-length sequences.

Launch:
    accelerate launch --num_processes 8 --mixed_precision no \\
        runs/train_oncache_ddp.py \\
        --student kharchevnykov/distil \\
        --cache /root/base_pipeline/caches/train_continuations/sglang_10k.pt \\
        --lr 5e-7 --max_steps 3000 --warmup_steps 50 --save_every 500 \\
        --output_dir /root/checkpoints/exp_oncache_ddp_fullft_lr5e-7

Add --lora --lora_rank R for LoRA training; default is FULL FINE-TUNE.
"""

import argparse, copy, json, math, os, random, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


def kl_top128_renorm(t_topk_idx, t_topk_logp, s_logits):
    t_log_p = t_topk_logp - t_topk_logp.logsumexp(dim=-1, keepdim=True)
    s_log_p_full = F.log_softmax(s_logits.float(), dim=-1)
    s_log_p_k = s_log_p_full.gather(-1, t_topk_idx)
    s_log_p_k_norm = s_log_p_k - s_log_p_k.logsumexp(dim=-1, keepdim=True)
    kl_per_pos = (t_log_p.exp() * (t_log_p - s_log_p_k_norm)).sum(dim=-1)
    return kl_per_pos.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", required=True)
    # LoRA toggle (default FULL FT)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_target", default="q_proj,k_proj,v_proj,o_proj")
    args = ap.parse_args()

    accelerator = Accelerator(mixed_precision="no")  # keep bf16 on the model itself
    is_main = accelerator.is_main_process
    rank = accelerator.process_index
    world = accelerator.num_processes
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        log_path = os.path.join(args.output_dir, "train.log")
        log_f = open(log_path, "a")
    def log(msg):
        if is_main:
            s = f"[{time.strftime('%H:%M:%S')}] {msg}"
            print(s, flush=True); log_f.write(s + "\n"); log_f.flush()

    log(f"world_size={world}  effective_batch={world}")
    log(f"loading cache: {args.cache}")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    entries = cache["entries"]
    log(f"  {len(entries)} entries (K={cache.get('logprobs_k', 128)})")

    log(f"loading student: {args.student}")
    t0 = time.time()
    student = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=torch.bfloat16, trust_remote_code=True,
    )
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
    else:
        log(f"  FULL fine-tune (all params trainable)")
    student.train()
    student.gradient_checkpointing_enable()

    optimizer = AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay, fused=True,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, 100_000)

    student, optimizer, scheduler = accelerator.prepare(student, optimizer, scheduler)

    if is_main:
        n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
        log(f"  loaded in {time.time()-t0:.1f}s  trainable={n_train:,}  "
            f"VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB/GPU")
        log(f"  optimizer: AdamW(fused=True), LR={args.lr}, warmup={args.warmup_steps}")

    # Each rank gets its own deterministic shuffled order over the cache.
    rng = random.Random(args.seed + rank)
    indices = list(range(len(entries))); rng.shuffle(indices)
    pos = 0
    metrics = []
    best_kl = float("inf")
    t_start = time.time()

    for step in range(1, args.max_steps + 1):
        # one entry per rank; gradients all-reduced inside accelerator step
        if pos >= len(indices):
            rng.shuffle(indices); pos = 0
        entry = entries[indices[pos]]; pos += 1

        full_ids = entry["full_ids"].to(accelerator.device)
        plen = int(entry["prompt_len"])
        t_idx = entry["teacher_topk_indices"].to(accelerator.device)
        t_lp = entry["teacher_topk_logprobs"].to(accelerator.device)

        with accelerator.accumulate(student):
            s_full = student(full_ids).logits
            s_cont = s_full[:, plen - 1:-1, :]
            min_len = min(s_cont.shape[1], t_idx.shape[1])
            if min_len == 0:
                # rare; do a no-op zero loss to keep DDP in sync
                loss = (s_full.sum() * 0.0)
            else:
                loss = kl_top128_renorm(
                    t_idx[:, :min_len, :], t_lp[:, :min_len, :], s_cont[:, :min_len, :],
                )
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad], 1.0,
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # gather loss across ranks for logging
        loss_gathered = accelerator.gather(loss.detach().unsqueeze(0)).mean().item()
        if is_main:
            metrics.append({"step": step, "kl": loss_gathered, "lr": scheduler.get_last_lr()[0]})
            if loss_gathered < best_kl: best_kl = loss_gathered
            if step == 1 or step % 25 == 0 or step == args.max_steps:
                elapsed = time.time() - t_start
                rate = step / elapsed
                eta = (args.max_steps - step) / rate if rate > 0 else 0
                log(f"step {step:>5d}/{args.max_steps}  KL={loss_gathered:.4f}  "
                    f"LR={scheduler.get_last_lr()[0]:.2e}  ({elapsed:.0f}s, ETA {eta:.0f}s)")

        del s_full, s_cont, t_idx, t_lp, loss, full_ids

        if step % args.save_every == 0 or step == args.max_steps:
            ck_dir = os.path.join(args.output_dir, f"step_{step}" if step != args.max_steps else "final")
            accelerator.wait_for_everyone()
            if is_main:
                os.makedirs(ck_dir, exist_ok=True)
                log(f"  saving checkpoint → {ck_dir}")
                unwrapped = accelerator.unwrap_model(student)
                try:
                    from peft import PeftModel
                    is_peft = isinstance(unwrapped, PeftModel)
                except Exception:
                    is_peft = False
                if is_peft:
                    merged = copy.deepcopy(unwrapped).to("cpu").merge_and_unload()
                    merged.save_pretrained(ck_dir, safe_serialization=True)
                    del merged
                else:
                    unwrapped.save_pretrained(ck_dir, safe_serialization=True)
                tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
                tok.save_pretrained(ck_dir)
            accelerator.wait_for_everyone()

    if is_main:
        with open(os.path.join(args.output_dir, "train_metrics.json"), "w") as f:
            json.dump({"args": vars(args), "metrics": metrics, "best_kl": best_kl}, f, indent=2)
        log(f"DONE  best_kl={best_kl:.4f}  total={time.time()-t_start:.0f}s")
        log_f.close()


if __name__ == "__main__":
    main()
