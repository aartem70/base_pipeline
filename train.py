#!/usr/bin/env python3
"""
KL Distillation Training

Train a student model to match the teacher's (Qwen3.5-35B-A3B) output distribution
using forward KL divergence on random prompts from karpathy/climbmix-400b-shuffle.

Requirements:
    pip install transformers>=5.3.0 torch datasets huggingface-hub

Saved checkpoints rewrite config.json to Qwen3_5ForConditionalGeneration layout and add
preprocessor_config.json so ``evaluate.py`` / the validator see the expected Hub format.

Usage (2 GPUs - teacher on GPU 0, student on GPU 1):
    python train.py --teacher_gpu 0 --student_gpu 1

If only one CUDA device is available, invalid GPU indices automatically enable --single_gpu.

Usage (start from a specific base model):
    python train.py --student some_user/their_model --teacher_gpu 0 --student_gpu 1

Usage (single GPU with device_map auto - needs ~90GB+ VRAM):
    python train.py --single_gpu

Usage (local dev with smaller models, e.g. 2x 24GB GPUs):
    python train.py --teacher Qwen/Qwen3.5-4B --student Qwen/Qwen3.5-0.8B --teacher_gpu 0 --student_gpu 1

Prompt sampling:
    --prompts_per_step    Number of random prompts per training step (default: 60)
    --seed                Random seed for prompt sampling (default: 42)
    --resample_every      Resample new prompts every N steps (default: 1, i.e. fresh prompts each step)

Hyperparameters:
    --lr              Learning rate (default: 1e-4)
    --warmup_steps    LR warmup steps (default: 10)
    --max_seq_len     Max sequence length (default: 640)
    --kl_start_pos    Compute KL from this position onward (default: 128)

Stability:
    On a single GPU, teacher+student ``device_map='auto'`` is fragile; if logged KL jumps
    to tens, lower ``--lr`` (try 2e-5) and use ``--save_every 1``. The run also writes
    ``output_dir/best_train_kl/`` whenever the per-step train KL improves, so you do not
    lose a good checkpoint when a later step destroys weights (``final/`` is always last).

Output:
    By default the entire ``--output_dir`` is deleted before a new run. Use ``--resume-from``
    or ``--no-clean-output`` to preserve it.

    Each step appends one row to ``output_dir/train_metrics.csv`` and, when matplotlib is
    available, writes ``output_dir/train_curves.png`` (KL + LR vs step). Use ``--no_train_plots``
    to disable, or ``--plot_every N`` to refresh the PNG during training.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import copy
import csv
import gc
import json
import logging
import math
import random
import shutil
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"
STUDENT_MODEL = "Qwen/Qwen3.5-4B"

# Dataset
CLIMBMIX_DATASET = "karpathy/climbmix-400b-shuffle"
CLIMBMIX_NUM_SHARDS = 6542
FINEWEB_DATASET = "HuggingFaceFW/fineweb"

# Training defaults
LR = 1e-4
WARMUP = 10
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
MAX_SEQ_LEN = 640
KL_START_POS = 128
PROMPTS_PER_STEP = 60
SAVE_EVERY = 500

# Hugging Face repo to copy preprocessor_config.json from (matches evaluate.py hint).
_PREPROCESSOR_TEMPLATE_REPO = "Qwen/Qwen3.5-4B"

TRAIN_METRICS_CSV = "train_metrics.csv"
TRAIN_CURVES_PNG = "train_curves.png"
_TRAIN_METRIC_FIELDS = ("step", "kl", "lr", "step_time_s", "samples_per_sec", "total_sampled")


def finalize_checkpoint_for_submission(checkpoint_dir: str) -> None:
    """Rewrite config.json into Qwen3_5ForConditionalGeneration layout and ensure preprocessor.

    ``AutoModelForCausalLM.save_pretrained`` writes a flat causal LM config; validators expect
    top-level ``model_type: qwen3_5`` + ``Qwen3_5ForConditionalGeneration`` with the same JSON
    nested under ``text_config`` (see successful Hub submissions). Weight keys already use
    ``model.language_model.*`` for Qwen3.5 students, so only metadata + preprocessor need fixing.
    """
    cfg_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return

    with open(cfg_path) as f:
        inner = json.load(f)

    arch = inner.get("architectures") or []
    if inner.get("model_type") == "qwen3_5" and "Qwen3_5ForConditionalGeneration" in arch:
        pass  # already wrapped; still ensure preprocessor below
    elif inner.get("text_config") is not None:
        # Unusual: partially wrapped; do not replace.
        log.warning("config.json has text_config but not submission arch layout; skipping rewrite: %s", checkpoint_dir)
    else:
        text_config = copy.deepcopy(inner)
        outer = {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": text_config,
            "tie_word_embeddings": text_config.get("tie_word_embeddings", False),
        }
        if text_config.get("dtype"):
            outer["torch_dtype"] = text_config["dtype"]
        if text_config.get("transformers_version"):
            outer["transformers_version"] = text_config["transformers_version"]

        with open(cfg_path, "w") as f:
            json.dump(outer, f, indent=2)
            f.write("\n")
        log.info("  Wrote submission-style config (Qwen3_5ForConditionalGeneration + text_config)")

    pre = os.path.join(checkpoint_dir, "preprocessor_config.json")
    if os.path.isfile(pre):
        return
    try:
        from huggingface_hub import hf_hub_download

        src = hf_hub_download(_PREPROCESSOR_TEMPLATE_REPO, "preprocessor_config.json")
        shutil.copy(src, pre)
        log.info("  Copied preprocessor_config.json from %s", _PREPROCESSOR_TEMPLATE_REPO)
    except Exception as e:
        log.warning(
            "  Could not add preprocessor_config.json (evaluate.py expects it): %s — copy manually from %s",
            e,
            _PREPROCESSOR_TEMPLATE_REPO,
        )


def save_student_checkpoint(
    student, tokenizer, optimizer, args, global_step, dest_name: str, total_sampled: int,
) -> str:
    """Write student weights, tokenizer, submission config, optimizer, and train_state.json."""
    d = os.path.join(args.output_dir, dest_name)
    os.makedirs(d, exist_ok=True)
    student.save_pretrained(d, safe_serialization=True)
    tokenizer.save_pretrained(d)
    finalize_checkpoint_for_submission(d)
    torch.save(optimizer.state_dict(), os.path.join(d, "optimizer.pt"))
    with open(os.path.join(d, "train_state.json"), "w") as f:
        json.dump({
            "global_step": global_step,
            "total_sampled": total_sampled,
            "seed": args.seed,
        }, f, indent=2)
    log.info(f"  Saved checkpoint: {d}")
    return d


class RandomPromptSampler:
    """Samples random prompts from ClimbMix shards, with FineWeb fallback."""

    def __init__(self, seed=42, min_chars=2560, max_chars=10000):
        self._rng = random.Random(seed)
        self._min_chars = min_chars
        self._max_chars = max_chars
        self._current_shard = None
        self._current_indices = []
        self._index_pos = 0
        self._total_sampled = 0

    def _load_shard(self):
        """Load a random shard from ClimbMix."""
        from datasets import load_dataset

        shard_idx = self._rng.randint(0, CLIMBMIX_NUM_SHARDS - 1)
        shard_file = f"shard_{shard_idx:05d}.parquet"
        log.info(f"Loading shard {shard_idx}/{CLIMBMIX_NUM_SHARDS} from {CLIMBMIX_DATASET}...")

        try:
            ds = load_dataset(
                CLIMBMIX_DATASET,
                data_files=shard_file,
                split="train",
            )
            indices = list(range(len(ds)))
            self._rng.shuffle(indices)
            self._current_shard = ds
            self._current_indices = indices
            self._index_pos = 0
            log.info(f"  Shard {shard_idx} loaded: {len(ds)} rows")
            return True
        except Exception as e:
            log.warning(f"  Failed to load shard {shard_idx}: {e}")
            return False

    def _load_fineweb_fallback(self, n):
        """Fallback: stream from FineWeb."""
        from datasets import load_dataset

        skip_offset = self._rng.randint(0, 5_000_000)
        log.info(f"Fallback: sampling from {FINEWEB_DATASET} (skip={skip_offset:,})...")

        ds = load_dataset(FINEWEB_DATASET, split="train", streaming=True, name="default")
        ds_shuffled = ds.shuffle(seed=self._rng.randint(0, 2**31), buffer_size=50_000)
        ds_skipped = ds_shuffled.skip(skip_offset)

        texts = []
        seen = 0
        for item in ds_skipped:
            seen += 1
            text = item.get("text", "")
            if not text or len(text) < self._min_chars:
                continue
            if len(text) > self._max_chars:
                text = text[:self._max_chars]
                last_space = text.rfind(' ')
                if last_space > self._max_chars // 2:
                    text = text[:last_space]
            texts.append(text)
            if len(texts) >= n:
                break
            if seen > n * 20:
                break

        log.info(f"  FineWeb fallback got {len(texts)} prompts (scanned {seen})")
        return texts

    def sample(self, n):
        """Sample n random prompts."""
        texts = []
        attempts = 0

        while len(texts) < n and attempts < 5:
            # Load a new shard if needed
            if self._current_shard is None or self._index_pos >= len(self._current_indices):
                if not self._load_shard():
                    attempts += 1
                    continue

            # Sample from current shard
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

            # If shard exhausted, load another
            if len(texts) < n:
                self._current_shard = None

        # Final fallback to FineWeb
        if len(texts) < n:
            texts.extend(self._load_fineweb_fallback(n - len(texts)))

        self._total_sampled += len(texts)
        return texts

    @property
    def total_sampled(self):
        return self._total_sampled


def kl_loss(student_logits, teacher_logits, start_pos=KL_START_POS):
    """Forward KL(teacher || student) from start_pos onward."""
    s = student_logits[:, start_pos:, :].contiguous()
    t = teacher_logits[:, start_pos:, :].detach().to(s.device).contiguous()
    t_log_p = F.log_softmax(t.float(), dim=-1)
    s_log_p = F.log_softmax(s.float(), dim=-1)
    t_p = t_log_p.exp()
    return (t_p * (t_log_p - s_log_p)).sum(-1).mean()


def _load_train_metrics_csv(path: str) -> list:
    """Load previous metrics rows for resume (plot continues across restarts)."""
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(_TRAIN_METRIC_FIELDS):
            log.warning(
                "train_metrics.csv header mismatch; not loading prior rows: %s",
                path,
            )
            return []
        return [dict(row) for row in reader]


def _append_train_metric_row(csv_path: str, row: dict) -> None:
    new_file = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_TRAIN_METRIC_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row[k] for k in _TRAIN_METRIC_FIELDS})


def save_train_curves_plot(output_dir: str, history: list) -> None:
    """Write train_curves.png: train KL and LR vs global step (non-interactive backend)."""
    if not history:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning(
            "matplotlib not installed; skipping %s (pip install matplotlib)",
            TRAIN_CURVES_PNG,
        )
        return

    steps = [int(h["step"]) for h in history]
    kls = [float(h["kl"]) for h in history]
    lrs = [float(h["lr"]) for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")
    ax1.set_xlabel("step")
    ax1.set_ylabel("train KL", color="C0")
    ax1.plot(steps, kls, color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax2 = ax1.twinx()
    ax2.set_ylabel("learning rate", color="C1")
    ax2.plot(steps, lrs, color="C1", linestyle="--", alpha=0.85)
    ax2.tick_params(axis="y", labelcolor="C1")
    fig.suptitle("Training curves")
    fig.tight_layout()
    out_path = os.path.join(output_dir, TRAIN_CURVES_PNG)
    fig.savefig(out_path, dpi=120, facecolor="white", edgecolor="none")
    plt.close(fig)
    log.info("  Wrote %s", out_path)


def main():
    parser = argparse.ArgumentParser(
        description="KL Distillation Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model args
    parser.add_argument("--teacher", type=str, default=TEACHER_MODEL,
                        help="Teacher model (HuggingFace repo)")
    parser.add_argument("--student", type=str, default=STUDENT_MODEL,
                        help="Student model (HuggingFace repo or local path)")
    parser.add_argument("--teacher_gpu", type=int, default=0,
                        help="GPU index for teacher")
    parser.add_argument("--student_gpu", type=int, default=1,
                        help="GPU index for student")
    parser.add_argument("--single_gpu", action="store_true",
                        help="Use device_map='auto' for both models on a single large GPU")

    # Training hyperparams
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--warmup_steps", type=int, default=WARMUP)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN,
                        help="Max sequence length for tokenization")
    parser.add_argument("--kl_start_pos", type=int, default=KL_START_POS,
                        help="Compute KL from this token position onward (skip early context)")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Stop after N steps (0 = run forever)")

    # Prompt sampling
    parser.add_argument("--prompts_per_step", type=int, default=PROMPTS_PER_STEP,
                        help="Number of random prompts sampled per training step")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for prompt sampling")
    parser.add_argument("--resample_every", type=int, default=1,
                        help="Resample fresh prompts every N steps (1 = every step)")

    # Output
    parser.add_argument("--output_dir", type=str, default="./distil-checkpoints",
                        help="Directory to save checkpoints (removed and recreated each run unless "
                             "--resume-from or --no-clean-output)")
    parser.add_argument("--save_every", type=int, default=SAVE_EVERY,
                        help="Save step_N/ every N steps (use 1 when max_steps is small)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from a checkpoint directory (loads optimizer + step count)")
    parser.add_argument("--no-clean-output", action="store_true", dest="no_clean_output",
                        help="Keep existing output_dir contents (default: delete output_dir when not resuming)")

    # Logging
    parser.add_argument("--wandb_project", type=str, default="distil-training")
    parser.add_argument("--wandb_run", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--no_train_plots", action="store_true",
                        help="Do not write train_metrics.csv or train_curves.png")
    parser.add_argument("--plot_every", type=int, default=0,
                        help="Rewrite train_curves.png every N steps (0 = only at training end)")

    args = parser.parse_args()

    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        if not args.single_gpu and (
            args.teacher_gpu < 0
            or args.student_gpu < 0
            or args.teacher_gpu >= n_gpu
            or args.student_gpu >= n_gpu
        ):
            log.warning(
                "CUDA devices: %d; teacher_gpu=%s and student_gpu=%s are not all valid. "
                "Using --single_gpu (device_map='auto' for both models). "
                "For two GPUs, pass e.g. --teacher_gpu 0 --student_gpu 1.",
                n_gpu,
                args.teacher_gpu,
                args.student_gpu,
            )
            args.single_gpu = True

    if args.max_steps > 0 and args.save_every > args.max_steps:
        log.warning(
            "save_every=%d > max_steps=%d: no step_* folders will be written; "
            "only final/ (last weights). Use --save_every 1 to snapshot each step, "
            "or rely on best_train_kl/ (best train KL so far).",
            args.save_every,
            args.max_steps,
        )

    out_abs = os.path.abspath(os.path.expanduser(args.output_dir))
    if args.resume_from is None and not args.no_clean_output:
        if os.path.isdir(out_abs):
            log.info("Clearing output directory: %s", out_abs)
            shutil.rmtree(out_abs)
    elif args.resume_from and not args.no_clean_output:
        log.info("Keeping output directory (training with --resume-from).")

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Wandb
    if not args.no_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=args.wandb_run or "distil-kl", config=vars(args))
        except ImportError:
            log.warning("wandb not installed, disabling logging. pip install wandb to enable.")
            args.no_wandb = True

    # ── Load models ───────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.single_gpu:
        log.info(f"Loading teacher ({args.teacher}) with device_map='auto'...")
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher, dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tdev = teacher.device if hasattr(teacher, 'device') else torch.device("cuda:0")
    else:
        log.info(f"Loading teacher ({args.teacher}) on GPU {args.teacher_gpu}...")
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher, dtype=torch.bfloat16,
            device_map={"": args.teacher_gpu},
            trust_remote_code=True,
        )
        tdev = torch.device(f"cuda:{args.teacher_gpu}")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    log.info(f"  Teacher VRAM: {torch.cuda.memory_allocated(tdev)/1e9:.1f}GB")

    if args.single_gpu:
        log.info(f"Loading student ({args.student}) with device_map='auto'...")
        student = AutoModelForCausalLM.from_pretrained(
            args.student, dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        sdev = student.device if hasattr(student, 'device') else torch.device("cuda:0")
    else:
        log.info(f"Loading student ({args.student}) on GPU {args.student_gpu}...")
        student = AutoModelForCausalLM.from_pretrained(
            args.student, dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(f"cuda:{args.student_gpu}")
        sdev = torch.device(f"cuda:{args.student_gpu}")

    student.train()
    n_params = sum(p.numel() for p in student.parameters())
    n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    log.info(f"  Student: {n_params:,} params ({n_trainable:,} trainable), "
             f"VRAM: {torch.cuda.memory_allocated(sdev)/1e9:.1f}GB")

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    optimizer = AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, 100_000)

    global_step = 0

    # Resume from checkpoint
    if args.resume_from and os.path.isdir(args.resume_from):
        opt_path = os.path.join(args.resume_from, "optimizer.pt")
        state_path = os.path.join(args.resume_from, "train_state.json")
        if os.path.exists(opt_path):
            log.info(f"Resuming optimizer from {opt_path}")
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu", weights_only=True))
        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
            global_step = state.get("global_step", 0)
            log.info(f"  Resuming from step {global_step}")

    metrics_csv_path = os.path.join(args.output_dir, TRAIN_METRICS_CSV)
    train_history = []
    if args.resume_from and os.path.isfile(metrics_csv_path) and not args.no_train_plots:
        train_history = _load_train_metrics_csv(metrics_csv_path)
        if train_history:
            log.info(
                "  Loaded %d prior rows from %s for curve/plot continuity",
                len(train_history),
                metrics_csv_path,
            )

    # ── Prompt sampler ────────────────────────────────────────────────────
    sampler = RandomPromptSampler(seed=args.seed)
    cached_texts = None
    best_train_kl = float("inf")

    # ── Training loop ─────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Starting training")
    log.info("=" * 60)
    log.info(f"  LR: {args.lr}, Warmup: {args.warmup_steps}, Prompts/step: {args.prompts_per_step}")
    log.info(f"  Seq len: {args.max_seq_len}, KL from pos {args.kl_start_pos}")
    log.info(f"  Seed: {args.seed}, Resample every: {args.resample_every} step(s)")
    if args.max_steps > 0:
        log.info(f"  Max steps: {args.max_steps}")
    log.info("")

    while True:
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

        t0 = time.time()

        # Sample prompts (resample or reuse cached)
        if cached_texts is None or global_step % args.resample_every == 0:
            cached_texts = sampler.sample(args.prompts_per_step)
            if not cached_texts:
                log.warning("No prompts sampled, stopping.")
                break

        # Tokenize
        tokens = [
            tokenizer(t, return_tensors="pt", truncation=True,
                      max_length=args.max_seq_len).input_ids.squeeze(0)
            for t in cached_texts
        ]
        # Filter out sequences too short for KL computation
        tokens = [t for t in tokens if t.shape[0] > args.kl_start_pos + 10]

        if not tokens:
            log.warning(f"Step {global_step}: all prompts too short after tokenization, resampling...")
            cached_texts = None
            continue

        optimizer.zero_grad()
        total_loss = 0.0
        n = 0

        for ids in tokens:
            ids = ids.unsqueeze(0)
            with torch.no_grad():
                t_logits = teacher(ids.to(tdev)).logits.to(sdev)
            s_logits = student(ids.to(sdev)).logits
            loss = kl_loss(s_logits, t_logits, start_pos=args.kl_start_pos)
            (loss / len(tokens)).backward()
            total_loss += loss.item()
            n += 1
            del t_logits, s_logits, loss

        avg_kl = total_loss / max(n, 1)
        if not math.isfinite(avg_kl):
            log.error("Non-finite training KL; skipping optimizer update and stopping.")
            optimizer.zero_grad(set_to_none=True)
            break

        torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        global_step += 1

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]
        log.info(f"Step {global_step} | KL: {avg_kl:.4f} | LR: {lr:.2e} | "
                 f"{elapsed:.1f}s ({n/elapsed:.1f} samp/s) | total_sampled: {sampler.total_sampled:,}")

        if not args.no_wandb:
            import wandb
            wandb.log({"train/kl": avg_kl, "train/lr": lr,
                       "perf/step_time": elapsed, "perf/samples_per_sec": n / elapsed,
                       "data/total_sampled": sampler.total_sampled}, step=global_step)

        if not args.no_train_plots:
            metric_row = {
                "step": global_step,
                "kl": f"{avg_kl:.6f}",
                "lr": f"{lr:.10e}",
                "step_time_s": f"{elapsed:.4f}",
                "samples_per_sec": f"{(n / elapsed):.4f}" if elapsed > 0 else "0",
                "total_sampled": sampler.total_sampled,
            }
            train_history.append(metric_row)
            _append_train_metric_row(metrics_csv_path, metric_row)
            if args.plot_every > 0 and global_step % args.plot_every == 0:
                save_train_curves_plot(args.output_dir, train_history)

        if avg_kl < best_train_kl:
            best_train_kl = avg_kl
            save_student_checkpoint(
                student, tokenizer, optimizer, args, global_step,
                "best_train_kl", sampler.total_sampled,
            )

        if global_step % args.save_every == 0:
            save_student_checkpoint(
                student, tokenizer, optimizer, args, global_step,
                f"step_{global_step}", sampler.total_sampled,
            )

        # Periodic cleanup
        if global_step % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    if not args.no_train_plots and train_history:
        save_train_curves_plot(args.output_dir, train_history)

    # Final save (last weights after last optimizer step — may be worse than best_train_kl/)
    save_student_checkpoint(
        student, tokenizer, optimizer, args, global_step,
        "final", sampler.total_sampled,
    )
    if best_train_kl < float("inf"):
        log.info(
            "Training complete at step %d. For eval, prefer best_train_kl/ if train KL spiked "
            "(best train KL was %.4f; see final/ for last step only).",
            global_step,
            best_train_kl,
        )
    else:
        log.info("Training complete at step %d.", global_step)

    if not args.no_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
