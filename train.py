#!/usr/bin/env python3
"""
KL Distillation Training (v2 — batched, accelerated)

Train a student model to match the teacher's (Qwen3.5-35B-A3B) output distribution
using forward KL divergence on random prompts from karpathy/climbmix-400b-shuffle.

Requirements:
    pip install transformers>=5.3.0 torch datasets huggingface-hub

Saved checkpoints rewrite config.json to Qwen3_5ForConditionalGeneration layout and add
preprocessor_config.json so ``evaluate.py`` / the validator see the expected Hub format.

Key optimizations over v1:
    - Batched tokenization (single call, with padding)
    - Batched forward passes (teacher + student) instead of sample-by-sample
    - Masked KL computation (ignores padding tokens)
    - Optional top-k distillation (--topk_distil K) to reduce cross-GPU bandwidth
    - Flash attention enabled by default when available

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
    --max_seq_len     Max sequence length (default: 1024)
    --kl_start_pos    Compute KL from this position onward (default: 0)
    --batch_size      Micro-batch size for forward passes (default: 4)
    --topk_distil     Transfer only top-K logits across GPUs (0 = full vocab, default: 0)

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

# Training defaults
LR = 1e-4
WARMUP = 10
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
MAX_SEQ_LEN = 1024
KL_START_POS = 0
PROMPTS_PER_STEP = 60
SAVE_EVERY = 500
BATCH_SIZE = 4
PLOT_EVERY = 10

# Hugging Face repo to copy preprocessor_config.json from (matches evaluate.py hint).
_PREPROCESSOR_TEMPLATE_REPO = "Qwen/Qwen3.5-4B"

TRAIN_METRICS_CSV = "train_metrics.csv"
TRAIN_CURVES_PNG = "train_curves.png"
_TRAIN_METRIC_FIELDS = ("step", "kl", "lr", "step_time_s", "samples_per_sec", "total_sampled")


def finalize_checkpoint_for_submission(checkpoint_dir: str) -> None:
    """Rewrite config.json into Qwen3_5ForConditionalGeneration layout and ensure preprocessor."""
    cfg_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return

    with open(cfg_path) as f:
        inner = json.load(f)

    arch = inner.get("architectures") or []
    if inner.get("model_type") == "qwen3_5" and "Qwen3_5ForConditionalGeneration" in arch:
        pass
    elif inner.get("text_config") is not None:
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
            "  Could not add preprocessor_config.json (evaluate.py expects it): %s",
            e,
        )


def save_student_checkpoint(
    student, tokenizer, optimizer, args, global_step, dest_name: str, total_sampled: int,
) -> str:
    d = os.path.join(args.output_dir, dest_name)
    os.makedirs(d, exist_ok=True)

    # If student is a PEFT-wrapped model, save a merged full model so that
    # evaluate.py can load it with AutoModelForCausalLM. We deepcopy to CPU
    # and merge there; the in-memory PEFT wrapper stays intact for training.
    try:
        from peft import PeftModel
        is_peft = isinstance(student, PeftModel)
    except Exception:
        is_peft = False
    if is_peft:
        import copy
        log.info(f"  [LoRA] merging adapters and saving full model to {d}")
        merged = copy.deepcopy(student).to("cpu").merge_and_unload()
        merged.save_pretrained(d, safe_serialization=True)
        del merged
    else:
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


class JSONLPromptSampler:
    """Samples prompts from a local JSONL file. Each line must be a JSON object
    with a 'text' field. Shuffles once on load, then streams in order."""

    def __init__(self, path, seed=42, min_chars=2560, max_chars=10000):
        self._path = path
        self._rng = random.Random(seed)
        self._min_chars = min_chars
        self._max_chars = max_chars
        self._texts = None
        self._pos = 0
        self._total_sampled = 0
        self._load()

    def _load(self):
        texts = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = obj.get("text", "")
                if not t or len(t) < self._min_chars:
                    continue
                if len(t) > self._max_chars:
                    t = t[:self._max_chars]
                    ls = t.rfind(" ")
                    if ls > self._max_chars // 2:
                        t = t[:ls]
                texts.append(t)
        self._rng.shuffle(texts)
        self._texts = texts
        self._pos = 0
        log.info(f"Loaded {len(texts)} prompts from {self._path}")

    def sample(self, n):
        out = []
        while len(out) < n:
            if self._pos >= len(self._texts):
                # reshuffle and wrap
                self._rng.shuffle(self._texts)
                self._pos = 0
            out.append(self._texts[self._pos])
            self._pos += 1
        self._total_sampled += len(out)
        return out

    @property
    def total_sampled(self):
        return self._total_sampled


class RandomPromptSampler:
    """Samples random prompts from ClimbMix shards only."""

    def __init__(self, seed=42, min_chars=2560, max_chars=10000):
        self._rng = random.Random(seed)
        self._min_chars = min_chars
        self._max_chars = max_chars
        self._current_shard = None
        self._current_indices = []
        self._index_pos = 0
        self._total_sampled = 0

    def _load_shard(self):
        from datasets import load_dataset

        shard_idx = self._rng.randint(0, CLIMBMIX_NUM_SHARDS - 1)
        shard_file = f"shard_{shard_idx:05d}.parquet"
        log.info(f"Loading shard {shard_idx}/{CLIMBMIX_NUM_SHARDS} from {CLIMBMIX_DATASET}...")

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

        self._total_sampled += len(texts)
        return texts

    @property
    def total_sampled(self):
        return self._total_sampled


# -- Batched KL loss with padding mask -----------------------------------------

def batched_kl_loss(student_logits, teacher_logits, attention_mask,
                    start_pos=KL_START_POS, kl_mode="top128", kl_topk=128):
    """Forward KL(teacher || student), masked and averaged.

    kl_mode:
      "fullvocab"  — legacy full-vocab forward KL over all logits (pre-Day-11 loss).
      "top128"     — prod-validator-matching loss. Exactly the math of
                     compute_kl_from_sparse() in unarbos/distil/scripts/pod_eval_vllm.py:

                       t_log_p_k   = log_softmax_k( teacher_logits.topk(K).values )
                       s_log_p_k   = log_softmax_V( student_logits ).gather(at_teacher_top_k_idx)
                       s_log_p_k_n = log_softmax_k( s_log_p_k )
                       KL[b,l]     = sum_k exp(t_log_p_k) * ( t_log_p_k - s_log_p_k_n )

                     i.e. both teacher and student are renormalized to a proper
                     probability simplex over the teacher's top-K support before
                     the KL is taken. K=128 matches pod_eval_vllm.py --logprobs-k 128.

    Args:
        student_logits: [B, L, V]   student raw logits
        teacher_logits: [B, L, V]   teacher raw logits (already on student device)
        attention_mask: [B, L]      1 = real token, 0 = padding
        start_pos:                  drop the first N positions (>=0)
        kl_mode:                    "fullvocab" | "top128"
        kl_topk:                    K for "top128" (default 128)
    """
    s = student_logits[:, start_pos:, :].float()
    t = teacher_logits[:, start_pos:, :].detach().float()
    mask = attention_mask[:, start_pos:].float()   # [B, L-start_pos]

    if kl_mode == "fullvocab":
        t_log_p = F.log_softmax(t, dim=-1)
        s_log_p = F.log_softmax(s, dim=-1)
        kl_per_pos = (t_log_p.exp() * (t_log_p - s_log_p)).sum(dim=-1)    # [B, L-start_pos]
    elif kl_mode == "top128":
        # Teacher: top-K logits → renormalize as a distribution over K.
        t_topk_vals, t_topk_idx = t.topk(kl_topk, dim=-1)                 # [B, L, K]
        t_log_p_k = F.log_softmax(t_topk_vals, dim=-1)                    # [B, L, K]
        # Student: full-vocab log-softmax → gather at teacher's K indices → renorm.
        s_log_p_full = F.log_softmax(s, dim=-1)
        s_log_p_k = s_log_p_full.gather(-1, t_topk_idx)                   # [B, L, K]
        del s_log_p_full
        s_log_p_k_norm = s_log_p_k - s_log_p_k.logsumexp(dim=-1, keepdim=True)
        kl_per_pos = (t_log_p_k.exp() * (t_log_p_k - s_log_p_k_norm)).sum(dim=-1)
    else:
        raise ValueError(f"unknown kl_mode: {kl_mode!r} (use 'top128' or 'fullvocab')")

    kl_per_pos = kl_per_pos * mask
    n_valid = mask.sum()
    if n_valid == 0:
        return torch.tensor(0.0, device=s.device, requires_grad=True)
    return kl_per_pos.sum() / n_valid


def topk_compress(logits, k):
    """Compress logits to top-k values and indices for bandwidth-efficient transfer.

    Returns (values, indices) where values is [B, L, K] and indices is [B, L, K].
    """
    return torch.topk(logits, k=k, dim=-1)


def topk_decompress(values, indices, vocab_size, device):
    """Reconstruct full logits from top-k values and indices, filling rest with -inf."""
    B, L, K = values.shape
    full = torch.full((B, L, vocab_size), -1e10, device=device, dtype=values.dtype)
    full.scatter_(-1, indices.to(device), values.to(device))
    return full


# -- vLLM teacher wrapper ------------------------------------------------------

class VLLMTeacher:
    """Wraps vLLM engine for fast teacher logit extraction.

    Uses vLLM's prompt_logprobs to get teacher's token-level log-probabilities
    without autoregressive generation. Much faster than HuggingFace forward pass
    due to PagedAttention, continuous batching, and optimized CUDA kernels.
    """

    def __init__(self, model_name, gpu_id=0, gpu_memory_utilization=0.90, topk=512, max_model_len=4096):
        from vllm import LLM

        self.topk = topk
        self.model_name = model_name
        log.info(f"Loading teacher via vLLM ({model_name}) on GPU {gpu_id}...")
        log.info(f"  gpu_memory_utilization={gpu_memory_utilization}, prompt_logprobs top-k={topk}, max_model_len={max_model_len}")

        self.llm = LLM(
            model=model_name,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            trust_remote_code=True,
            enforce_eager=True,
            max_model_len=max_model_len,
        )
        log.info("  vLLM teacher loaded.")

    def get_logits(self, input_ids_batch, attention_mask_batch, vocab_size, device):
        """Get teacher logits for a batch of token sequences.

        Args:
            input_ids_batch: [B, L] tensor of token ids
            attention_mask_batch: [B, L] tensor (1=real, 0=pad)
            vocab_size: full vocabulary size
            device: target device for output tensor

        Returns:
            [B, L, vocab_size] tensor of logits on target device (sparse, top-k filled)
        """
        from vllm import SamplingParams
        import torch

        B, L = input_ids_batch.shape
        k = min(self.topk, vocab_size)

        # Convert padded batch to list of variable-length token id lists
        prompt_token_ids = []
        for i in range(B):
            seq_len = attention_mask_batch[i].sum().item()
            ids = input_ids_batch[i, :seq_len].tolist()
            prompt_token_ids.append(ids)

        # Use prompt_logprobs to get teacher's view of each input token
        params = SamplingParams(
            max_tokens=1,
            temperature=0,
            prompt_logprobs=k,
        )
        outputs = self.llm.generate(
            prompt_token_ids=prompt_token_ids,
            sampling_params=params,
        )

        # Reconstruct logits tensor [B, L, V] from prompt_logprobs
        # vLLM returns prompt_logprobs as list of dicts per position
        full_logits = torch.full((B, L, vocab_size), -1e10, dtype=torch.bfloat16, device=device)

        for b, output in enumerate(outputs):
            plogprobs = output.prompt_logprobs  # list of (dict or None) per position
            if plogprobs is None:
                continue
            for pos, token_logprobs in enumerate(plogprobs):
                if token_logprobs is None:
                    continue  # first position has no logprobs
                for token_id, logprob_obj in token_logprobs.items():
                    lp = logprob_obj.logprob if hasattr(logprob_obj, 'logprob') else logprob_obj
                    full_logits[b, pos, token_id] = lp

        return full_logits


# -- CSV / plotting helpers ----------------------------------------------------

def _load_train_metrics_csv(path: str) -> list:
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(_TRAIN_METRIC_FIELDS):
            log.warning("train_metrics.csv header mismatch; not loading prior rows: %s", path)
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
    if not history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping %s", TRAIN_CURVES_PNG)
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


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KL Distillation Training (v2 — batched, accelerated)",
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
    parser.add_argument("--kl_mode", choices=["top128", "fullvocab"], default="top128",
                        help="Loss form. 'top128' matches the prod validator "
                             "(unarbos/distil compute_kl_from_sparse) exactly: renorm KL over the "
                             "teacher's top-K support. 'fullvocab' is the legacy pre-Day-11 loss.")
    parser.add_argument("--kl_topk", type=int, default=128,
                        help="K for --kl_mode top128 (default 128, matches prod --logprobs-k 128).")
    parser.add_argument("--no_fused_adam", action="store_true",
                        help="Disable fused AdamW kernel (default: on when CUDA is available). "
                             "Fused keeps optimizer moments in fp32 so updates don't silently "
                             "round to zero under bf16 params at small LR.")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Stop after N steps (0 = run forever)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Micro-batch size for forward passes (adjust based on GPU memory)")
    parser.add_argument("--topk_distil", type=int, default=0,
                        help="Transfer only top-K logits between GPUs (0 = full vocab). "
                             "Reduces cross-GPU bandwidth. Try 512 or 1024.")
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use vLLM for teacher inference (2-5x faster). "
                             "Requires: pip install vllm. Teacher runs as vLLM engine "
                             "instead of HuggingFace model.")
    parser.add_argument("--vllm_gpu_util", type=float, default=0.90,
                        help="vLLM gpu_memory_utilization (0-1). Lower if OOM.")

    # Prompt sampling
    parser.add_argument("--prompts_per_step", type=int, default=PROMPTS_PER_STEP,
                        help="Number of random prompts sampled per training step")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for prompt sampling")
    parser.add_argument("--resample_every", type=int, default=1,
                        help="Resample fresh prompts every N steps (1 = every step)")
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="Optional local JSONL file with prompts (each line: {'text': ...}). "
                             "If set, overrides the HF ClimbMix sampler. Used for per-category training.")

    # LoRA / PEFT options (via HF peft library)
    parser.add_argument("--lora", action="store_true",
                        help="Wrap student with LoRA adapters (uses HF peft).")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA rank r (default 8, matching Day 7 Exp 7.4).")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha (scaling). Default 16 (= 2x rank).")
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target", type=str,
                        default="q_proj,k_proj,v_proj,o_proj",
                        help="Comma-separated list of linear module names to adapt "
                             "(default: attn-only, matching Day 7 Exp 7.4 safe recipe).")
    parser.add_argument("--use_dora", action="store_true",
                        help="Use DoRA (weight-decomposition LoRA) instead of vanilla LoRA.")
    parser.add_argument("--use_rslora", action="store_true",
                        help="Use rank-stabilized LoRA scaling (alpha / sqrt(r)).")

    # Output
    parser.add_argument("--output_dir", type=str, default="./distil-checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--save_every", type=int, default=SAVE_EVERY,
                        help="Save step_N/ every N steps")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from a checkpoint directory (loads optimizer + step count)")
    parser.add_argument("--no-clean-output", action="store_true", dest="no_clean_output",
                        help="Keep existing output_dir contents")

    # Logging
    parser.add_argument("--wandb_project", type=str, default="distil-training")
    parser.add_argument("--wandb_run", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--no_train_plots", action="store_true",
                        help="Do not write train_metrics.csv or train_curves.png")
    parser.add_argument("--plot_every", type=int, default=PLOT_EVERY,
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
                "Using --single_gpu.",
                n_gpu, args.teacher_gpu, args.student_gpu,
            )
            args.single_gpu = True

    if args.max_steps > 0 and args.save_every > args.max_steps:
        log.warning(
            "save_every=%d > max_steps=%d: no step_* folders will be written; "
            "only final/ (last weights).",
            args.save_every, args.max_steps,
        )

    out_abs = os.path.abspath(os.path.expanduser(args.output_dir))
    if args.resume_from is None and not args.no_clean_output:
        if os.path.isdir(out_abs):
            log.info("Clearing output directory: %s", out_abs)
            shutil.rmtree(out_abs)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Wandb
    if not args.no_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=args.wandb_run or "distil-kl", config=vars(args))
        except ImportError:
            log.warning("wandb not installed, disabling logging.")
            args.no_wandb = True

    # -- Load models -----------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Try flash attention for student (and teacher if not using vLLM)
    attn_kwargs = {}
    try:
        import flash_attn  # noqa: F401
        attn_kwargs["attn_implementation"] = "flash_attention_2"
        log.info("Flash Attention 2 available.")
    except ImportError:
        log.info("Flash Attention not available, using default attention.")

    # -- Load teacher ----------------------------------------------------------
    # Skip teacher loading entirely when using a pre-built cache file
    skip_teacher = (
        getattr(args, "continuation", False)
        and getattr(args, "cache_continuations", None)
        and os.path.isfile(getattr(args, "cache_continuations", ""))
    )
    vllm_teacher = None

    if skip_teacher:
        log.info("Pre-built cache found, skipping teacher loading entirely.")
        teacher = None
        tdev = None
    elif args.use_vllm:
        # vLLM manages its own GPU placement
        topk_for_vllm = args.topk_distil if args.topk_distil > 0 else 512
        if args.topk_distil == 0:
            log.warning(
                "--use_vllm requires top-k logit extraction. Setting --topk_distil=%d "
                "(vLLM returns log-probabilities, not raw logits for full vocab).",
                topk_for_vllm,
            )
            args.topk_distil = topk_for_vllm

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.teacher_gpu)
        vllm_teacher = VLLMTeacher(
            args.teacher,
            gpu_id=args.teacher_gpu,
            gpu_memory_utilization=args.vllm_gpu_util,
            topk=args.topk_distil,
            max_model_len=args.max_seq_len + 128,  # slight headroom over training seq len
        )
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        teacher = None
        tdev = None
    else:
        if args.single_gpu:
            log.info(f"Loading teacher ({args.teacher}) with device_map='auto'...")
            teacher = AutoModelForCausalLM.from_pretrained(
                args.teacher, dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True,
                **attn_kwargs,
            )
            tdev = teacher.device if hasattr(teacher, 'device') else torch.device("cuda:0")
        else:
            log.info(f"Loading teacher ({args.teacher}) on GPU {args.teacher_gpu}...")
            teacher = AutoModelForCausalLM.from_pretrained(
                args.teacher, dtype=torch.bfloat16,
                device_map={"": args.teacher_gpu},
                trust_remote_code=True,
                **attn_kwargs,
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
            trust_remote_code=True, device_map="auto",
            **attn_kwargs,
        )
        sdev = student.device if hasattr(student, 'device') else torch.device("cuda:0")
    else:
        log.info(f"Loading student ({args.student}) on GPU {args.student_gpu}...")
        student = AutoModelForCausalLM.from_pretrained(
            args.student, dtype=torch.bfloat16,
            trust_remote_code=True,
            **attn_kwargs,
        ).to(f"cuda:{args.student_gpu}")
        sdev = torch.device(f"cuda:{args.student_gpu}")

    # Optional LoRA/DoRA wrap (HF peft). Must happen BEFORE optimizer init so
    # only adapter params get AdamW state.
    if args.lora:
        from peft import LoraConfig, get_peft_model
        targets = [t.strip() for t in args.lora_target.split(",") if t.strip()]
        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
            use_dora=args.use_dora,
            use_rslora=args.use_rslora,
        )
        student = get_peft_model(student, lora_cfg)
        # Required so gradients flow through frozen embeddings when
        # gradient_checkpointing is enabled.
        if hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()
        log.info(f"  LoRA enabled: rank={args.lora_rank} alpha={args.lora_alpha} "
                 f"targets={targets} dora={args.use_dora} rslora={args.use_rslora}")

    student.train()
    student.gradient_checkpointing_enable()
    log.info("  Gradient checkpointing enabled (trades compute for ~50%% less activation memory)")
    n_params = sum(p.numel() for p in student.parameters())
    n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    log.info(f"  Student: {n_params:,} params ({n_trainable:,} trainable), "
             f"VRAM: {torch.cuda.memory_allocated(sdev)/1e9:.1f}GB")

    # Log optimization settings
    log.info(f"  Batch size: {args.batch_size}, Top-k distil: {args.topk_distil or 'off (full vocab)'}")

    # -- Optimizer & scheduler -------------------------------------------------
    # fused=True on CUDA keeps exp_avg / exp_avg_sq in fp32 regardless of param
    # dtype, avoiding the bf16-quantization-to-zero update issue at small LR
    # (Day 9 Exp 11 saw this at LR=1e-6 with bf16 params). A fallback is kept
    # for the rare case where fused kernels aren't available.
    use_fused = torch.cuda.is_available() and not args.no_fused_adam
    try:
        optimizer = AdamW(
            [p for p in student.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay,
            fused=use_fused,
        )
        log.info(f"  Optimizer: AdamW(fused={use_fused})")
    except (TypeError, RuntimeError) as e:
        log.warning(f"  fused AdamW unavailable ({e}); falling back to non-fused")
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
            log.info("  Loaded %d prior rows from %s", len(train_history), metrics_csv_path)

    # -- Prompt sampler --------------------------------------------------------
    if getattr(args, "prompts_file", None):
        sampler = JSONLPromptSampler(args.prompts_file, seed=args.seed)
    else:
        sampler = RandomPromptSampler(seed=args.seed)
    cached_texts = None
    best_train_kl = float("inf")

    if teacher is not None:
        vocab_size = teacher.config.vocab_size if hasattr(teacher.config, 'vocab_size') else 248320
    elif hasattr(student, 'config'):
        vocab_size = getattr(student.config, 'vocab_size', 248320)
    else:
        vocab_size = 248320

    # -- Training loop ---------------------------------------------------------
    log.info("=" * 60)
    log.info("Starting training (v2 — batched)")
    log.info("=" * 60)
    log.info(f"  LR: {args.lr}, Warmup: {args.warmup_steps}, Prompts/step: {args.prompts_per_step}")
    log.info(f"  Seq len: {args.max_seq_len}, KL from pos {args.kl_start_pos}")
    log.info(f"  Batch size: {args.batch_size}")
    log.info(f"  Seed: {args.seed}, Resample every: {args.resample_every} step(s)")
    if args.topk_distil > 0:
        log.info(f"  Top-k distillation: k={args.topk_distil} (transferring {args.topk_distil}/{vocab_size} logits)")
    if args.max_steps > 0:
        log.info(f"  Max steps: {args.max_steps}")
    log.info("")

    while True:
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

        t0 = time.time()

        # Sample prompts
        if cached_texts is None or global_step % args.resample_every == 0:
            cached_texts = sampler.sample(args.prompts_per_step)
            if not cached_texts:
                log.warning("No prompts sampled, stopping.")
                break

        # ── Batched tokenization ──────────────────────────────────────────
        batch_enc = tokenizer(
            cached_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_len,
        )
        input_ids = batch_enc["input_ids"]          # [N, L]
        attention_mask = batch_enc["attention_mask"]  # [N, L]

        # Filter: keep only sequences long enough for KL
        seq_lens = attention_mask.sum(dim=1)
        keep = seq_lens > args.kl_start_pos + 10
        input_ids = input_ids[keep]
        attention_mask = attention_mask[keep]

        if input_ids.shape[0] == 0:
            log.warning(f"Step {global_step}: all prompts too short, resampling...")
            cached_texts = None
            continue

        n_samples = input_ids.shape[0]
        optimizer.zero_grad()
        total_loss = 0.0
        n_batches = 0

        # ── Micro-batched forward passes ──────────────────────────────────
        for i in range(0, n_samples, args.batch_size):
            mb_ids = input_ids[i:i + args.batch_size]
            mb_mask = attention_mask[i:i + args.batch_size]

            # Teacher forward (no grad)
            with torch.no_grad():
                if vllm_teacher is not None:
                    # vLLM path: returns logits directly on student device
                    t_logits = vllm_teacher.get_logits(
                        mb_ids, mb_mask, vocab_size, sdev,
                    )
                else:
                    # HuggingFace path
                    t_out = teacher(
                        mb_ids.to(tdev),
                        attention_mask=mb_mask.to(tdev),
                    )

                    if args.topk_distil > 0:
                        topk_vals, topk_idx = topk_compress(t_out.logits, args.topk_distil)
                        t_logits = topk_decompress(
                            topk_vals, topk_idx, vocab_size, sdev,
                        )
                        del topk_vals, topk_idx
                    else:
                        t_logits = t_out.logits.to(sdev)
                    del t_out

            # Student forward (with grad)
            s_out = student(
                mb_ids.to(sdev),
                attention_mask=mb_mask.to(sdev),
            )
            s_logits = s_out.logits
            del s_out

            # Masked KL loss
            loss = batched_kl_loss(
                s_logits, t_logits,
                mb_mask.to(sdev),
                start_pos=args.kl_start_pos,
                kl_mode=args.kl_mode,
                kl_topk=args.kl_topk,
            )

            # Scale loss by number of micro-batches for gradient accumulation
            n_total_batches = math.ceil(n_samples / args.batch_size)
            (loss / n_total_batches).backward()

            total_loss += loss.item()
            n_batches += 1

            del t_logits, s_logits, loss, mb_ids, mb_mask

        avg_kl = total_loss / max(n_batches, 1)
        if not math.isfinite(avg_kl):
            log.error("Non-finite training KL; stopping.")
            optimizer.zero_grad(set_to_none=True)
            break

        torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        global_step += 1

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]
        log.info(f"Step {global_step} | KL: {avg_kl:.4f} | LR: {lr:.2e} | "
                 f"{elapsed:.1f}s ({n_samples/elapsed:.1f} samp/s) | total_sampled: {sampler.total_sampled:,}")

        if not args.no_wandb:
            import wandb
            wandb.log({"train/kl": avg_kl, "train/lr": lr,
                       "perf/step_time": elapsed, "perf/samples_per_sec": n_samples / elapsed,
                       "data/total_sampled": sampler.total_sampled}, step=global_step)

        if not args.no_train_plots:
            metric_row = {
                "step": global_step,
                "kl": f"{avg_kl:.6f}",
                "lr": f"{lr:.10e}",
                "step_time_s": f"{elapsed:.4f}",
                "samples_per_sec": f"{(n_samples / elapsed):.4f}" if elapsed > 0 else "0",
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

    save_student_checkpoint(
        student, tokenizer, optimizer, args, global_step,
        "final", sampler.total_sampled,
    )
    if best_train_kl < float("inf"):
        log.info(
            "Training complete at step %d. For eval, prefer best_train_kl/ if train KL spiked "
            "(best train KL was %.4f; see final/ for last step only).",
            global_step, best_train_kl,
        )
    else:
        log.info("Training complete at step %d.", global_step)

    if not args.no_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
