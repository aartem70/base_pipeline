#!/usr/bin/env python3
"""
Distil - Pre-Submission Model Checker

Run this BEFORE committing your model to avoid wasting registration fees.
Performs ALL the same checks the validator runs, including anti-cheat detection.

Requirements:
    pip install click huggingface_hub transformers safetensors

For --eval mode (optional):
    pip install torch datasets accelerate  # + CUDA GPU (accelerate required for device_map="auto")

Usage:
    # Basic pre-submission check (no GPU needed):
    python evaluate.py --model-repo user/my-distilled-model

    # Local checkpoint directory (same checks except Hugging Face hub access):
    python evaluate.py --model-repo ./my-checkpoints/final

    # With specific revision:
    python evaluate.py --model-repo user/my-distilled-model --revision abc123

    # Full eval against teacher (requires GPU):
    python evaluate.py --model-repo user/my-distilled-model --eval

    # Eval with custom prompt count:
    python evaluate.py --model-repo user/my-distilled-model --eval --prompts 20

    # Eval with a fixed random seed for reproducibility:
    python evaluate.py --model-repo user/my-distilled-model --eval --seed 42 --prompts 180
"""
import os
import sys
import json
import time
import random
import logging
from pathlib import Path
from types import SimpleNamespace

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger("check_model")

# -- Constants (must match validator) ------------------------------------------
TEACHER_MODEL = "Qwen/Qwen3.5-35B-A3B"
TEACHER_TOTAL_PARAMS_B = 35.0
MAX_PARAM_RATIO = 0.15  # ~5.25B max
BASELINE_VOCAB_SIZE = 248320
MIN_MODEL_BYTES = 500_000_000     # 500MB minimum
MAX_STUDENT_VRAM_GB = 20.0        # Real 4B ~ 8-10GB
MIN_TOKENS_PER_SEC = 50           # Real 4B on B200 does 100+ tok/s
KL_FRAUD_THRESHOLD = 1e-6         # KL <= this = identical to teacher = fraud
FINGERPRINT_COSINE_THRESHOLD = 0.9999  # functional copy detection

VALIDATOR_MIN_COMPLETION_TOKENS = 64
VALIDATOR_MAX_NEW_TOKENS = 512

# Dataset config
CLIMBMIX_DATASET = "karpathy/climbmix-400b-shuffle"
CLIMBMIX_NUM_SHARDS = 6542
CLIMBMIX_TEXT_FIELD = "text"
FINEWEB_DATASET = "HuggingFaceFW/fineweb"


def banner(text: str, char: str = "=", width: int = 60):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def check_pass(name: str, detail: str = ""):
    print(f"  PASS {name}{f' -- {detail}' if detail else ''}")


def check_fail(name: str, detail: str = ""):
    print(f"  FAIL {name}{f' -- {detail}' if detail else ''}")


def check_warn(name: str, detail: str = ""):
    print(f"  WARN {name}{f' -- {detail}' if detail else ''}")


def check_info(name: str, detail: str = ""):
    print(f"  INFO {name}{f' -- {detail}' if detail else ''}")


def sample_random_prompts(
    n: int,
    seed: int = 42,
    dataset_name: str = CLIMBMIX_DATASET,
    min_chars: int = 500,
    max_chars: int = 10000,
) -> list[str]:
    """Sample n random prompts from the dataset.

    Picks a random shard from ClimbMix, loads it, and samples n prompts.
    Falls back to FineWeb streaming if ClimbMix fails.
    """
    from datasets import load_dataset

    rng = random.Random(seed)

    # -- Primary: ClimbMix shard-based sampling --
    try:
        shard_idx = rng.randint(0, CLIMBMIX_NUM_SHARDS - 1)
        shard_file = f"shard_{shard_idx:05d}.parquet"

        print(
            f"[dataset] Sampling {n} prompts from {CLIMBMIX_DATASET} "
            f"(seed={seed}, shard={shard_idx}/{CLIMBMIX_NUM_SHARDS})",
            flush=True,
        )

        ds = load_dataset(
            CLIMBMIX_DATASET,
            data_files=shard_file,
            split="train",
        )

        indices = list(range(len(ds)))
        rng.shuffle(indices)

        prompts: list[str] = []
        for idx in indices:
            text = ds[idx].get(CLIMBMIX_TEXT_FIELD, "")
            if not text or len(text) < min_chars:
                continue
            if len(text) > max_chars:
                text = text[:max_chars]
                last_space = text.rfind(' ')
                if last_space > max_chars // 2:
                    text = text[:last_space]
            prompts.append(text)
            if len(prompts) >= n:
                break

        if len(prompts) >= n:
            print(f"[dataset] Got {len(prompts)} prompts from shard {shard_idx}", flush=True)
            return prompts

        print(f"[dataset] Only got {len(prompts)}/{n} from shard, falling back to FineWeb", flush=True)
    except Exception as e:
        print(f"[dataset] ClimbMix failed ({e}), falling back to FineWeb", flush=True)

    # -- Fallback: FineWeb streaming --
    skip_offset = rng.randint(0, 5_000_000)
    print(
        f"[dataset] Fallback: sampling {n} prompts from {FINEWEB_DATASET} "
        f"(seed={seed}, skip={skip_offset:,})",
        flush=True,
    )

    ds = load_dataset(FINEWEB_DATASET, split="train", streaming=True, name="default")
    ds_shuffled = ds.shuffle(seed=seed, buffer_size=50_000)
    ds_skipped = ds_shuffled.skip(skip_offset)

    prompts = []
    seen = 0
    max_scan = n * 20

    for item in ds_skipped:
        seen += 1
        text = item.get("text", "")
        if not text or len(text) < min_chars:
            continue
        if len(text) > max_chars:
            text = text[:max_chars]
            last_space = text.rfind(' ')
            if last_space > max_chars // 2:
                text = text[:last_space]
        prompts.append(text)
        if len(prompts) >= n:
            break
        if seen > max_scan:
            break

    print(f"[dataset] Got {len(prompts)} prompts (scanned {seen} items)", flush=True)
    return prompts


def format_prompt(text: str, max_chars: int = 10000) -> str:
    """Format a raw pretraining text as a continuation prompt.

    Includes sanitization to prevent malformed inputs from crashing
    the tokenizer or model.
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.replace("\x00", "")
    text = "".join(
        c for c in text
        if c in ("\n", "\t", "\r") or (ord(c) >= 32) or (ord(c) >= 128)
    )

    text = text.strip()
    if not text:
        return ""

    printable_count = sum(1 for c in text if c.isprintable() or c in "\n\t\r")
    if printable_count < len(text) * 0.5:
        return ""

    if len(text) > max_chars:
        cut = text[:max_chars].rfind(". ")
        if cut > max_chars // 3:
            text = text[: cut + 1]
        else:
            text = text[:max_chars]
            last_space = text.rfind(' ')
            if last_space > max_chars // 2:
                text = text[:last_space]

    return text


def apply_min_completion_filter(
    full_sequences,
    teacher_logits_list,
    prompt_lens_list,
    eval_prompts,
    min_completion_tokens: int,
):
    """Drop short teacher completions."""
    if min_completion_tokens <= 0:
        return full_sequences, teacher_logits_list, prompt_lens_list, eval_prompts, 0
    n = len(full_sequences)
    keep_idx = [
        i
        for i in range(n)
        if full_sequences[i].shape[1] - prompt_lens_list[i] >= min_completion_tokens
    ]
    n_filtered = n - len(keep_idx)
    if n_filtered == 0:
        return full_sequences, teacher_logits_list, prompt_lens_list, eval_prompts, 0
    return (
        [full_sequences[i] for i in keep_idx],
        [teacher_logits_list[i] for i in keep_idx],
        [prompt_lens_list[i] for i in keep_idx],
        [eval_prompts[i] for i in keep_idx],
        n_filtered,
    )


def align_student_logits_to_teacher_vocab(student_cont_logits, teacher_logits_for_vocab_ref):
    """Pad/slice student vocab dim to match teacher."""
    import torch

    t_vocab = teacher_logits_for_vocab_ref.shape[-1]
    s_vocab = student_cont_logits.shape[-1]
    if s_vocab < t_vocab:
        pad = torch.full(
            (*student_cont_logits.shape[:-1], t_vocab - s_vocab),
            -1e10,
            device=student_cont_logits.device,
            dtype=student_cont_logits.dtype,
        )
        return torch.cat([student_cont_logits, pad], dim=-1)
    if s_vocab > t_vocab:
        return student_cont_logits[..., :t_vocab]
    return student_cont_logits


@click.command()
@click.option(
    "--model-repo",
    required=True,
    help="Hugging Face repo id (e.g. 'user/my-model') or path to a local model directory",
)
@click.option("--revision", default=None, help="Specific HF revision/commit SHA")
@click.option("--eval", "run_eval", is_flag=True, default=False,
              help="Run a realistic eval against the teacher (requires GPU)")
@click.option("--prompts", type=int, default=20,
              help="Number of prompts for --eval mode (default: 20)")
@click.option("--seed", type=int, default=42,
              help="Random seed for prompt sampling (default: 42)")
@click.option("--teacher-cache", default=None, type=click.Path(),
              help="Path to teacher_cache.pt (skips teacher inference if provided)")
@click.option("--dataset", default=CLIMBMIX_DATASET,
              help="Dataset for eval prompts")
@click.option("--king-repo", default=None,
              help="King model repo for eval comparison (optional)")
@click.option("--king-revision", default=None,
              help="King model revision")
@click.option(
    "--min-completion-tokens",
    type=int,
    default=VALIDATOR_MIN_COMPLETION_TOKENS,
    show_default=True,
    help="Drop prompts where teacher generated fewer new tokens (0 = score all prompts; default: 64)",
)
def main(
    model_repo,
    revision,
    run_eval,
    prompts,
    seed,
    teacher_cache,
    dataset,
    king_repo,
    king_revision,
    min_completion_tokens,
):
    """
    Comprehensive pre-submission checker for distilled models.

    Runs every check the validator performs so you know BEFORE committing
    whether your model will be accepted or rejected.
    """
    from huggingface_hub import model_info as hf_model_info, hf_hub_download, repo_info

    from eval.model_checker import (
        compute_moe_params,
        get_safetensors_param_count,
        is_local_checkpoint_dir,
        local_dir_siblings,
    )

    max_params_b = TEACHER_TOTAL_PARAMS_B * MAX_PARAM_RATIO
    max_model_bytes = max_params_b * 2.2e9

    failures = []
    warnings = []

    local_path = Path(model_repo).expanduser().resolve()
    is_local = is_local_checkpoint_dir(model_repo)
    model_ref = str(local_path) if is_local else model_repo
    if is_local and revision:
        check_warn("--revision", "Ignored when --model-repo is a local directory")
        revision = None

    resolved_revision = revision

    # -- Resolve revision (Hub only) -------------------------------------------
    if not is_local:
        if not resolved_revision:
            try:
                resolved_revision = repo_info(model_repo, repo_type="model").sha
            except Exception as e:
                check_fail("Resolve revision", str(e))
                failures.append(("revision", str(e)))
                _print_summary(failures, warnings)
                sys.exit(1)
    else:
        resolved_revision = None

    banner("PRE-SUBMISSION MODEL CHECKER")
    print(f"  Model: {model_repo}")
    if is_local:
        print(f"  Revision: (local directory)")
    else:
        rev_note = "user-specified" if revision else "pinned from repo latest"
        print(f"  Hub revision: {resolved_revision[:12]}... ({rev_note})")
    print(f"  Max params: {max_params_b:.2f}B")
    if is_local:
        print(f"  Resolved path: {local_path}")

    # ==========================================================================
    # CHECK 1: Repository accessibility
    # ==========================================================================
    banner("CHECK 1: Repository Accessibility")
    if is_local:
        check_pass(
            "Local checkpoint",
            "Skipping Hugging Face hub check — submission still needs a public model repo",
        )
        info = SimpleNamespace(
            siblings=local_dir_siblings(local_path),
            private=False,
            disabled=False,
        )
    else:
        try:
            info = hf_model_info(
                model_repo, revision=resolved_revision, files_metadata=True
            )
            if info.private:
                check_fail("Public access", "Model is PRIVATE -- must be public")
                failures.append(("accessibility", "Model is private"))
            elif info.disabled:
                check_fail("Public access", "Model is DISABLED on HuggingFace")
                failures.append(("accessibility", "Model is disabled"))
            else:
                check_pass("Public access", "Model is publicly accessible")
        except Exception as e:
            err = str(e)
            if "404" in err:
                check_fail("Public access", "Model not found (404)")
                failures.append(("accessibility", "Model not found"))
            elif "403" in err:
                check_fail("Public access", "Model is restricted/gated (403)")
                failures.append(("accessibility", "Model is restricted"))
            else:
                check_fail("Public access", f"Error: {err}")
                failures.append(("accessibility", err))
            _print_summary(failures, warnings)
            sys.exit(1)

    # ==========================================================================
    # CHECK 2: No custom code (security)
    # ==========================================================================
    banner("CHECK 2: Security -- No Custom Code")
    dangerous_files = []
    all_files = []
    for sibling in (info.siblings or []):
        fname = sibling.rfilename
        all_files.append(fname)
        if fname.endswith('.py') and fname != '__init__.py':
            dangerous_files.append(fname)

    if dangerous_files:
        check_fail("No custom code",
                    f"Found Python files: {', '.join(dangerous_files)}. "
                    f"Custom code is NOT allowed -- students must use standard architectures only.")
        failures.append(("custom_code", f"Files: {', '.join(dangerous_files)}"))
    else:
        check_pass("No custom code", "No .py files found in repo")

    # ==========================================================================
    # CHECK 3: Weight file format (safetensors required)
    # ==========================================================================
    banner("CHECK 3: Weight File Format & Sizes")
    total_st_bytes = 0
    total_pt_bytes = 0
    st_files = []
    pt_files = []

    for sibling in (info.siblings or []):
        fname = sibling.rfilename
        fsize = 0
        if hasattr(sibling, 'size') and sibling.size is not None:
            fsize = sibling.size
        elif hasattr(sibling, 'lfs') and sibling.lfs:
            fsize = sibling.lfs.get('size', 0)

        if fname.endswith('.safetensors'):
            total_st_bytes += fsize
            st_files.append((fname, fsize))
        elif fname.endswith('.bin') and 'pytorch_model' in fname:
            total_pt_bytes += fsize
            pt_files.append((fname, fsize))

    check_info("Safetensors files", f"{len(st_files)} files, {total_st_bytes / 1e9:.2f} GB")
    if pt_files:
        check_info("PyTorch .bin files", f"{len(pt_files)} files, {total_pt_bytes / 1e9:.2f} GB")

    # RULE: pytorch_model.bin only -> rejected
    if pt_files and not st_files:
        check_fail("Safetensors required",
                    f"Only pytorch_model.bin found ({len(pt_files)} files, {total_pt_bytes / 1e9:.1f}GB). "
                    f"Convert with: model.save_pretrained('output', safe_serialization=True)")
        failures.append(("format", "Safetensors required, only .bin found"))
    elif st_files:
        check_pass("Safetensors present")

    # RULE: Tiny safetensors + large .bin = fraud attempt
    if st_files and pt_files:
        if total_st_bytes < MIN_MODEL_BYTES and total_pt_bytes > MIN_MODEL_BYTES:
            check_fail("Weight file integrity",
                       f"FRAUD PATTERN: Tiny safetensors ({total_st_bytes:,}B) alongside large "
                       f"pytorch_model.bin ({total_pt_bytes:,}B). Real model hidden in .bin files.")
            failures.append(("fraud_hidden_weights", "Tiny ST + large .bin"))

    # RULE: Minimum file size
    total_weight_bytes = max(total_st_bytes, total_pt_bytes)
    if 0 < total_weight_bytes < MIN_MODEL_BYTES:
        check_fail("Minimum model size",
                    f"Weight files total {total_weight_bytes:,} bytes -- too small for a real model "
                    f"(minimum: {MIN_MODEL_BYTES:,} bytes)")
        failures.append(("min_size", f"Only {total_weight_bytes:,} bytes"))
    elif total_weight_bytes >= MIN_MODEL_BYTES:
        check_pass("Minimum model size", f"{total_weight_bytes / 1e9:.2f} GB")

    # RULE: Maximum file size
    if total_weight_bytes > max_model_bytes:
        check_fail("Maximum model size",
                    f"Weight files total {total_weight_bytes / 1e9:.1f}GB -- too large for "
                    f"{max_params_b:.1f}B params (max ~{max_model_bytes / 1e9:.1f}GB in bf16)")
        failures.append(("max_size", f"{total_weight_bytes / 1e9:.1f}GB exceeds limit"))
    elif total_weight_bytes > 0:
        check_pass("Maximum model size", f"Under {max_model_bytes / 1e9:.1f}GB limit")

    # ==========================================================================
    # CHECK 4: Config analysis (param count, MoE, vocab size)
    # ==========================================================================
    banner("CHECK 4: Model Configuration")
    try:
        if is_local:
            config_path = local_path / "config.json"
            if not config_path.is_file():
                raise FileNotFoundError(f"Missing {config_path}")
        else:
            config_path = hf_hub_download(
                repo_id=model_repo,
                filename="config.json",
                revision=resolved_revision,
            )
        with open(config_path) as f:
            config = json.load(f)

        moe_info = compute_moe_params(config)
        config_total_b = moe_info["total_params"] / 1e9
        config_active_b = moe_info["active_params"] / 1e9

        safetensors_params_b = get_safetensors_param_count(
            model_ref, resolved_revision
        )
        total_params_b = safetensors_params_b if safetensors_params_b > 0 else config_total_b

        check_info("Config total params", f"{config_total_b:.2f}B (from config)")
        if safetensors_params_b > 0:
            check_info("Safetensors params", f"{safetensors_params_b:.2f}B (verified)")
        check_info("Active params", f"{config_active_b:.2f}B")

        if moe_info["is_moe"]:
            check_info("MoE detected",
                       f"{moe_info['num_experts']} experts, "
                       f"{moe_info['num_active_experts']} active/token")

        # RULE: Total params <= max
        if total_params_b > max_params_b:
            check_fail("Parameter count",
                       f"{total_params_b:.2f}B > {max_params_b:.1f}B max (total params, not active)")
            failures.append(("params", f"{total_params_b:.2f}B > {max_params_b:.1f}B"))
        elif total_params_b > 0:
            check_pass("Parameter count", f"{total_params_b:.2f}B <= {max_params_b:.1f}B")

        # RULE: Cross-validate config vs file size
        if total_weight_bytes > 0 and total_params_b > 0:
            estimated_params_from_size = total_weight_bytes / 2e9  # bf16 estimate
            if estimated_params_from_size > total_params_b * 2.5:
                check_fail("Config vs file size",
                           f"Config claims {total_params_b:.2f}B but files suggest "
                           f"~{estimated_params_from_size:.1f}B (bf16). Possible teacher in disguise.")
                failures.append(("cross_validate", "Config/file size mismatch"))
            else:
                check_pass("Config vs file size", "Consistent")

        # RULE: No quantization
        quant_config = config.get("quantization_config", {})
        if quant_config:
            quant_method = quant_config.get("quant_method", "unknown")
            check_fail("No quantization",
                       f"Quantized model detected ({quant_method}). "
                       f"Requires bf16/fp16 architecture distillation.")
            failures.append(("quantized", quant_method))
        else:
            check_pass("No quantization")

        # RULE: Must use Qwen3_5ForConditionalGeneration (vLLM-native architecture)
        archs = config.get("architectures", [])
        model_type = config.get("model_type", "")
        has_preproc = any(
            getattr(s, "rfilename", "") == "preprocessor_config.json"
            for s in (info.siblings or [])
        ) if info else False
        if model_type == "qwen3_5" and "Qwen3_5ForConditionalGeneration" in archs and has_preproc:
            check_pass("Architecture", f"Qwen3_5ForConditionalGeneration (vLLM-native)")
        elif model_type == "qwen3_5" and "Qwen3_5ForConditionalGeneration" in archs:
            check_warn("Architecture",
                       f"Qwen3_5ForConditionalGeneration found but missing preprocessor_config.json. "
                       f"Copy it from Qwen/Qwen3.5-4B.")
        else:
            check_fail("Architecture",
                       f"Must use Qwen3_5ForConditionalGeneration (model_type=qwen3_5). "
                       f"Found: {','.join(archs)} (model_type={model_type}). "
                       f"See Discord announcement for conversion instructions.")
            failures.append(("architecture", f"{','.join(archs)} / {model_type}"))

        # RULE: Vocab size matches teacher
        vocab_size = config.get("vocab_size", 0)
        if not vocab_size:
            vocab_size = config.get("text_config", {}).get("vocab_size", 0)

        if vocab_size != BASELINE_VOCAB_SIZE:
            check_fail("Vocab size",
                       f"{vocab_size} != {BASELINE_VOCAB_SIZE} (teacher). "
                       f"Must use same tokenizer as Qwen3.5-35B-A3B.")
            failures.append(("vocab_size", f"{vocab_size} != {BASELINE_VOCAB_SIZE}"))
        else:
            check_pass("Vocab size", f"{vocab_size} matches teacher")

        # RULE: Nested MoE detection (text_config with hidden experts)
        text_cfg = config.get("text_config", {})
        nested_experts = text_cfg.get("num_local_experts", 0) or text_cfg.get("num_experts", 0)
        top_experts = config.get("num_local_experts", 0) or config.get("num_experts", 0)
        if nested_experts > 1 and not top_experts:
            check_warn("Nested MoE config",
                       f"text_config has {nested_experts} experts but top-level config doesn't. "
                       f"This pattern is flagged as suspicious.")
            warnings.append(("nested_moe", f"text_config.num_experts={nested_experts}"))
        else:
            check_pass("No nested MoE config")

    except Exception as e:
        check_fail("Config analysis", str(e))
        failures.append(("config", str(e)))

    # ==========================================================================
    # CHECK 5: Tokenizer compatibility
    # ==========================================================================
    banner("CHECK 5: Tokenizer Compatibility")
    try:
        from transformers import AutoTokenizer

        teacher_tok = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
        try:
            student_tok = AutoTokenizer.from_pretrained(
                model_ref, revision=resolved_revision, trust_remote_code=False
            )
        except Exception:
            student_tok = AutoTokenizer.from_pretrained(
                model_ref, revision=resolved_revision, trust_remote_code=True
            )

        test_strings = [
            "The quick brown fox jumps over the lazy dog.",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "日本語のテスト文字列です。Unicode handling matters.",
            "KL(P||Q) = sum P(x) log(P(x)/Q(x)) for all x in vocabulary",
        ]

        mismatch = False
        for s in test_strings:
            t_ids = teacher_tok.encode(s)
            s_ids = student_tok.encode(s)
            if t_ids != s_ids:
                check_fail("Tokenizer encoding",
                           f"Mismatch on: '{s[:40]}...' "
                           f"(teacher: {len(t_ids)} tokens, student: {len(s_ids)} tokens)")
                failures.append(("tokenizer", f"Encoding mismatch"))
                mismatch = True
                break

        if not mismatch:
            check_pass("Tokenizer encoding", "All test strings match teacher")

    except Exception as e:
        check_warn("Tokenizer check", f"Could not verify: {e}")
        warnings.append(("tokenizer", str(e)))

    # ==========================================================================
    # CHECK 6: Duplicate hash detection
    # ==========================================================================
    banner("CHECK 6: Model Identity (Duplicate Detection)")
    try:
        from eval.model_checker import compute_model_hash

        model_hash = compute_model_hash(model_ref, resolved_revision)
        if model_hash:
            check_info("Model hash", f"{model_hash[:16]}...")
            hash_file = Path("state/model_hashes.json")
            if hash_file.exists():
                known = json.loads(hash_file.read_text())
                for uid_str, known_hash in known.items():
                    if known_hash == model_hash:
                        check_warn("Duplicate check",
                                   f"Same hash as UID {uid_str} already submitted. "
                                   f"Submitting a copy will be auto-rejected (earlier commit wins).")
                        warnings.append(("duplicate", f"Matches UID {uid_str}"))
                        break
                else:
                    check_pass("Duplicate check", "No known duplicates")
            else:
                check_info("Duplicate check",
                           "Cannot check (no state/model_hashes.json). "
                           "Validator will check on submission.")
        else:
            check_warn("Model hash", "Could not compute hash -- no safetensors found?")
            warnings.append(("hash", "Could not compute"))

    except Exception as e:
        check_warn("Duplicate check", f"Error: {e}")

    # ==========================================================================
    # CHECK 7: Model integrity (weights unchanged)
    # ==========================================================================
    banner("CHECK 7: Model Integrity")
    try:
        from eval.model_checker import verify_model_integrity

        integrity = verify_model_integrity(model_ref, resolved_revision)
        if integrity["pass"]:
            check_pass("Integrity", "Model accessible and weights verifiable")
        else:
            check_fail("Integrity", integrity["reason"])
            failures.append(("integrity", integrity["reason"]))
    except Exception as e:
        check_warn("Integrity", f"Error: {e}")

    # ==========================================================================
    # SUMMARY (pre-GPU checks)
    # ==========================================================================
    _print_summary(failures, warnings)

    if failures:
        print("\n  Your model will be REJECTED by the validator.")
        print("   Fix the issues above before committing to avoid wasting registration fees.")
        sys.exit(1)

    if not run_eval:
        print("\n  All pre-submission checks passed!")
        print("   Your model should be accepted by the validator.")
        print()
        print("   TIP: Run with --eval to test KL divergence on GPU:")
        print(f"   python evaluate.py --model-repo {model_ref} --eval")
        sys.exit(0)

    # ==========================================================================
    # OPTIONAL: GPU-based evaluation
    # ==========================================================================
    banner("GPU EVALUATION", char="#")
    print(f"  Running {prompts}-prompt eval against teacher (seed={seed})")
    if king_repo:
        print(f"  King comparison: {king_repo}")
    print()

    try:
        import torch
        import torch.nn.functional as F

        if not torch.cuda.is_available():
            check_fail("GPU check", "No CUDA GPU available. --eval requires a GPU.")
            sys.exit(1)

        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        check_info("GPU", f"{gpu_name} ({gpu_mem:.0f}GB)")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # -- Load teacher -------------------------------------------------------
        banner("Loading Teacher Model")
        teacher_tok = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)

        # -- Sample prompts (random) --------------------------------------------
        banner("Sampling Eval Prompts")
        check_info("Prompt seed", f"seed={seed}")

        raw_prompts = sample_random_prompts(
            n=prompts,
            seed=seed,
            dataset_name=dataset,
        )
        eval_prompts = []
        for text in raw_prompts:
            formatted = format_prompt(text)
            if formatted:
                eval_prompts.append(formatted)
            if len(eval_prompts) >= prompts:
                break
        print(f"  Sampled {len(eval_prompts)} prompts (format_prompt filtered from {len(raw_prompts)})")

        if len(eval_prompts) == 0:
            check_fail("Prompt sampling", "No valid prompts after filtering")
            sys.exit(1)

        MAX_NEW_TOKENS = 512

        teacher_loaded = False
        teacher_logits_list = []  # continuation-only logits per prompt
        full_sequences = []       # full token sequences (prompt + continuation)
        prompt_lens_list = []     # prompt token length per prompt

        if teacher_cache and Path(teacher_cache).exists():
            print(f"  Loading cached teacher data from {teacher_cache}...")
            try:
                cache_data = torch.load(teacher_cache, map_location="cpu", weights_only=False)
                if (len(cache_data.get("full_sequences", [])) >= len(eval_prompts)
                    and cache_data.get("teacher_logits")
                    and cache_data.get("prompt_lens")):
                    full_sequences = cache_data["full_sequences"][:len(eval_prompts)]
                    teacher_logits_list = cache_data["teacher_logits"][:len(eval_prompts)]
                    prompt_lens_list = cache_data["prompt_lens"][:len(eval_prompts)]
                    teacher_loaded = True
                    print(f"  Loaded {len(full_sequences)} cached prompt sequences")
                else:
                    print(f"  Cache incompatible, will regenerate")
            except Exception as e:
                print(f"  Cache load failed: {e}, will regenerate")

        if not teacher_loaded:
            print(f"  Loading {TEACHER_MODEL}...")
            t0 = time.time()
            try:
                teacher = AutoModelForCausalLM.from_pretrained(
                    TEACHER_MODEL,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                )
            except Exception:
                teacher = AutoModelForCausalLM.from_pretrained(
                    TEACHER_MODEL,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            teacher.eval()
            print(f"  Teacher loaded in {time.time() - t0:.1f}s")
            teacher_vram = torch.cuda.memory_allocated() / 1024**3
            print(f"  Teacher VRAM: {teacher_vram:.1f}GB")

            # -- Generate teacher continuations & extract logits ----------------
            banner("Generating Teacher Continuations + Logits")
            with torch.no_grad():
                for i, prompt_text in enumerate(eval_prompts):
                    prompt_ids = teacher_tok(prompt_text, return_tensors="pt", truncation=False).input_ids.to(teacher.device)
                    prompt_len = prompt_ids.shape[1]

                    output_ids = teacher.generate(
                        prompt_ids, max_new_tokens=VALIDATOR_MAX_NEW_TOKENS,
                        do_sample=False, use_cache=True,
                    )
                    gen_len = output_ids.shape[1] - prompt_len

                    logits = teacher(output_ids).logits.float()
                    cont_logits = logits[:, prompt_len - 1:-1, :]

                    full_sequences.append(output_ids.cpu())
                    teacher_logits_list.append(cont_logits.cpu())
                    prompt_lens_list.append(prompt_len)

                    del logits, cont_logits
                    if (i + 1) % 5 == 0 or i == len(eval_prompts) - 1:
                        print(f"  Teacher: {i + 1}/{len(eval_prompts)} prompts "
                              f"({prompt_len}+{gen_len} tokens)", flush=True)

            print(f"  Teacher logits generated for {len(eval_prompts)} prompts")

            del teacher
            torch.cuda.empty_cache()

        full_sequences, teacher_logits_list, prompt_lens_list, eval_prompts, n_short = (
            apply_min_completion_filter(
                full_sequences,
                teacher_logits_list,
                prompt_lens_list,
                eval_prompts,
                min_completion_tokens,
            )
        )
        if n_short:
            check_info(
                "Min completion filter",
                f"Dropped {n_short} prompts with <{min_completion_tokens} new tokens",
            )
        check_info("Prompts to score", f"{len(full_sequences)} (sampled {prompts}, after filter)")
        if len(full_sequences) == 0:
            check_fail(
                "Prompt scoring",
                f"No prompts left after min_completion_tokens={min_completion_tokens} filter",
            )
            sys.exit(1)

        full_sequences = [s.to("cuda") for s in full_sequences]

        # -- Load student -------------------------------------------------------
        banner("Loading Student Model")
        t0 = time.time()
        try:
            student = AutoModelForCausalLM.from_pretrained(
                model_ref,
                revision=resolved_revision,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=False,
                attn_implementation="flash_attention_2",
            )
        except Exception:
            student = AutoModelForCausalLM.from_pretrained(
                model_ref,
                revision=resolved_revision,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=False,
            )
        student.eval()
        load_time = time.time() - t0
        student_vram = torch.cuda.memory_allocated() / 1024**3
        print(f"  Student loaded in {load_time:.1f}s, VRAM: {student_vram:.1f}GB")

        # ANTI-CHEAT: VRAM check
        banner("CHECK 8: Runtime Anti-Cheat (VRAM)")
        if student_vram > MAX_STUDENT_VRAM_GB:
            check_fail("VRAM usage",
                       f"Student uses {student_vram:.1f}GB (max {MAX_STUDENT_VRAM_GB}GB). "
                       f"A real <=5B model uses ~8-10GB. Likely a larger model in disguise.")
            failures.append(("vram_fraud", f"{student_vram:.1f}GB"))
        else:
            check_pass("VRAM usage", f"{student_vram:.1f}GB (max {MAX_STUDENT_VRAM_GB}GB)")

        # ANTI-CHEAT: Generation speed
        banner("CHECK 9: Runtime Anti-Cheat (Speed)")
        try:
            bench_text = "The quick brown fox"
            bench_ids = teacher_tok(bench_text, return_tensors="pt").input_ids.to(student.device)
            with torch.no_grad():
                t0 = time.time()
                out = student.generate(bench_ids, max_new_tokens=128, do_sample=False)
                gen_time = time.time() - t0
            actual_new = out.shape[1] - bench_ids.shape[1]
            tokens_per_sec = round(actual_new / gen_time, 1)
            print(f"  Generation speed: {tokens_per_sec} tok/s ({actual_new} tokens in {gen_time:.2f}s)")

            if tokens_per_sec < MIN_TOKENS_PER_SEC:
                check_warn("Generation speed",
                           f"{tokens_per_sec} tok/s < {MIN_TOKENS_PER_SEC} minimum. "
                           f"Validator will FLAG this as suspicious.")
                warnings.append(("speed", f"{tokens_per_sec} tok/s"))
            else:
                check_pass("Generation speed", f"{tokens_per_sec} tok/s")
        except Exception as e:
            check_warn("Generation speed", f"Benchmark failed: {e}")

        # -- Score KL divergence (continuation-only, matches production) --------
        banner("CHECK 10: KL Divergence Scoring")

        kl_scores = []
        for i in range(len(eval_prompts)):
            full_seq = full_sequences[i]
            prompt_len = prompt_lens_list[i]

            t_logits = teacher_logits_list[i].to(student.device).float()
            t_log_p = F.log_softmax(t_logits, dim=-1)

            with torch.no_grad():
                s_logits = student(full_seq).logits.float()
                cont_s = s_logits[:, prompt_len - 1:-1, :]

            min_len = min(cont_s.shape[1], t_log_p.shape[1])
            t_lp_slice = t_log_p[:, :min_len, :]
            s_lp_slice = F.log_softmax(cont_s[:, :min_len, :], dim=-1)

            kl_per_pos = F.kl_div(
                s_lp_slice, t_lp_slice, log_target=True, reduction='none'
            ).sum(dim=-1)
            kl_mean = kl_per_pos.mean().item()
            kl_scores.append(kl_mean)

            del s_logits, cont_s, t_logits, t_log_p, t_lp_slice, s_lp_slice, kl_per_pos

            if (i + 1) % 5 == 0:
                running_avg = sum(kl_scores) / len(kl_scores)
                print(f"  Prompt {i + 1}/{len(eval_prompts)}: "
                      f"KL={kl_mean:.6f} (running avg: {running_avg:.6f})", flush=True)

        kl_global = sum(kl_scores) / len(kl_scores)
        import statistics
        kl_std = statistics.stdev(kl_scores) if len(kl_scores) > 1 else 0
        kl_ci_low = kl_global - 1.96 * kl_std / (len(kl_scores) ** 0.5)
        kl_ci_high = kl_global + 1.96 * kl_std / (len(kl_scores) ** 0.5)

        print(f"\n  KL Divergence: {kl_global:.6f}")
        print(f"  95% CI: [{kl_ci_low:.6f}, {kl_ci_high:.6f}]")
        print(f"  Std dev: {kl_std:.6f} over {len(kl_scores)} prompts")

        # ANTI-CHEAT: KL too low = teacher copy
        banner("CHECK 11: KL Fraud Detection")
        if kl_global <= KL_FRAUD_THRESHOLD:
            check_fail("KL fraud check",
                       f"KL={kl_global:.10f} <= {KL_FRAUD_THRESHOLD}. "
                       f"Model is identical to teacher -- automatic DQ.")
            failures.append(("kl_fraud", f"KL={kl_global}"))
        elif kl_global < 0.001:
            check_warn("KL suspiciously low",
                       f"KL={kl_global:.6f} is extremely low. "
                       f"Validator may flag for manual review.")
            warnings.append(("kl_low", f"KL={kl_global:.6f}"))
        else:
            check_pass("KL fraud check", f"KL={kl_global:.6f} (legitimate)")

        # -- Compare against king -----------------------------------------------
        if king_repo:
            banner("KING COMPARISON")
            print(f"  Loading king: {king_repo}...")
            del student
            torch.cuda.empty_cache()

            try:
                king = AutoModelForCausalLM.from_pretrained(
                    king_repo,
                    revision=king_revision,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=False,
                    attn_implementation="flash_attention_2",
                )
            except Exception:
                king = AutoModelForCausalLM.from_pretrained(
                    king_repo,
                    revision=king_revision,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=False,
                )
            king.eval()

            king_kl_scores = []
            with torch.no_grad():
                for i in range(len(eval_prompts)):
                    full_seq = full_sequences[i]
                    prompt_len = prompt_lens_list[i]
                    t_logits = teacher_logits_list[i].to(king.device).float()
                    t_log_p = F.log_softmax(t_logits, dim=-1)
                    k_logits = king(full_seq).logits.float()
                    cont_k = k_logits[:, prompt_len - 1:-1, :]
                    min_len = min(cont_k.shape[1], t_log_p.shape[1])
                    t_lp_slice = t_log_p[:, :min_len, :]
                    k_lp_slice = F.log_softmax(cont_k[:, :min_len, :], dim=-1)
                    kl_per_pos = F.kl_div(
                        k_lp_slice, t_lp_slice, log_target=True, reduction='none'
                    ).sum(dim=-1)
                    king_kl_scores.append(kl_per_pos.mean().item())
                    del t_logits, t_log_p, k_logits, cont_k, t_lp_slice, k_lp_slice, kl_per_pos

            king_kl = sum(king_kl_scores) / len(king_kl_scores)

            del king
            torch.cuda.empty_cache()

            print(f"\n  Your model:  KL = {kl_global:.6f}")
            print(f"  Current king: KL = {king_kl:.6f}")
            diff_pct = (kl_global - king_kl) / king_kl * 100
            if kl_global < king_kl:
                print(f"  Your model BEATS the king by {abs(diff_pct):.2f}%!")
            else:
                print(f"  King is still better by {abs(diff_pct):.2f}%")
                print(f"     You need KL < {king_kl:.6f} to dethrone.")

        _print_summary(failures, warnings, kl=kl_global)

    except Exception as e:
        import traceback
        traceback.print_exc()
        check_fail("GPU evaluation", str(e))
        failures.append(("eval", str(e)))
        _print_summary(failures, warnings)
        sys.exit(1)

    sys.exit(1 if failures else 0)


def _print_summary(failures, warnings, kl=None):
    banner("SUMMARY")
    if failures:
        print(f"  {len(failures)} FAILURE(S) -- model will be REJECTED:")
        for name, detail in failures:
            print(f"     - {name}: {detail}")
    if warnings:
        print(f"  {len(warnings)} WARNING(S) -- may cause issues:")
        for name, detail in warnings:
            print(f"     - {name}: {detail}")
    if not failures and not warnings:
        print(f"  All checks passed!")
    if kl is not None:
        print(f"\n  KL Divergence: {kl:.6f}")
    print()


if __name__ == "__main__":
    main()
