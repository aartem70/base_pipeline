"""Model validation helpers for Hugging Face repos and local checkpoint directories."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def is_local_checkpoint_dir(model_repo: str) -> bool:
    return Path(model_repo).expanduser().resolve().is_dir()


def local_dir_siblings(root: Path) -> list[Any]:
    """Build sibling-like objects matching huggingface_hub file metadata shape."""
    sibs = []
    root = root.resolve()
    for fp in root.rglob("*"):
        if not fp.is_file():
            continue
        try:
            rel = fp.relative_to(root).as_posix()
        except ValueError:
            continue
        s = SimpleNamespace(rfilename=rel, size=fp.stat().st_size, lfs=None)
        sibs.append(s)
    return sibs


def _count_params_in_safetensors_file(path: Path) -> int:
    from safetensors import safe_open

    total = 0
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            shape = f.get_slice(key).get_shape()
            total += int(math.prod(shape)) if shape else 0
    return total


def get_safetensors_param_count(model_repo: str, revision: str | None) -> float:
    """Verified parameter count in billions from safetensors tensors; 0 if unavailable."""
    root = Path(model_repo).expanduser().resolve()
    if root.is_dir():
        total = 0
        for fp in sorted(root.glob("*.safetensors")):
            total += _count_params_in_safetensors_file(fp)
        return total / 1e9 if total else 0.0

    from huggingface_hub import hf_hub_download, list_repo_files

    total = 0
    for fname in list_repo_files(repo_id=model_repo, revision=revision):
        if not fname.endswith(".safetensors"):
            continue
        local = hf_hub_download(
            repo_id=model_repo, filename=fname, revision=revision,
        )
        total += _count_params_in_safetensors_file(Path(local))
    return total / 1e9 if total else 0.0


def compute_moe_params(config: dict) -> dict[str, Any]:
    """Rough param breakdown from config (safetensors count preferred when present)."""

    def pick(key: str, default: int = 0) -> int:
        v = config.get(key)
        if v is not None:
            return int(v)
        tc = config.get("text_config")
        if isinstance(tc, dict) and tc.get(key) is not None:
            return int(tc[key])
        return default

    if "num_parameters" in config:
        n = int(config["num_parameters"])
        return {
            "total_params": n,
            "active_params": n,
            "is_moe": False,
            "num_experts": 0,
            "num_active_experts": 0,
        }

    num_experts = pick("num_local_experts", 0) or pick("num_experts", 0)
    num_active = pick("num_experts_per_tok", 0) or pick("moe_router_topk", 0)
    is_moe = num_experts > 1

    h = pick("hidden_size", 0)
    L = pick("num_hidden_layers", 0)
    V = pick("vocab_size", 0)
    intermediate = pick("intermediate_size", h * 4 if h else 0)
    n_heads = pick("num_attention_heads", 0)
    n_kv = pick("num_key_value_heads", n_heads)
    tie = bool(config.get("tie_word_embeddings", True))

    if not (h and L and V and n_heads):
        return {
            "total_params": 0,
            "active_params": 0,
            "is_moe": is_moe,
            "num_experts": num_experts,
            "num_active_experts": num_active if is_moe else 0,
        }

    head_dim = h // n_heads
    attn_params = (h * h) + 2 * (h * n_kv * head_dim) + (h * h)
    mlp_params = 3 * h * intermediate
    block = attn_params + mlp_params
    embed = V * h * (1 if tie else 2)
    total = int(embed + L * block)

    return {
        "total_params": total,
        "active_params": total,
        "is_moe": is_moe,
        "num_experts": num_experts,
        "num_active_experts": num_active if is_moe else 0,
    }


def compute_model_hash(model_repo: str, revision: str | None) -> str | None:
    h = hashlib.sha256()

    root = Path(model_repo).expanduser().resolve()
    if root.is_dir():
        files = sorted(root.glob("*.safetensors"))
        if not files:
            return None
        for fp in files:
            h.update(fp.name.encode())
            with fp.open("rb") as f:
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    h.update(chunk)
        return h.hexdigest()

    from huggingface_hub import hf_hub_download, list_repo_files

    names = sorted(
        f for f in list_repo_files(repo_id=model_repo, revision=revision)
        if f.endswith(".safetensors")
    )
    if not names:
        return None
    for fname in names:
        h.update(fname.encode())
        local = Path(
            hf_hub_download(repo_id=model_repo, filename=fname, revision=revision)
        )
        with local.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def verify_model_integrity(model_repo: str, revision: str | None) -> dict[str, Any]:
    from safetensors import safe_open

    root = Path(model_repo).expanduser().resolve()
    if root.is_dir():
        if not (root / "config.json").is_file():
            return {"pass": False, "reason": "Missing config.json"}
        st = sorted(root.glob("*.safetensors"))
        if not st:
            return {"pass": False, "reason": "No .safetensors files in directory"}
        for fp in st:
            with safe_open(str(fp), framework="pt") as f:
                list(f.keys())
        return {"pass": True, "reason": ""}

    from huggingface_hub import hf_hub_download, list_repo_files

    try:
        names = [
            f for f in list_repo_files(repo_id=model_repo, revision=revision)
            if f.endswith(".safetensors")
        ]
        if not names:
            return {"pass": False, "reason": "No .safetensors in repo"}
        for fname in names:
            local = hf_hub_download(
                repo_id=model_repo, filename=fname, revision=revision,
            )
            with safe_open(local, framework="pt") as f:
                list(f.keys())
        return {"pass": True, "reason": ""}
    except Exception as e:
        return {"pass": False, "reason": str(e)}
