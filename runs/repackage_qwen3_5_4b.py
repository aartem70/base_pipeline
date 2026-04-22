"""
Repackage Qwen/Qwen3.5-4B into distil's text-only wrapper format.

Steps:
  1. Load Qwen3.5-4B safetensors shards (keep model.language_model.* only;
     drop model.visual.* and mtp.*).
  2. Untie lm_head: create lm_head.weight as an independent copy of
     model.language_model.embed_tokens.weight.
  3. Save as a single safetensors file (or sharded).
  4. Build a distil-style config.json: text_config nested, no vision_config,
     tie_word_embeddings=False, transformers_version=5.4.0.
  5. Copy tokenizer/chat_template/generation_config from distil for
     drop-in compatibility with our training + eval scripts.

Output: local HF-format directory, loadable by AutoModelForCausalLM.
"""
import argparse
import json
import os
import shutil
import glob

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen_dir", required=True,
                    help="Snapshot dir of Qwen/Qwen3.5-4B")
    ap.add_argument("--distil_dir", required=True,
                    help="Snapshot dir of kharchevnykov/distil (for config/tokenizer template)")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--shard_size_gb", type=float, default=4.5,
                    help="safetensors shard size")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # --- 1. Load weights from Qwen3.5-4B, dropping vision and mtp ---
    idx_path = os.path.join(args.qwen_dir, "model.safetensors.index.json")
    idx = json.load(open(idx_path))
    wm = idx["weight_map"]
    keep_keys = [k for k in wm if "visual" not in k and not k.startswith("mtp")]
    print(f"[load] {len(keep_keys)} tensors kept (drop {len(wm)-len(keep_keys)} "
          f"vision/mtp)", flush=True)

    # open shard files lazily
    shards = {}
    def get_tensor(name):
        fn = wm[name]
        if fn not in shards:
            shards[fn] = safe_open(os.path.join(args.qwen_dir, fn), framework="pt")
        return shards[fn].get_tensor(name)

    state = {}
    for k in keep_keys:
        state[k] = get_tensor(k)
    print(f"[load] loaded {len(state)} tensors", flush=True)

    # --- 2. Untie lm_head ---
    embed = state["model.language_model.embed_tokens.weight"]
    state["lm_head.weight"] = embed.detach().clone()
    print(f"[untie] lm_head.weight = clone of embed_tokens.weight "
          f"shape={embed.shape} dtype={embed.dtype}", flush=True)

    # --- 3. Verify input_layernorm is in normal scale (sanity) ---
    sample = state["model.language_model.layers.0.input_layernorm.weight"].float()
    print(f"[check] L0 input_layernorm: mean={sample.mean():.4f} "
          f"max={sample.max():.4f} — should be ~O(1), NOT thousands", flush=True)
    if sample.mean() > 100:
        raise RuntimeError("Unexpected: input_layernorm appears amplified like distil")

    # --- 4. Save as single safetensors file (or sharded) ---
    # Compute approx size to decide
    total_bytes = sum(t.numel() * t.element_size() for t in state.values())
    print(f"[save] total weight size: {total_bytes/1e9:.2f} GB", flush=True)
    shard_bytes = int(args.shard_size_gb * 1e9)

    if total_bytes <= shard_bytes:
        out_path = os.path.join(args.out, "model.safetensors")
        save_file(state, out_path, metadata={"format": "pt"})
        print(f"[save] {out_path}", flush=True)
    else:
        # sharded save
        shard_files, current_shard, current_bytes = {}, {}, 0
        shard_idx = 1
        shard_name = lambda i, total: f"model-{i:05d}-of-{total:05d}.safetensors"
        # First pass: decide shard boundaries
        ordered = list(state.items())
        shards_list = [[]]
        running = 0
        for k, t in ordered:
            b = t.numel() * t.element_size()
            if running + b > shard_bytes and shards_list[-1]:
                shards_list.append([])
                running = 0
            shards_list[-1].append((k, b))
            running += b
        n_shards = len(shards_list)
        weight_map = {}
        for i, shard_keys in enumerate(shards_list, 1):
            sd = {k: state[k] for k, _ in shard_keys}
            fn = shard_name(i, n_shards)
            out_path = os.path.join(args.out, fn)
            save_file(sd, out_path, metadata={"format": "pt"})
            print(f"[save] {out_path}  ({len(sd)} tensors, "
                  f"{sum(b for _,b in shard_keys)/1e9:.2f} GB)", flush=True)
            for k, _ in shard_keys:
                weight_map[k] = fn
        index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
        with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)
        print("[save] model.safetensors.index.json", flush=True)

    # --- 5. Build distil-style config.json ---
    distil_cfg = json.load(open(os.path.join(args.distil_dir, "config.json")))
    qwen_cfg = json.load(open(os.path.join(args.qwen_dir, "config.json")))

    # start from distil's config, overlay text_config from 4B
    new_cfg = dict(distil_cfg)
    new_cfg["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    new_cfg["model_type"] = "qwen3_5"
    # Take text_config straight from Qwen3.5-4B (it's the unchanged one we want)
    # then force tie_word_embeddings=False and record transformers_version
    tc = dict(qwen_cfg.get("text_config", {}))
    tc["tie_word_embeddings"] = False
    tc["architectures"] = ["Qwen3_5ForCausalLM"]
    new_cfg["text_config"] = tc
    # drop vision_config explicitly (not present in distil)
    new_cfg.pop("vision_config", None)
    new_cfg["transformers_version"] = "5.4.0"
    # top-level tie flag too (some loaders read top-level)
    new_cfg["tie_word_embeddings"] = False

    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(new_cfg, f, indent=2)
    print(f"[save] config.json (distil-style, tie_word_embeddings=False)", flush=True)

    # --- 6. Copy tokenizer + generation_config + chat template from distil ---
    for fname in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                  "generation_config.json"]:
        src = os.path.join(args.distil_dir, fname)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(args.out, fname))
            print(f"[copy] {fname} from distil", flush=True)

    print(f"[done] repackaged Qwen3.5-4B saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
