"""
Create a tied-embeddings variant of a repackaged model, for controlled A/B.

Takes a directory produced by repackage_qwen3_5_4b.py (which has
tie_word_embeddings=False and an independent lm_head.weight) and writes a
parallel directory that:
  - drops lm_head.weight from the safetensors shards,
  - sets tie_word_embeddings=True (top-level and in text_config),
  - preserves everything else.

Usage:
    python runs/make_tied_variant.py --src /ephemeral/runs/qwen3_5_4b_repackaged \
        --out /ephemeral/runs/qwen3_5_4b_repackaged_tied
"""
import argparse
import json
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Copy non-weight files
    for fname in os.listdir(args.src):
        if fname == "model.safetensors.index.json" or fname.endswith(".safetensors"):
            continue
        shutil.copy(os.path.join(args.src, fname), os.path.join(args.out, fname))

    # Patch config
    cfg_path = os.path.join(args.out, "config.json")
    cfg = json.load(open(cfg_path))
    cfg["tie_word_embeddings"] = True
    if "text_config" in cfg:
        cfg["text_config"]["tie_word_embeddings"] = True
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("[config] tie_word_embeddings=True", flush=True)

    # Rewrite weights without lm_head.weight
    idx_path = os.path.join(args.src, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        idx = json.load(open(idx_path))
        wm = idx["weight_map"]
        # per-shard collect, drop lm_head
        by_shard = {}
        for k, fn in wm.items():
            if k == "lm_head.weight":
                continue
            by_shard.setdefault(fn, []).append(k)
        # re-save each shard without lm_head
        total_bytes = 0
        new_wm = {}
        for i, (shard_fn, keys) in enumerate(sorted(by_shard.items()), 1):
            sd = {}
            with safe_open(os.path.join(args.src, shard_fn), framework="pt") as f:
                for k in keys:
                    sd[k] = f.get_tensor(k)
            out_fn = shard_fn  # keep same naming
            save_file(sd, os.path.join(args.out, out_fn), metadata={"format": "pt"})
            for k in keys:
                new_wm[k] = out_fn
            shard_bytes = sum(t.numel() * t.element_size() for t in sd.values())
            total_bytes += shard_bytes
            print(f"[save] {out_fn}  ({len(sd)} tensors, {shard_bytes/1e9:.2f} GB)",
                  flush=True)
        new_idx = {"metadata": {"total_size": total_bytes}, "weight_map": new_wm}
        with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as f:
            json.dump(new_idx, f, indent=2)
    else:
        # single-file case
        sd = {}
        with safe_open(os.path.join(args.src, "model.safetensors"), framework="pt") as f:
            for k in f.keys():
                if k == "lm_head.weight":
                    continue
                sd[k] = f.get_tensor(k)
        save_file(sd, os.path.join(args.out, "model.safetensors"), metadata={"format": "pt"})
        print(f"[save] model.safetensors  ({len(sd)} tensors)", flush=True)

    print(f"[done] tied variant at {args.out}", flush=True)


if __name__ == "__main__":
    main()
