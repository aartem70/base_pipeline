"""
Build variants of the repackaged Qwen3.5-4B with modified input_layernorm.weight,
to probe whether distil's amplified-norm pattern is the source of its eval-KL edge.

Two modes:
  --mode copy_distil     : replace 32 input_layernorm.weight tensors with distil's values
  --mode scalar --scalar X : multiply repackaged Qwen3.5-4B's input_layernorm.weight by X

Output is a new HF-format dir ready for eval_bootstrap.py. Only safetensors shards
containing modified tensors are rewritten; others are hardlinked/copied unchanged.
"""
import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="repackaged Qwen3.5-4B dir")
    ap.add_argument("--distil", default=None, help="distil snapshot dir (for copy_distil mode)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["copy_distil", "scalar"], required=True)
    ap.add_argument("--scalar", type=float, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Copy non-weight files
    for fname in os.listdir(args.src):
        if fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
            continue
        shutil.copy(os.path.join(args.src, fname), os.path.join(args.out, fname))

    # Load index, figure out which shard holds each input_layernorm
    idx_path = os.path.join(args.src, "model.safetensors.index.json")
    idx = json.load(open(idx_path))
    wm = idx["weight_map"]
    input_norm_keys = sorted([k for k in wm if k.endswith("input_layernorm.weight")])
    print(f"[plan] modifying {len(input_norm_keys)} input_layernorm.weight tensors", flush=True)

    # Load distil norms if needed
    distil_norms = {}
    if args.mode == "copy_distil":
        assert args.distil is not None
        distil_path = os.path.join(args.distil, "model.safetensors")
        with safe_open(distil_path, framework="pt") as f:
            for k in input_norm_keys:
                distil_norms[k] = f.get_tensor(k)
        # sanity
        sample = distil_norms[input_norm_keys[0]]
        print(f"[check] distil L0 input_ln mean={sample.float().abs().mean():.2f} "
              f"max={sample.float().abs().max():.2f}", flush=True)

    # Process each shard
    new_wm = dict(wm)
    shards_to_write = set()
    for k in input_norm_keys:
        shards_to_write.add(wm[k])
    all_shards = set(wm.values())
    for shard_fn in all_shards:
        src_path = os.path.join(args.src, shard_fn)
        dst_path = os.path.join(args.out, shard_fn)
        if shard_fn not in shards_to_write:
            # unchanged shard — hardlink to save space
            try:
                os.link(src_path, dst_path)
            except Exception:
                shutil.copy(src_path, dst_path)
            continue
        # rewrite this shard
        sd = {}
        with safe_open(src_path, framework="pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if k in input_norm_keys:
                    if args.mode == "copy_distil":
                        t = distil_norms[k].to(t.dtype)
                    else:  # scalar
                        t = (t.float() * args.scalar).to(t.dtype)
                sd[k] = t
        save_file(sd, dst_path, metadata={"format": "pt"})
        print(f"[save] {shard_fn}  ({len(sd)} tensors)", flush=True)

    # Write index unchanged (paths same, just modified content)
    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as f:
        json.dump(idx, f, indent=2)

    # Verification
    with safe_open(os.path.join(args.out, "model-00001-of-00003.safetensors"), framework="pt") as f:
        if "model.language_model.layers.0.input_layernorm.weight" in f.keys():
            t = f.get_tensor("model.language_model.layers.0.input_layernorm.weight").float()
            print(f"[verify] L0 input_ln  mean={t.abs().mean():.4f}  max={t.abs().max():.4f}",
                  flush=True)
    print(f"[done] {args.out}", flush=True)


if __name__ == "__main__":
    main()
