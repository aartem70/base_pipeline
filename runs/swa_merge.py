"""
Stochastic weight average multiple checkpoints of same architecture.

python swa_merge.py --ckpts A B C --out OUT_DIR [--weights w1 w2 w3]
"""
import argparse
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="checkpoint dirs to average")
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="Per-checkpoint weights (default uniform)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    n = len(args.ckpts)
    weights = args.weights if args.weights else [1.0 / n] * n
    assert len(weights) == n
    s = sum(weights)
    weights = [w / s for w in weights]

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"[swa] averaging {n} checkpoints with weights {weights}")
    models = []
    for i, ck in enumerate(args.ckpts):
        print(f"  [{i}] {ck}")
        m = AutoModelForCausalLM.from_pretrained(ck, dtype=dtype, trust_remote_code=True)
        models.append(m)

    # averaging on CPU in fp32 to avoid precision loss.
    # Skip params/buffers that don't exist in all checkpoints (e.g. rotary_emb.inv_freq
    # might be registered inconsistently across PyTorch versions).
    base = models[0]
    state_dicts = [m.state_dict() for m in models]

    for name, p in base.named_parameters():
        if not all(name in sd for sd in state_dicts):
            continue
        acc = torch.zeros_like(p, dtype=torch.float32)
        for sd, w in zip(state_dicts, weights):
            acc += w * sd[name].float()
        p.data.copy_(acc.to(p.dtype))

    for name, b in base.named_buffers():
        if not torch.is_floating_point(b):
            continue
        if not all(name in sd for sd in state_dicts):
            continue
        acc = torch.zeros_like(b, dtype=torch.float32)
        for sd, w in zip(state_dicts, weights):
            acc += w * sd[name].float()
        b.data.copy_(acc.to(b.dtype))

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)
    base.save_pretrained(args.out, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(args.ckpts[0], trust_remote_code=True)
    tok.save_pretrained(args.out)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
