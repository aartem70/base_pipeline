"""
Phase 1b sanity check: verify our saved checkpoints load correctly under
prod's trust_remote_code=False load path and produce the same logits as
under trust_remote_code=True.

Usage:
    python runs/check_checkpoint_load.py <checkpoint_dir> [--gpu N]
"""
import argparse, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--gpu", type=int, default=2)
    args = ap.parse_args()
    dev = f"cuda:{args.gpu}"

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=False)
    ids = tok("The mitochondria is the powerhouse of", return_tensors="pt").input_ids.to(dev)

    results = {}
    for label, trc in [("trc_true", True), ("trc_false", False)]:
        print(f"\n=== loading with trust_remote_code={trc} ===", flush=True)
        try:
            m = AutoModelForCausalLM.from_pretrained(
                args.checkpoint, dtype=torch.bfloat16, trust_remote_code=trc,
            ).to(dev)
            m.eval()
            with torch.no_grad():
                logits = m(ids).logits.float()
            print(f"  class: {type(m).__name__}")
            print(f"  logits shape: {tuple(logits.shape)}")
            print(f"  last-token top-5 ids: {logits[0,-1].topk(5).indices.tolist()}")
            results[label] = logits[0, -1].cpu()
            del m
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}")
            results[label] = None

    if results["trc_true"] is not None and results["trc_false"] is not None:
        diff = (results["trc_true"] - results["trc_false"]).abs().max().item()
        print(f"\n=== comparison ===")
        print(f"  max abs diff on last-token logits: {diff:.6e}")
        if diff < 1e-3:
            print("  ✓ IDENTICAL (prod load path is safe)")
        else:
            print("  ✗ DIFFERENT — prod may mis-score this checkpoint!")
    elif results["trc_false"] is None:
        print(f"\n=== comparison ===")
        print("  ✗ trust_remote_code=False LOAD FAILED — prod would fail to load this submission")
        sys.exit(1)


if __name__ == "__main__":
    main()
