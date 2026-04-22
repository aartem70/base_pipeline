"""Average state_dicts of N saved checkpoints into a single SWA checkpoint."""
import argparse, os, shutil, glob
import torch
from safetensors.torch import load_file, save_file
from collections import OrderedDict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="checkpoint dirs to average")
    ap.add_argument("--output", required=True, help="output dir")
    args = ap.parse_args()

    n = len(args.inputs)
    print(f"Averaging {n} checkpoints into {args.output}")

    # Copy non-weight files (config, tokenizer, etc.) from first checkpoint
    os.makedirs(args.output, exist_ok=True)
    src0 = args.inputs[0]
    for fname in os.listdir(src0):
        if fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
            continue
        s = os.path.join(src0, fname)
        d = os.path.join(args.output, fname)
        if os.path.isfile(s):
            shutil.copy2(s, d)

    # Detect single-file or sharded
    single_file = os.path.join(src0, "model.safetensors")
    if os.path.isfile(single_file):
        # All checkpoints have a single .safetensors file
        print("  single-file safetensors mode")
        avg = None
        for ck in args.inputs:
            sd = load_file(os.path.join(ck, "model.safetensors"))
            if avg is None:
                avg = OrderedDict((k, v.clone().to(torch.float32)) for k, v in sd.items())
            else:
                for k in avg:
                    avg[k].add_(sd[k].to(torch.float32))
            del sd
        for k in avg:
            avg[k].div_(n)
            avg[k] = avg[k].to(torch.bfloat16)   # match original dtype
        save_file(avg, os.path.join(args.output, "model.safetensors"))
    else:
        # Sharded — load index, then each shard
        raise NotImplementedError("sharded checkpoint averaging not implemented; use single-file mode")

    print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
