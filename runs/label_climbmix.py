"""
Fast multi-GPU LLM-labeler for ClimbMix.

Design:
  - Pre-built byte-offset indices (see index_shards.py) for random access.
  - One process per GPU (launched via bash xargs or just repeated invocations),
    each handles a disjoint slice of doc_ids.
  - Batched single-forward letter-choice classification.

Usage (single GPU):
  python label_climbmix.py --shards 0-11 --n 2000 --gpu 0 \
      --out pilot_2k.parquet

Usage (8-GPU split, launched by wrapper script):
  for g in 0 1 2 3 4 5 6 7; do
    python label_climbmix.py --shards 0-11 --n 200000 --gpu $g \
        --num-shards 8 --shard-id $g --out bulk_200k_g${g}.parquet &
  done
  wait
  python -c "import pandas as pd, glob; \
      pd.concat([pd.read_parquet(p) for p in sorted(glob.glob('bulk_200k_g*.parquet'))]).to_parquet('bulk_200k.parquet')"
"""
import argparse, json, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

RAW_DIR = Path("/root/base_pipeline/caches/climbmix_raw")
IDX_DIR = Path("/root/base_pipeline/caches/climbmix_inspect")

CATEGORIES = [
    ("A", "narrative", ""),
    ("B", "dialogue", ""),
    ("C", "qa", ""),
    ("D", "howto", ""),
    ("E", "code", ""),
    ("F", "math", ""),
    ("G", "news", ""),
    ("H", "encyclopedic", ""),
    ("I", "product_desc", ""),
    ("J", "academic", ""),
    ("K", "forum", ""),
    ("L", "opinion", ""),
    ("M", "other", ""),
]

PROMPT_TEMPLATE = """Classify the following document into exactly one category.

Document:
{doc_head}

Categories:
{cats}

Answer with a single letter (A-M) for the best category. Answer:"""


def build_prompt(head: str) -> str:
    cats = "\n".join(f"{L}) {name}" for L, name, _ in CATEGORIES)
    return PROMPT_TEMPLATE.format(doc_head=head, cats=cats)


def read_line(fp, offset: int) -> str:
    fp.seek(offset)
    return fp.readline().decode("utf-8", errors="replace")


def parse_shards(spec):
    out = []
    for p in spec.split(","):
        if "-" in p:
            a, b = p.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(p))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--shards", type=str, default="0-11")
    ap.add_argument("--n", type=int, default=2000, help="total docs to label across all GPUs")
    ap.add_argument("--max-head", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1, help="number of GPU workers")
    ap.add_argument("--shard-id", type=int, default=0, help="this worker's id (0..num-shards-1)")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    shards = parse_shards(args.shards)
    device = f"cuda:{args.gpu}"

    # load all shard indices, build a global list of (shard, idx) pairs
    print(f"[gpu {args.gpu}] loading indices for shards {shards}", flush=True)
    all_pairs = []
    for s in shards:
        idx_path = IDX_DIR / f"part_{s}.idx.npy"
        if not idx_path.exists():
            raise FileNotFoundError(f"Missing index: {idx_path}  (run index_shards.py first)")
        arr = np.load(idx_path)
        for i in range(len(arr)):
            all_pairs.append((s, i))
    print(f"[gpu {args.gpu}] total docs in corpus: {len(all_pairs)}", flush=True)

    # deterministic sample of --n docs
    rng = random.Random(args.seed)
    if args.n < len(all_pairs):
        sample = rng.sample(all_pairs, args.n)
    else:
        sample = list(all_pairs)
    print(f"[gpu {args.gpu}] sampled {len(sample)} docs globally", flush=True)

    # this worker's slice
    if args.num_shards > 1:
        my = [p for i, p in enumerate(sample) if i % args.num_shards == args.shard_id]
    else:
        my = sample
    print(f"[gpu {args.gpu}] worker slice: {len(my)} docs", flush=True)

    # open shard files + read offsets
    open_files = {s: (RAW_DIR / f"part_{s}.jsonl").open("rb") for s in shards}
    offsets = {s: np.load(IDX_DIR / f"part_{s}.idx.npy") for s in shards}

    # materialize doc heads
    print(f"[gpu {args.gpu}] reading doc heads...", flush=True)
    t0 = time.time()
    docs = []  # (shard, idx, head)
    for s, i in my:
        off = int(offsets[s][i])
        line = read_line(open_files[s], off)
        try:
            obj = json.loads(line)
            text = obj.get("text", "")
        except Exception:
            continue
        if len(text) < 100:
            continue
        docs.append((s, i, text[: args.max_head]))
    print(f"[gpu {args.gpu}] materialized {len(docs)} docs in {time.time()-t0:.1f}s", flush=True)
    for fp in open_files.values():
        fp.close()

    # load model
    print(f"[gpu {args.gpu}] loading {args.model}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"[gpu {args.gpu}] loaded in {time.time()-t0:.1f}s", flush=True)

    # letter token ids (single-token for A-M in Qwen3.5 tokenizer)
    letter_ids = [tok.encode(L, add_special_tokens=False)[0] for L, _, _ in CATEGORIES]
    letter_ids_t = torch.tensor(letter_ids, device=device)
    id_to_letter = [L for L, _, _ in CATEGORIES]
    id_to_name = {L: n for L, n, _ in CATEGORIES}

    results = []
    t0 = time.time()
    last_print = t0
    for bstart in range(0, len(docs), args.batch):
        batch = docs[bstart:bstart + args.batch]
        prompts = [build_prompt(h) for _, _, h in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq).to(device)
        with torch.no_grad():
            # logits_to_keep=1 asks HF to only materialize logits for the last
            # token, which cuts memory from [B, L, V] to [B, 1, V] — ~2048x less.
            try:
                out = model(**enc, logits_to_keep=1)
            except TypeError:
                out = model(**enc)
        last_logits = out.logits[:, -1, :]
        cat_logits = last_logits[:, letter_ids_t]
        del out
        probs = F.softmax(cat_logits, dim=-1)
        conf, best = probs.max(dim=-1)
        top2v, top2i = probs.topk(2, dim=-1)
        for (s, i, _), b, c, t2i, t2v in zip(batch, best.tolist(), conf.tolist(), top2i.tolist(), top2v.tolist()):
            L = id_to_letter[b]
            results.append({
                "shard": s,
                "idx": i,
                "letter": L,
                "category": id_to_name[L],
                "confidence": float(c),
                "top2": "/".join(id_to_letter[x] for x in t2i),
                "margin": float(t2v[0] - t2v[1]) if len(t2v) >= 2 else 1.0,
            })
        now = time.time()
        if now - last_print > 15 or bstart + args.batch >= len(docs):
            done = len(results)
            rate = done / max(now - t0, 1e-6)
            eta = (len(docs) - done) / rate if rate > 0 else 0
            print(f"[gpu {args.gpu}] [{done}/{len(docs)}] {rate:.1f} docs/s  ETA {eta:.0f}s", flush=True)
            last_print = now

    # distribution
    from collections import Counter
    dist = Counter(r["category"] for r in results)
    print(f"\n[gpu {args.gpu}] === label distribution ===", flush=True)
    for name, c in dist.most_common():
        print(f"  {name:15s}  {c:>6d}  ({100*c/len(results):5.1f}%)", flush=True)

    import pandas as pd
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
    df.to_parquet(args.out, index=False)
    print(f"[gpu {args.gpu}] wrote {args.out} ({len(df)} rows)", flush=True)


if __name__ == "__main__":
    main()
