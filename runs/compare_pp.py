"""
Paired bootstrap comparison of per-prompt KL JSON dumps.

Usage:
    python compare_pp.py --baseline A.json --candidate B.json
"""
import argparse
import json
import random
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    a = json.load(open(args.baseline))
    b = json.load(open(args.candidate))
    pa = a["per_prompt_kl"]
    pb = b["per_prompt_kl"]
    assert len(pa) == len(pb), f"different n: {len(pa)} vs {len(pb)}"
    n = len(pa)

    deltas = [b_i - a_i for a_i, b_i in zip(pa, pb)]  # positive = B worse
    mean_a = sum(pa) / n
    mean_b = sum(pb) / n
    mean_delta = mean_b - mean_a
    std_delta = statistics.stdev(deltas) if n > 1 else 0
    se_delta = std_delta / (n ** 0.5)

    rng = random.Random(args.seed)
    boots = []
    for _ in range(args.bootstrap):
        s = [deltas[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(s) / n)
    boots.sort()
    lo = boots[int(0.025 * args.bootstrap)]
    hi = boots[int(0.975 * args.bootstrap)]
    # one-sided p-value that candidate is worse (delta > 0)
    n_worse = sum(1 for x in boots if x > 0)
    p_worse = n_worse / args.bootstrap
    # one-sided p-value that candidate is better (delta < 0)
    n_better = sum(1 for x in boots if x < 0)
    p_better = n_better / args.bootstrap

    print(f"  baseline : {a['model']}  mean KL = {mean_a:.6f}")
    print(f"  candidate: {b['model']}  mean KL = {mean_b:.6f}")
    print(f"  delta (cand - base): {mean_delta:+.6f}")
    print(f"  paired SE   : {se_delta:.6f}   (std_delta={std_delta:.6f})")
    print(f"  95% paired bootstrap CI: [{lo:+.6f}, {hi:+.6f}]")
    print(f"  P(delta > 0 | bootstrap) = {p_worse:.4f}  (candidate worse)")
    print(f"  P(delta < 0 | bootstrap) = {p_better:.4f}  (candidate better)")
    print(f"  n = {n}")

    # verdict
    if lo > 0:
        verdict = "candidate is WORSE (95% CI above zero)"
    elif hi < 0:
        verdict = "candidate is BETTER (95% CI below zero)"
    else:
        verdict = "indistinguishable (95% CI includes zero)"
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
