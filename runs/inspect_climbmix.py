"""
Inspect and categorize ClimbMix shards (v2).

v2 changes over v1:
- Expanded taxonomy: book_description, course_syllabus, forum_post, blog_post
  added; tightened regexes for qa/dialogue/code/math/encyclopedic to reduce
  false positives seen on the 5k pilot.
- Per-shard breakdown tables printed compactly for 12 shards.
- Cross-shard variance (stdev across shards) reported per category — identifies
  which categories are source-correlated.

Usage:
    python inspect_climbmix.py --shards 0-11 --sample 5000
    python inspect_climbmix.py --shards 0-11 --all
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

RAW_DIR = Path("/root/base_pipeline/caches/climbmix_raw")
OUT_DIR = Path("/root/base_pipeline/caches/climbmix_inspect")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --- heuristic categorizer v2 ------------------------------------------------
# Order matters; first match wins.

# CODE: needs actual code-looking tokens beyond a single keyword
RE_CODE_STRONG = re.compile(r"```|^\s*(def |class |import |from \w+ import|#include|public static|function \w+\(|var \w+\s*=|const \w+\s*=)", re.MULTILINE)
RE_CODE_SYMBOLS = re.compile(r"[{};]\s*\n.*[{};]\s*\n", re.MULTILINE)  # multiple lines ending with ; or {

# MATH: LaTeX or dense math symbols
RE_MATH_LATEX = re.compile(r"\$\$|\\frac|\\begin\{(equation|align|matrix)|\\int|\\sum|\\prod|\\mathbb|\\mathcal")
RE_MATH_DENSE = re.compile(r"[≈≠≤≥∑∫∂∇∈∀∃ℝℕℤℚℂ]")

# QA FORUM: explicit Q/A markers, not just rhetorical questions
RE_QA_FORUM = re.compile(r"(^|\n)\s*(Q|Question)\s*[:.-]\s+.{10,}?(\n|$).{0,200}?(^|\n)\s*(A|Answer)\s*[:.-]\s+", re.DOTALL | re.IGNORECASE)
RE_QA_STACKEX = re.compile(r"\b(asked \d|answered \d|votes?|upvoted?|accepted answer)\b", re.IGNORECASE)

# DIALOGUE: quoted speech with said/asked, or multi-turn "Name: ..." pattern
RE_DIALOGUE_QUOTED = re.compile(r'"[^"]{5,}"\s*,?\s*(said|asked|replied|whispered|shouted|muttered|exclaimed)\b', re.IGNORECASE)
RE_DIALOGUE_SCRIPT = re.compile(r"(^|\n)[A-Z][A-Z\s]{2,25}:\s+[A-Z]", re.MULTILINE)  # SCRIPT-STYLE name in caps

# STORY: narrative cues
RE_STORY = re.compile(r"\b(once upon a time|chapter (one|two|three|\d+)|the (old|young) (man|woman|boy|girl))\b|(^|\n)(He|She|They) (walked|ran|smiled|laughed|sighed)\b", re.IGNORECASE)

# HOWTO: numbered steps with action verbs OR tutorial keywords + structural list
RE_HOWTO_STEPS = re.compile(r"(^|\n)\s*(Step\s*\d+[:.]|[1-9]\.\s+(Open|Click|Select|Go to|Press|Enter|Type|Install|Download|Run|Create|Choose|Use|Add|Remove|Set|Configure|Save|Return|Now|First|Next|Finally))", re.MULTILINE | re.IGNORECASE)
RE_HOWTO_TUTORIAL = re.compile(r"\b(this (tutorial|guide) will|in this (tutorial|guide)|follow these steps|step-by-step)\b", re.IGNORECASE)

# NEWS: journalism cues
RE_NEWS_LEAD = re.compile(r"\b(reported|announced|confirmed|according to (sources|officials|reports))\b", re.IGNORECASE)
RE_NEWS_DATELINE = re.compile(r"\b(WASHINGTON|NEW YORK|LONDON|BEIJING|TOKYO|BERLIN|PARIS|MOSCOW|MUMBAI|TORONTO|SYDNEY)\s*[—–-]\s*", re.IGNORECASE)
RE_NEWS_AGENCY = re.compile(r"\b(Reuters|Associated Press|AP|AFP|Bloomberg|Al Jazeera)\b")

# BOOK DESCRIPTION: product/book jacket style
RE_BOOK = re.compile(r"\b(this book|in this book|book description|release date|pages?\s*:\s*\d+|paperback|hardcover|ISBN|edition|author)\b", re.IGNORECASE)

# COURSE SYLLABUS: course description cues
RE_SYLLABUS = re.compile(r"\b(course (description|code|cost|cost:)|prerequisites?|grade level|credit hours|course (objectives|goals|outline))\b", re.IGNORECASE)

# BLOG POST
RE_BLOG = re.compile(r"\b(posted (by|on)|published on|author:|by \w+ \w+\s*[|•]|filed under|tags?:)\b", re.IGNORECASE)

# FORUM POST
RE_FORUM = re.compile(r"\b(posted:|re:|reply|thread|username|member since)\b", re.IGNORECASE)

# LIST: many bullets
RE_LIST = re.compile(r"(^|\n)\s*[-*•]\s+\S", re.MULTILINE)

# ENCYCLOPEDIC: wikipedia-style opening
RE_ENCYC = re.compile(r"^[\w\s]{3,80}\s+(is|was|are|were)\s+(a|an|the)\s+\w+", re.IGNORECASE)


def categorize(text: str) -> str:
    if not text or len(text) < 100:
        return "short"

    head = text[:500]

    # CODE: strong structural signals only
    if RE_CODE_STRONG.search(text) or RE_CODE_SYMBOLS.search(text):
        return "code"

    # MATH
    if RE_MATH_LATEX.search(text) or len(RE_MATH_DENSE.findall(text)) >= 3:
        return "math"

    # QA FORUM: needs actual Q: ... A: structure, not rhetorical questions
    if RE_QA_FORUM.search(text) or RE_QA_STACKEX.search(text):
        return "qa"

    # DIALOGUE
    if RE_DIALOGUE_QUOTED.search(text) or len(RE_DIALOGUE_SCRIPT.findall(text)) >= 2:
        return "dialogue"

    # HOWTO: needs structured steps
    if len(RE_HOWTO_STEPS.findall(text)) >= 2 or (RE_HOWTO_TUTORIAL.search(head) and len(RE_HOWTO_STEPS.findall(text)) >= 1):
        return "howto"

    # NEWS: dateline or agency is high-precision; news lead alone is weaker
    if RE_NEWS_DATELINE.search(head) or RE_NEWS_AGENCY.search(head):
        return "news"
    if RE_NEWS_LEAD.search(head):
        return "news"

    # BOOK DESCRIPTION: common in shard 0
    if RE_BOOK.search(head):
        return "book_desc"

    # COURSE SYLLABUS
    if RE_SYLLABUS.search(head):
        return "syllabus"

    # STORY
    if RE_STORY.search(text):
        return "story"

    # BLOG
    if RE_BLOG.search(head):
        return "blog"

    # FORUM
    if RE_FORUM.search(head):
        return "forum"

    # LIST: >=4 bullets
    if len(RE_LIST.findall(text)) >= 4:
        return "list"

    # ENCYCLOPEDIC: wiki-style definition in opening
    if RE_ENCYC.match(head):
        return "encyclopedic"

    return "other"


# --- shard iteration ---------------------------------------------------------

def iter_shard(path: Path, limit=None):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            try:
                obj = json.loads(line)
                yield i, obj.get("text", "")
            except json.JSONDecodeError:
                continue


def parse_shards(spec: str):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=str, default="0-11", help="e.g. 0-11 or 0,1,5")
    ap.add_argument("--sample", type=int, default=5000)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--print-samples", type=int, default=2)
    ap.add_argument("--out", type=str, default=str(OUT_DIR / "shard_labels.parquet"))
    args = ap.parse_args()

    shards = parse_shards(args.shards)
    limit = None if args.all else args.sample

    rows = []
    per_shard_counts = {}  # shard -> Counter
    cat_samples = defaultdict(list)  # category -> [(shard, i, snippet)]

    for s in shards:
        shard_path = RAW_DIR / f"part_{s}.jsonl"
        if not shard_path.exists():
            print(f"MISSING: {shard_path}", flush=True)
            continue
        t0 = time.time()
        counts = Counter()
        total = 0
        total_chars = 0
        for i, text in iter_shard(shard_path, limit=limit):
            cat = categorize(text)
            counts[cat] += 1
            total += 1
            total_chars += len(text)
            rows.append((s, i, cat, len(text), text[:200]))
            if len(cat_samples[cat]) < args.print_samples:
                cat_samples[cat].append((s, i, text[:350]))
        per_shard_counts[s] = counts
        dt = time.time() - t0
        print(f"shard {s:>2d}: {total} docs in {dt:4.1f}s  avg {total_chars/max(total,1):5.0f} chars", flush=True)

    # ---- per-shard category table ----
    all_cats = sorted({c for cnts in per_shard_counts.values() for c in cnts})
    print("\n=== per-shard category percentages ===", flush=True)
    header = "shard  " + "  ".join(f"{c[:11]:>11s}" for c in all_cats)
    print(header, flush=True)
    mat = {}  # shard -> dict[cat, pct]
    for s, cnts in per_shard_counts.items():
        tot = sum(cnts.values())
        mat[s] = {c: 100 * cnts.get(c, 0) / max(tot, 1) for c in all_cats}
        row = f"{s:>5d}  " + "  ".join(f"{mat[s][c]:>10.2f}%" for c in all_cats)
        print(row, flush=True)

    # mean / stdev per category across shards
    import statistics
    print("\n=== cross-shard stats per category (mean ± stdev, min, max) ===", flush=True)
    print(f"  {'category':15s}  {'mean%':>7s}  {'stdev%':>7s}  {'min%':>7s}  {'max%':>7s}  source-correlated?", flush=True)
    for c in all_cats:
        vals = [mat[s][c] for s in sorted(mat)]
        mu = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        mn, mx = min(vals), max(vals)
        cv = sd / mu if mu > 0.1 else float("inf")  # coefficient of variation
        flag = "YES (non-iid)" if cv > 0.5 else ""
        print(f"  {c:15s}  {mu:>6.2f}  {sd:>6.2f}  {mn:>6.2f}  {mx:>6.2f}   {flag}", flush=True)

    # samples per category
    print("\n=== sample docs per category ===", flush=True)
    for cat in sorted(cat_samples):
        print(f"\n--- {cat} ({sum(per_shard_counts[s].get(cat,0) for s in per_shard_counts)} total) ---", flush=True)
        for s, i, snippet in cat_samples[cat][:args.print_samples]:
            print(f"  [s{s} #{i}] {snippet[:300].replace(chr(10), ' | ')}", flush=True)

    # parquet
    try:
        import pandas as pd
        df = pd.DataFrame(rows, columns=["shard", "idx", "category", "length_chars", "first_200"])
        df.to_parquet(args.out, index=False)
        print(f"\nwrote {args.out} ({len(df)} rows, {os.path.getsize(args.out)/1e6:.1f} MB)", flush=True)
    except Exception as e:
        print(f"parquet write failed: {e}", flush=True)


if __name__ == "__main__":
    main()
