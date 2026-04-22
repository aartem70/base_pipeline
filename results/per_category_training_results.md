# Per-Category Distil Training Results

**Date:** 2026-04-21 (Day 7 continuation)
**Working dir:** `/root/base_pipeline`
**Goal:** Train `kharchevnykov/distil` on category-specific subsets of ClimbMix to find which content type preserves or improves on distil's eval KL = **0.078296** baseline.

---

## TL;DR

| category       | final eval KL | vs baseline    |
|----------------|---------------|----------------|
| **baseline distil** | **0.0783** | — |
| forum          | 0.5049        | +545% (worst: news; best here) |
| academic       | 0.4745        | +506%          |
| encyclopedic   | 0.6098        | +679%          |
| product_desc   | 0.5917        | +656%          |
| news           | 4.2660        | +5348% (catastrophic) |

**No category beats the 0.0783 baseline at LR=5e-7 full fine-tune.** Confirms the Day 7 "distil sits in a sharp minimum" finding. Per-category differentiation appears as *degradation severity*, not as improvement:

- **Forum** causes the least damage (step_50 KL = 0.4353 — lowest across all categories/steps).
- **Academic** is nearly identical to forum in behavior.
- **Encyclopedic** and **product_desc** degrade slightly more.
- **News** is uniquely catastrophic — 7.4 at step_50, 4.3 at step_200.

---

## Pipeline

### 1. Data acquisition (ClimbMix)

- Downloaded 12 shards of `OptimalScale/ClimbMix` (~200 GB raw JSONL, ~66M docs).
- Built byte-offset indices (`index_shards_fast.py`) for O(1) random access.

### 2. Labeling

Used **Qwen3.5-4B** as zero-shot letter-choice classifier with single-forward-pass logit extraction across 8 GPUs in parallel.

**Taxonomy (v1, 13 categories):** narrative, dialogue, qa, howto, code, math, news, encyclopedic, product_desc, academic, forum, opinion, other.

**Why v1 not a revised taxonomy:** Attempted v2/v3/v4 with fewer, more specific categories (educational/reference/research etc.) — all collapsed to 95% "other" or 68% "howto" depending on letter order. Qwen3.5-4B proved letter-position-sensitive; only v1's category set produced stable distributions across pilot and bulk runs.

**Bulk label distribution (199,511 docs):**

| category | count | pct |
|----------|-------|-----|
| academic     | 118,695 | 59.49% |
| forum        | 24,988  | 12.52% |
| news         | 23,554  | 11.81% |
| product_desc | 14,207  | 7.12% |
| encyclopedic | 12,887  | 6.46% |
| math         | 1,695   | 0.85% |
| qa           | 1,644   | 0.82% |
| howto        | 1,534   | 0.77% |
| code         | 139     | 0.07% |
| dialogue     | 95      | 0.05% |
| narrative    | 53      | 0.03% |
| other        | 19      | 0.01% |
| opinion      | 1       | 0.00% |

"Academic" is a catch-all for educational-adjacent content (textbooks, lesson plans, course listings) rather than strict scholarly writing.

### 3. Per-category training sets (after filters)

Filter: `confidence ≥ 0.5 AND margin ≥ 0.1 AND length ≥ 2560 chars`.

Only 5 categories crossed the 500-doc threshold:

| category     | training docs |
|--------------|---------------|
| academic     | 30,672 |
| news         | 8,605 |
| forum        | 4,947 |
| encyclopedic | 2,077 |
| product_desc | 695 |

Dropped: qa/math/howto (each ~50-200 after filter), code/dialogue/narrative/other/opinion (too rare).

### 4. Training (identical config per category)

- **Student:** `kharchevnykov/distil` (4B params, full fine-tune)
- **Teacher:** `Qwen/Qwen3.5-35B-A3B` (bf16 on single GPU)
- **Loss:** forward KL(teacher || student)
- **LR:** 5e-7 (cosine, 10-step warmup)
- **Batch:** 4
- **Prompts/step:** 60 (fresh sample each step)
- **Steps:** 200
- **Seq len:** 1024

Ran 5 trainings in parallel across 8 GPUs (2 GPUs per training: teacher + student).

### 5. Evaluation

Standard seed-42 60-prompt benchmark (from `karpathy/climbmix-400b-shuffle`, `min_completion_tokens=64` filter → 36 scored), teacher cache built once and reused.

---

## Full results table

**baseline distil:** KL = 0.0783, CI = [0.0550, 0.1096]

All values below are eval KL on the 60-prompt benchmark (mean across 36 prompts with `min_completion_tokens≥64`).

| category      | best_train_kl ckpt | step_50 | step_100 | step_150 | step_200 (final) |
|---------------|-------------------:|--------:|---------:|---------:|-----------------:|
| forum         | 0.0783             | 0.4353  | 0.4747   | 0.4779   | 0.5049           |
| academic      | 0.1236             | 0.4648  | 0.4444   | 0.4751   | 0.4745           |
| encyclopedic  | 0.1072             | 0.5798  | 0.5818   | 0.6231   | 0.6098           |
| product_desc  | 0.5917             | 0.6635  | 0.5997   | 0.6248   | 0.5917           |
| news          | 0.0783             | 7.3976  | 5.8715   | 4.9148   | 4.2660           |

Notes on `best_train_kl` column: this checkpoint corresponds to whatever training step had the lowest train KL.
- For **news** and **forum**, best_train_kl was **step 1** (LR = 5e-8 × 1 update ≈ no measurable weight change in bf16 → matches baseline bit-exactly at KL 0.0783).
- For **academic**, it was **step 2** (slight drift visible: 0.1236).
- For **encyclopedic**, similar early step (0.1072).
- For **product_desc**, `best_train_kl` was a **late-training checkpoint** where train KL reached 0.15; that corresponds to eval KL 0.5917 — a degraded state, not a preserved one. **Low train KL does not imply preserved eval KL.**

---

## Train-KL vs eval-KL trajectories

Final-step **train** KL during training (last-logged step):

| category      | final train KL |
|---------------|---------------:|
| product_desc  | ~0.17          |
| forum         | ~0.65          |
| academic      | ~1.10          |
| news          | ~3.90          |
| encyclopedic  | (not tracked)  |

**Key surprise:** Product_desc had the lowest train KL (0.17) but *higher* eval KL than forum/academic. The student was over-specializing to product-description statistics — which diverge from the eval distribution (karpathy/climbmix-400b-shuffle random prompts).

---

## Interpretation

1. **Sharp-minimum hypothesis fully confirmed.** Matches Day 7's D3 task-vector sweep (±5% deviation from distil collapses KL to 10+). Full fine-tuning at LR=5e-7 moves weights enough that every category training ends at eval KL 0.4–0.7 (or worse for news), none below 0.0783.

2. **Clear per-category content-difficulty gradient** emerges at step_50:
   ```
   forum (0.44) ≲ academic (0.46) ≪ encyclopedic (0.58) ≲ product_desc (0.66) ≪≪ news (7.40)
   ```
   Conversational web text (forum) is closest to distil's pretraining distribution. News is anomalously far.

3. **Training-distribution KL is a poor proxy for eval-distribution KL.** The category with the lowest train loss (product_desc, ~0.17) produced the third-worst eval KL. Training drives the student toward the category's distribution; that distribution is not the eval's.

4. **News is catastrophic because** (a) it contains many dense factual claims, dates, and named entities that may not appear in distil's pretraining; (b) high local entropy in news continuations makes per-token KL explode; (c) consistent with prior observation that news-like content is the hardest OOD regime for distil.

---

## What remains untested

- **Lower LR (5e-8 or 1e-7)** — may show per-category *preservation* differences rather than degradation differences. At small enough LR, some categories may nudge toward 0.0783 rather than past it.
- **LoRA training per category** — prior Exp 7.4 (LoRA r=8 attn-only, LR=5e-7, 200 steps) preserved 0.0783 on random ClimbMix. Per-category LoRA could reveal which category *nudges* rather than *breaks*.
- **qa/math/howto/narrative/dialogue/code/opinion categories** — too rare after filters. Would need 1M-5M labeled docs to get 500 usable.

---

## File locations

- Raw shards: `/root/base_pipeline/caches/climbmix_raw/part_{0..11}.jsonl`
- Shard indices: `/root/base_pipeline/caches/climbmix_inspect/part_{0..11}.idx.npy`
- Bulk labels: `/root/base_pipeline/caches/climbmix_inspect/bulk_200k.parquet`
- Per-category JSONLs: `/root/base_pipeline/caches/climbmix_per_cat/cat_*.jsonl`
- Training checkpoints: `/root/base_pipeline/runs/exp_percat_<cat>_lr5e-7_steps200/`
- Eval JSONs: `/root/base_pipeline/logs/per_cat_evals/*.json`
- Teacher eval cache: `/root/base_pipeline/caches/teacher_cache_60.pt` (16 GB, seed 42)

## Scripts

- `runs/label_climbmix_fast.py` — 8-GPU letter-classifier labeler
- `runs/launch_label_8gpu.sh` — wrapper for parallel labeling
- `runs/build_category_jsonls.py` — filter + materialize per-category JSONLs
- `runs/train_per_category.sh` — launch per-category training
- `runs/eval_per_cat.sh` — evaluate a checkpoint against seed-42 cache
- `train.py` — patched with `--prompts_file` flag + `JSONLPromptSampler` class
