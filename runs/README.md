# `runs/` — pipeline scripts and experiment outputs

Top-level entries fall into three groups: **data prep**, **training**, **evaluation**. Experiment checkpoints live in **`/root/checkpoints/`** (outside the repo entirely; the launcher scripts write there).

## Data prep (ClimbMix corpus + per-category JSONLs)

| Script | Purpose |
|---|---|
| `index_shards.py` | Vectorized numpy byte-offset indexer for `OptimalScale/ClimbMix` shards. Writes `part_<i>.idx.npy` (one int64 per line). ~0.6 GB/s. |
| `inspect_climbmix.py` | Heuristic regex categorizer across 12 shards. **Historical only** — abandoned for LLM-labeling after heuristics hit ~81% "other" ceiling on web text. |
| `label_climbmix.py` | 8-GPU Qwen3.5-4B letter-choice LLM classifier for ClimbMix docs. Single-forward-pass batch scoring, `logits_to_keep=1` for memory. Called by `launch_label_8gpu.sh`. |
| `launch_label_8gpu.sh` | Wrapper: splits labeling across 8 GPUs (1/8 slice each), merges per-GPU parquets at the end. |
| `spotcheck_labels.py` | Prints N sample docs per LLM-assigned category + lowest-confidence labels. Used to validate taxonomy quality. |
| `build_category_jsonls.py` | Materializes per-category training JSONLs from the bulk label parquet, applying confidence/margin/length filters. |

## Training

| Script | Purpose |
|---|---|
| `train_per_category.sh` | Launcher for per-category **full fine-tune** (Day 10 recipe). Takes `<category> <max_steps> <lr> <batch> <prompts/step> <teacher_gpu> <student_gpu>`. |
| `train_lora_longrun.sh` | Launcher for **LoRA r=8 attn-only** long runs (Day 11 recipe, preservation regime). Same arg order as above plus optional `--prompts_file <jsonl>` for per-category. |
| `train_kld_full.py` | Legacy full-vocab full-FT trainer (Day 8). Kept for historical runs — `train.py` is the current canonical trainer. |
| `train_kld_normbias.py` | Legacy norm/bias-only trainer (Day 8/9). |
| `train_kld_peft.py` | Legacy PEFT trainer (Day 8). Superseded by `train.py --lora` flag set. |
| `train_kld_top128.py` | Legacy sparse top-128-renorm-KL trainer on norm/bias subspace (Day 9 Exp 4). |
| `train_seqkd.py`, `train_seqkd_normbias.py` | Legacy sequence-KD trainers. |

## Evaluation

Canonical eval pipeline matches the prod validator's `compute_kl_from_sparse` exactly (see `experiments.log` Day 10/11 or the pinned equation discussion). Default `MIN_COMPLETION_TOKENS=10`, `logprobs_k=128`.

| Script | Purpose |
|---|---|
| `build_eval_cache.py` | Greedy teacher-continuation cache builder. Stores `full_ids`, `prompt_len`, `gen_len`, and top-128 `indices`+`logprobs` per gen position. ~20 MB for 60 prompts. |
| `build_fullseq_teacher_cache.py` | Full-sequence (all-position) teacher logit cache. Used by `eval_bootstrap_region.py` for prompt/continuation/answer region audits (Day 9 Exp 1). |
| `build_train_cache_multishard.py` | Multi-shard teacher cache for training on cached continuations. |
| `eval_bootstrap.py` | Canonical eval — top-128 renorm KL on continuation positions + bootstrap 95% CI. Prod-matching. |
| `eval_bootstrap_region.py` | Region-selectable variant (cont / prompt / answer / all). Used for Day 9 region audit. |
| `eval_per_cat.sh` | Thin wrapper: `eval_per_cat.sh <checkpoint_dir> <gpu_id> [out_json]`. Used for per-category / per-checkpoint evals. |
| `eval_all_ckpts.sh` | Loop: evaluate every `step_*/`, `final/`, `best_train_kl/` subdir under a given run + paired-bootstrap vs baseline. |
| `compare_pp.py` | Paired-bootstrap comparison between two per-prompt KL JSON files (from `eval_bootstrap.py`). |
| `extract_topk_sparse.py` | Convert dense full-vocab cache → top-k sparse cache (for legacy dense caches). |
| `filter_cache_by_repetition.py` | Filter out teacher continuations that loop/repeat (Day 9 Exp 1 diagnostics). |
| `strip_cache.py` | Remove bulky fields from a cache file. |

## Weight surgery / model prep

| Script | Purpose |
|---|---|
| `make_tied_variant.py` | Produce a tied-embedding variant of a model. |
| `norm_surgery.py` | Transfer layernorm scales between checkpoints (Day 9 probing). |
| `repackage_qwen3_5_4b.py` | Repackage Qwen3.5-4B into distil-compatible format. |
| `swa_merge.py` | Stochastic Weight Averaging across checkpoints. |

## Pipeline wrappers

| Script | Purpose |
|---|---|
| `long_pipeline.sh` | Multi-step experiment pipeline (train → eval → compare). Reference end-to-end recipe. |

## Experiment output directories (gitignored)

```
exp_percat_<cat>_lr<lr>_steps<N>/       Day 10 full-FT per-category runs (final, best_train_kl kept)
exp_longlora_<tag>_lr<lr>_steps<N>/     Day 11 LoRA long runs (final, best_train_kl, step_* kept)
```

Each experiment dir contains checkpoint subdirs (`final/`, `best_train_kl/`, `step_<N>/`), `train_config.json`, `train_metrics.csv`, `train_curves.png`.
