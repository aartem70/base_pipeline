# Environment Setup Guide

IMPORTANT: Install order matters. Follow the steps in order.

## Step 1: Basic Setup

```bash
# Clone the repo
git clone https://github.com/aartem70/base_pipeline.git
cd base_pipeline

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Set up HuggingFace token
cp .env.example .env
# Edit .env and paste your HF_TOKEN
```

## Step 2: System Dependencies (CUDA toolkit + C++ compiler)

CUDA extensions need nvcc and g++ to compile. Install these before any pip packages.

```bash
# Install CUDA 12.6 toolkit (provides nvcc)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6

# Install C++ build tools
sudo apt-get install -y build-essential g++-11

# Set environment variables
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=/usr/lib/gcc/x86_64-linux-gnu/11:$CUDA_HOME/bin:$PATH
```

Persist across sessions:

```bash
echo 'export CUDA_HOME=/usr/local/cuda-12.6' >> ~/.bashrc
echo 'export PATH=/usr/lib/gcc/x86_64-linux-gnu/11:$CUDA_HOME/bin:$PATH' >> ~/.bashrc
```

## Step 3: Install Python Packages

Many GPU rental machines ship with bleeding-edge PyTorch (e.g., torch 2.11+cu130)
that has no pre-built wheels for CUDA extensions. Downgrade to a stable version first:

```bash
# 1. Downgrade PyTorch to stable CUDA 12.6 build
pip install torch==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# 2. Install wheel (needed for building CUDA extensions)
pip install wheel

# 3. Install base dependencies
pip install -r requirements.txt

# 4. Install CUDA extensions (builds against torch 2.6)
pip install flash-attn --no-build-isolation --no-cache-dir
pip install causal-conv1d --no-build-isolation --no-cache-dir
```

This gives you all acceleration features (flash attention + fast linear attention).

## Step 4: Verify

```bash
# Check flash attention
python -c "import flash_attn; print('flash-attn OK')"

# Check causal-conv1d (no 'fast path' warning = success)
python -c "
from transformers import AutoModelForCausalLM
import torch
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-4B', dtype=torch.bfloat16, device_map='cuda:0')
x = torch.randint(0, 1000, (1, 32)).cuda()
m(x)
print('OK')
"
```

If `"fast path is not available"` warning is **gone**, everything is working.

## Step 5: SGLang Cache Builder (Optional, Recommended)

SGLang accelerates teacher continuation generation 5-10x vs HF generate().
It runs in a **separate venv** because it needs `transformers<5` while Qwen3.5 needs `>=5`.

### Setup SGLang (one-time)

```bash
bash setup_sglang.sh
```

This creates `~/.sglang_venv` with SGLang installed.

### Build a cache

**Terminal 1** — Start SGLang server (uses the separate venv automatically):

```bash
export HF_HOME=/ephemeral/hf_cache
bash setup_sglang.sh serve 0
# Wait until "The server is fired up" appears
```

**Terminal 2** — Run cache builder (your main venv):

```bash
source .venv/bin/activate
export HF_HOME=/ephemeral/hf_cache

# Build 5000-sample cache
python build_cache_sglang.py \
    --n 5000 \
    --output /ephemeral/cache_5000.pt \
    --logit_gpu 0 \
    --prompt_len 128 \
    --max_new_tokens 512

# When done, Ctrl+C the SGLang server in Terminal 1
```

**Note:** After SGLang generates all continuations, it kills the server isn't needed.
The script then loads the HF teacher on the same GPU for logit extraction.
So you can stop the SGLang server once Phase 1 completes.

### Use the cache for training

```bash
python train.py --student Qwen/Qwen3.5-4B --continuation \
    --cache_continuations /ephemeral/cache_5000.pt \
    --lr 5e-6 --max_steps 4000 --prompts_per_step 10 \
    --save_every 1000 --output_dir /ephemeral/bp-4000 \
    --teacher_gpu 0 --student_gpu 1
```

### Estimated build times (5000 samples)

| GPU | Phase 1 (SGLang gen) | Phase 2 (HF logits) | Total |
|-----|---------------------|---------------------|-------|
| A100 80GB | ~5-8h | ~2-3h | ~7-11h |
| H100 80GB | ~3-5h | ~1-2h | ~4-7h |

### Resume support

If interrupted, re-run the same command. The script saves a `.ckpt` file and
resumes from where it left off.

## Not Yet Available

### vLLM (transformers version conflict)

vLLM has the same `transformers<5` conflict as SGLang, but SGLang has better MoE
performance. Use the SGLang cache builder above instead.

Do NOT install vLLM in the main venv — it will downgrade transformers and break model loading.

### HuggingFace Accelerate (not useful for 2-GPU setup)

HF Accelerate provides multi-GPU data parallelism, but only benefits setups with 4+
GPUs where multiple student replicas can train in parallel. On our 2x A100 80GB setup,
both GPUs are already dedicated (teacher on GPU 0, student on GPU 1), so Accelerate
has nothing to parallelize.

Would become useful with 4+ GPUs (e.g., 4x A100: teacher on GPU 0, 3 student replicas
on GPUs 1-3 with gradient synchronization).

### `undefined symbol` after installing/changing packages

Any time PyTorch version changes (e.g., vLLM upgrades it), all CUDA extensions break.
Rebuild them:

```bash
pip uninstall flash-attn causal-conv1d -y
pip install flash-attn --no-build-isolation --no-cache-dir
pip install causal-conv1d --no-build-isolation --no-cache-dir
```

## Troubleshooting

### `nvcc: cannot execute 'cc1plus'`

nvcc can't find the C++ compiler. Fix:

```bash
export PATH=/usr/lib/gcc/x86_64-linux-gnu/11:$PATH
```

### `CUDA version mismatch` (detected X.X, PyTorch compiled with Y.Y)

Your CUDA toolkit doesn't match PyTorch. Check both:

```bash
python -c "import torch; print(torch.version.cuda)"  # PyTorch's CUDA
nvcc --version                                         # System CUDA
```

Install the CUDA toolkit version that matches PyTorch's. Or if the mismatch is small
(e.g., 12.6 vs 12.8), it usually still works.

### `ImportError: undefined symbol` in flash_attn or causal_conv1d .so files

These were compiled against a different PyTorch version. Rebuild both:

```bash
pip uninstall flash-attn causal-conv1d -y
pip install flash-attn --no-build-isolation --no-cache-dir
pip install causal-conv1d --no-build-isolation --no-cache-dir
```

This happens every time PyTorch version changes (e.g., installing vLLM upgrades torch).

### `ModuleNotFoundError: No module named 'wheel'`

```bash
pip install wheel
```

### `KeyError: 'qwen3_5'` or `Transformers does not recognize this architecture`

Your transformers version is too old. Qwen3.5 needs transformers>=5.x:

```bash
pip install --upgrade transformers
```

### Pre-built wheels not found (building from source fails)

Happens with very new PyTorch+CUDA combos. Solution: downgrade PyTorch to 2.6+cu126 (Step 3).

## Quick Reference

| Package | What it does | Required? |
|---------|-------------|-----------|
| `torch` | ML framework | Yes |
| `transformers` | Model loading (need >=5.x for Qwen3.5) | Yes |
| `datasets` | ClimbMix data loading | Yes |
| `flash-attn` | Fast full attention kernels | No (1.5x speedup) |
| `causal-conv1d` | Fast linear attention kernels | No (2x speedup) |
| `flash-linear-attention` | Python wrappers for FLA | Installed with causal-conv1d |
| `sglang` | Fast teacher generation (separate venv) | Optional (5-10x cache speedup) |
| `vllm` | Fast teacher inference | NOT COMPATIBLE (use SGLang instead) |
| `accelerate` | Multi-GPU data parallelism | NOT USEFUL (needs 4+ GPUs) |
