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
sudo apt-get install -y build-essential g++-11 wheel

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

# 2. Install base dependencies
pip install -r requirements.txt

# 3. Install CUDA extensions (builds against torch 2.6)
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

## vLLM Status (NOT COMPATIBLE YET)

vLLM 0.19 requires `transformers<5`, but Qwen3.5 models need `transformers>=5`.
These two requirements conflict — they cannot coexist in the same environment.

The `--use_vllm` flag in train.py is implemented and ready, but unusable until
vLLM releases a version that supports transformers 5.x. Check future vLLM releases.

For now, training uses HuggingFace for the teacher forward pass, which works fine
with flash-attn and causal-conv1d acceleration.

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

### `ImportError: undefined symbol` in flash_attn_2_cuda.so

flash-attn was compiled against a different PyTorch version. Rebuild:

```bash
pip uninstall flash-attn -y
pip install flash-attn --no-build-isolation --no-cache-dir
```

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
| `vllm` | Fast teacher inference | NOT COMPATIBLE (needs transformers 5.x support) |
