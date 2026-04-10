# Environment Setup Guide

## Step 1: Basic Setup

```bash
# Clone the repo
git clone https://github.com/aartem70/base_pipeline.git
cd base_pipeline

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install base dependencies
pip install -r requirements.txt

# Set up HuggingFace token
cp .env.example .env
# Edit .env and paste your HF_TOKEN
```

This is enough to run training with the slow (default) attention path.

## Step 2: CUDA Acceleration (recommended)

These packages speed up the linear attention layers in Qwen3.5 by ~2x. They require
a matching CUDA toolkit and C++ compiler to build from source.

### 2a: Downgrade PyTorch to a stable CUDA 12.6 build

Many GPU rental machines ship with bleeding-edge PyTorch+CUDA (e.g., torch 2.11+cu130)
that has no pre-built wheels for any CUDA extensions. Downgrade to a well-supported combo:

```bash
pip install torch==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

### 2b: Install CUDA 12.6 toolkit (provides nvcc compiler)

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6
```

### 2c: Install C++ build tools

```bash
sudo apt-get install -y build-essential g++-11
```

### 2d: Set environment variables

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=/usr/lib/gcc/x86_64-linux-gnu/11:$CUDA_HOME/bin:$PATH
```

Add these to your `~/.bashrc` to persist across sessions:

```bash
echo 'export CUDA_HOME=/usr/local/cuda-12.6' >> ~/.bashrc
echo 'export PATH=/usr/lib/gcc/x86_64-linux-gnu/11:$CUDA_HOME/bin:$PATH' >> ~/.bashrc
```

### 2e: Install CUDA extensions

```bash
# Fast linear attention kernels (fixes "fast path is not available" warning)
pip install causal-conv1d --no-build-isolation

# Flash attention for the full attention layers
pip install flash-attn --no-build-isolation
```

### 2f: Verify

```bash
python -c "
from transformers import AutoModelForCausalLM
import torch
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-4B', dtype=torch.bfloat16, device_map='cuda:0')
x = torch.randint(0, 1000, (1, 32)).cuda()
m(x)
print('OK')
"
```

If `"fast path is not available"` warning is **gone**, you're good.

## Troubleshooting

### `nvcc: cannot execute 'cc1plus'`

nvcc can't find the C++ compiler. Fix:

```bash
export PATH=/usr/lib/gcc/x86_64-linux-gnu/11:$PATH
```

### `CUDA version mismatch` (detected X.X, PyTorch compiled with Y.Y)

Your PyTorch CUDA version doesn't match the installed toolkit. Check both:

```bash
python -c "import torch; print(torch.version.cuda)"  # PyTorch's CUDA
nvcc --version                                         # System CUDA
```

They must match. Either reinstall PyTorch to match your toolkit, or install a different
toolkit version to match PyTorch.

### `ModuleNotFoundError: No module named 'wheel'`

```bash
pip install wheel
```

### Pre-built wheels not found (building from source fails)

This happens with very new PyTorch+CUDA combos. Solution: downgrade PyTorch (Step 2a).

## Quick Reference

| Package | What it does | Required? |
|---------|-------------|-----------|
| `torch` | ML framework | Yes |
| `transformers` | Model loading | Yes |
| `datasets` | ClimbMix data loading | Yes |
| `causal-conv1d` | Fast linear attention kernels | No (2x speedup) |
| `flash-attn` | Fast full attention kernels | No (1.5x speedup) |
| `flash-linear-attention` | Python wrappers for FLA | Installed with causal-conv1d |
| `vllm` | Fast teacher inference | No (future, requires code flag --use_vllm) |
