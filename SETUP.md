# Environment Setup Guide

IMPORTANT: Install order matters. vLLM pins its own PyTorch version, and all CUDA
extensions must be built against that same version. Follow the steps in order.

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

## Step 3: Install Python Packages (ORDER MATTERS)

```bash
# 1. Install vLLM FIRST — it pins its own PyTorch version
pip install vllm

# 2. Verify which PyTorch version vLLM installed
python -c "import torch; print(torch.__version__, torch.version.cuda)"

# 3. Install CUDA extensions AFTER vLLM (builds against vLLM's torch)
pip install flash-attn --no-build-isolation --no-cache-dir
pip install causal-conv1d --no-build-isolation --no-cache-dir

# 4. Install remaining dependencies
pip install -r requirements.txt
```

Why this order:
- vLLM controls the PyTorch + CUDA version
- flash-attn and causal-conv1d compile native CUDA code against PyTorch's headers
- If you install them before vLLM, vLLM will upgrade PyTorch and break the compiled binaries
- requirements.txt is last because it won't overwrite existing packages

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

# Check vLLM
python -c "from vllm import LLM; print('vLLM OK')"
```

If `"fast path is not available"` warning is **gone**, everything is working.

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

### vLLM upgraded my PyTorch and broke everything

This is why install order matters. Start over:

```bash
pip uninstall flash-attn causal-conv1d -y
pip install vllm
pip install flash-attn --no-build-isolation --no-cache-dir
pip install causal-conv1d --no-build-isolation --no-cache-dir
```

### `ModuleNotFoundError: No module named 'wheel'`

```bash
pip install wheel
```

### Pre-built wheels not found (building from source fails)

Happens with very new PyTorch+CUDA combos where no one has published wheels yet.
Using vLLM's pinned PyTorch version avoids this since vLLM targets stable releases.

## Quick Reference

| Package | What it does | Required? |
|---------|-------------|-----------|
| `vllm` | Fast teacher inference (2-5x), pins PyTorch | No (use --use_vllm flag) |
| `torch` | ML framework (installed by vLLM) | Yes |
| `flash-attn` | Fast full attention kernels | No (1.5x speedup) |
| `causal-conv1d` | Fast linear attention kernels | No (2x speedup) |
| `flash-linear-attention` | Python wrappers for FLA | Installed with causal-conv1d |
| `transformers` | Model loading | Yes |
| `datasets` | ClimbMix data loading | Yes |

## Without vLLM (simpler setup)

If you don't need vLLM, skip Step 3's vLLM install. However, many GPU rental machines
ship with bleeding-edge PyTorch (e.g., torch 2.11+cu130) that has no pre-built wheels
for CUDA extensions. Downgrade to a stable version first:

```bash
# Downgrade PyTorch to stable CUDA 12.6 build
pip install torch==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# Install remaining dependencies
pip install -r requirements.txt

# Install CUDA extensions (builds against torch 2.6)
pip install flash-attn --no-build-isolation --no-cache-dir
pip install causal-conv1d --no-build-isolation --no-cache-dir
```

Training works without vLLM — just don't pass `--use_vllm`.
