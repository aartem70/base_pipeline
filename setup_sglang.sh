#!/bin/bash
# Setup SGLang in a separate virtual environment.
# SGLang requires transformers<5 which conflicts with our main venv (transformers>=5).
# This creates an isolated venv just for the SGLang server.
#
# Usage:
#   bash setup_sglang.sh
#
# After setup, launch the server with:
#   bash setup_sglang.sh serve [GPU_ID]

set -e

SGLANG_VENV="${SGLANG_VENV:-$HOME/.sglang_venv}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3.5-35B-A3B}"
GPU_ID="${2:-0}"
PORT="${3:-30000}"

setup() {
    echo "=== Setting up SGLang venv at $SGLANG_VENV ==="

    if [ -d "$SGLANG_VENV" ]; then
        echo "  Venv already exists. To recreate, run: rm -rf $SGLANG_VENV && bash $0"
        echo "  Skipping setup, checking install..."
        source "$SGLANG_VENV/bin/activate"
        python -c "import sglang; print(f'SGLang {sglang.__version__} OK')" 2>/dev/null && {
            echo "  SGLang is installed and working."
            return 0
        }
        echo "  SGLang not working, reinstalling..."
    fi

    python3 -m venv "$SGLANG_VENV"
    source "$SGLANG_VENV/bin/activate"

    echo "=== Installing SGLang + dependencies ==="
    pip install --upgrade pip

    # Install PyTorch (match CUDA version on the machine)
    pip install torch==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126

    # Install SGLang with all dependencies
    pip install "sglang[all]"

    # Verify
    python -c "import sglang; print(f'SGLang {sglang.__version__} installed successfully')"

    echo ""
    echo "=== Setup complete ==="
    echo "To launch the server:"
    echo "  bash setup_sglang.sh serve [GPU_ID]"
}

serve() {
    echo "=== Launching SGLang server ==="
    echo "  Model: $TEACHER_MODEL"
    echo "  GPU: $GPU_ID"
    echo "  Port: $PORT"
    echo ""

    source "$SGLANG_VENV/bin/activate"

    # Set HF cache to ephemeral if available
    if [ -d "/ephemeral" ]; then
        export HF_HOME=/ephemeral/hf_cache
    fi

    CUDA_VISIBLE_DEVICES=$GPU_ID python -m sglang.launch_server \
        --model-path "$TEACHER_MODEL" \
        --port "$PORT" \
        --dtype bfloat16 \
        --trust-remote-code \
        --mem-fraction-static 0.85
}

case "${1:-setup}" in
    setup) setup ;;
    serve) serve ;;
    *)
        echo "Usage: bash setup_sglang.sh [setup|serve] [GPU_ID] [PORT]"
        echo "  setup  - Create venv and install SGLang (default)"
        echo "  serve  - Launch SGLang server for teacher model (default port: 30000)"
        exit 1
        ;;
esac
