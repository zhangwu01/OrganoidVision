#!/bin/bash
# ============================================================
# PyTorch installer — detects platform and installs the
# correct PyTorch build automatically.
# Run after: conda env create -f environment.yaml
# ============================================================

# Detect conda env Python
PYTHON=$(which python)
PIP=$(which pip)

echo "Detecting platform..."

# ── Apple Silicon (M1/M2/M3) ─────────────────────────────────
if [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    echo "Platform: Apple Silicon (MPS)"
    $PIP install torch==2.5.1 torchvision==0.20.1

# ── Linux with CUDA ───────────────────────────────────────────
elif [[ "$(uname)" == "Linux" ]]; then
    # Check CUDA version from nvcc or nvidia-smi
    if command -v nvcc &> /dev/null; then
        CUDA_VER=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    elif command -v nvidia-smi &> /dev/null; then
        CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    else
        CUDA_VER="none"
    fi

    echo "CUDA version detected: $CUDA_VER"

    if [[ "$CUDA_VER" == "12.1"* || "$CUDA_VER" == "12.2"* || "$CUDA_VER" == "12.3"* ]]; then
        echo "Platform: Linux CUDA 12.1"
        $PIP install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
            --index-url https://download.pytorch.org/whl/cu121

    elif [[ "$CUDA_VER" == "12.4"* || "$CUDA_VER" == "12.5"* || "$CUDA_VER" == "12.6"* ]]; then
        echo "Platform: Linux CUDA 12.4"
        $PIP install torch==2.5.1+cu124 torchvision==0.20.1+cu124 \
            --index-url https://download.pytorch.org/whl/cu124

    elif [[ "$CUDA_VER" == "11.8"* ]]; then
        echo "Platform: Linux CUDA 11.8"
        $PIP install torch==2.5.1+cu118 torchvision==0.20.1+cu118 \
            --index-url https://download.pytorch.org/whl/cu118

    else
        echo "Platform: Linux CPU (no CUDA found)"
        $PIP install torch==2.5.1 torchvision==0.20.1 \
            --index-url https://download.pytorch.org/whl/cpu
    fi

# ── Fallback CPU ──────────────────────────────────────────────
else
    echo "Platform: Unknown — installing CPU build"
    $PIP install torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cpu
fi

# ── Verify ───────────────────────────────────────────────────
echo ""
echo "Verifying installation..."
$PYTHON -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('Backend: Apple MPS')
else:
    print('Backend: CPU')
"
echo "Done."
