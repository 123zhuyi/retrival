#!/bin/bash
set -e

# =============================================================================
# NPU KernelBench container entrypoint.
# Verifies NPU devices are available, prints environment info, then runs CMD.
# =============================================================================

check_npu() {
    if [ -x "$(command -v npu-smi 2>/dev/null)" ]; then
        npu-smi info | head -10
        echo ""
        echo "NPU devices detected via npu-smi"
    elif python3 -c "import torch; print('torch available:', torch.__version__)" 2>/dev/null; then
        python3 -c "
try:
    import torch, torch_npu
    print(f'torch: {torch.__version__}')
    print(f'torch_npu: {torch_npu.__version__}')
    print(f'NPU available: {torch.npu.is_available()}')
    print(f'NPU count: {torch.npu.device_count()}')
except Exception as e:
    print(f'NPU check failed: {e}')
"
    else
        echo "WARNING: No NPU tools detected. NPU kernels will not be available."
    fi
}

print_env() {
    echo ""
    echo "=== NPU KernelBench Environment ==="
    echo "Python: $(python3 --version)"
    echo "Working dir: $(pwd)"
    echo "==================================="
    echo ""
}

check_npu
print_env

# Run the command passed to the container
exec "$@"
