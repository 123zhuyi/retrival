"""
Abs operator — Triton-Ascend skeleton.

Semantic: y = abs(x) element-wise.

This is a KNOWLEDGE ARTIFACT extracted from the CANN ops-math Abs operator.
It illustrates how the CANN elementwise pipeline (TPipe / ElementwiseSch /
Vec::Abs) maps onto Triton-Ascend primitives.

NOT guaranteed to compile or run on any specific backend.  No benchmarking
has been performed.  The purpose is to capture the design intent so that
code generators can reference the pattern.

References:
  - source-cann-ops-math-abs (full CANN Abs operator analysis)
  - kernel-elementwise-abs (KernelWiki page)
  - technique-tail-mask
  - technique-fp32-accumulation
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Real-dtype abs (float32, float16, int32, etc.)
# ---------------------------------------------------------------------------

@triton.jit
def abs_kernel_real(
    x_ptr,
    y_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DTYPE_X: tl.constexpr,  # e.g. tl.float32
):
    """Elementwise absolute value for real dtypes.

    CANN mapping
    ------------
    - ElementwiseSch<0, AbsDag<dtype, dtype>::OpDag>  →  this kernel
    - Vec::CopyIn<U>                                   →  tl.load
    - Vec::Abs<T>                                      →  tl.abs (or tl.where)
    - Vec::CopyOut<U>                                  →  tl.store
    - KERNEL_TYPE_AIV_ONLY                             →  grid over 1D programs
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.abs(x)                     # or: tl.where(x < 0, -x, x)
    tl.store(y_ptr + offsets, y, mask=mask)


# ---------------------------------------------------------------------------
# BF16 abs (FP32 intermediate)
# ---------------------------------------------------------------------------

@triton.jit
def abs_kernel_bf16(
    x_ptr,
    y_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """BF16 abs with FP32 intermediate.

    CANN mapping
    ------------
    - TILING_KEY_IS(101)
    - AbsDag<bfloat16_t, float>::OpDag
      = CopyIn<bfloat16> → Cast<float, bfloat16, 0> → Abs<float>
                          → Cast<bfloat16, float, 1> → CopyOut<bfloat16>
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x_bf16 = tl.load(x_ptr + offsets, mask=mask, other=0.0)       # CopyIn<BF16>
    x_f32 = x_bf16.to(tl.float32)                                  # Cast<f32, bf16>
    y_f32 = tl.abs(x_f32)                                          # Abs<f32>
    y_bf16 = y_f32.to(tl.bfloat16)                                 # Cast<bf16, f32>
    tl.store(y_ptr + offsets, y_bf16, mask=mask)                   # CopyOut<BF16>


# ---------------------------------------------------------------------------
# Complex abs (interleaved real/imag layout)
# ---------------------------------------------------------------------------

@triton.jit
def abs_kernel_complex(
    x_ptr,
    y_ptr,
    n_elements: tl.constexpr,   # number of complex elements
    BLOCK_SIZE: tl.constexpr,
):
    """Complex abs for interleaved [real, imag, real, imag, ...] layout.

    CANN mapping
    ------------
    - TILING_KEY_IS(103)
    - AbscomplexDag<complex64, float>::OpDag
      = CopyIn<complex64> → AbscomplexCustom<float, complex64> → CopyOut<float>

    The complex input is stored as interleaved 32-bit pairs (for COMPLEX64)
    or 16-bit pairs (for COMPLEX32).  Each complex element produces one
    output element: sqrt(real^2 + imag^2).

    NOTE: This skeleton assumes 32-bit interleaved real/imag for COMPLEX64.
    For COMPLEX32 adjust element sizes accordingly.
    """
    pid = tl.program_id(0)

    # Each complex element occupies 2 storage slots
    storage_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = storage_offsets < n_elements

    # Load interleaved pairs
    real_offsets = storage_offsets * 2
    imag_offsets = storage_offsets * 2 + 1

    real = tl.load(x_ptr + real_offsets, mask=mask, other=0.0)
    imag = tl.load(x_ptr + imag_offsets, mask=mask, other=0.0)

    # sqrt(real^2 + imag^2)
    sq = real * real + imag * imag
    y = tl.sqrt(sq)

    tl.store(y_ptr + storage_offsets, y, mask=mask)


# ---------------------------------------------------------------------------
# Host launcher
# ---------------------------------------------------------------------------

def launch_abs(x: torch.Tensor) -> torch.Tensor:
    """Dispatch to the appropriate abs kernel based on dtype."""
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    y = torch.empty_like(x)

    if x.dtype == torch.bfloat16:
        abs_kernel_bf16[grid](x, y, n_elements, BLOCK_SIZE)
    elif x.dtype in (torch.complex64, torch.complex32):
        y = torch.empty(n_elements, dtype=torch.float32
                        if x.dtype == torch.complex64 else torch.float16,
                        device=x.device)
        abs_kernel_complex[grid](x, y, n_elements, BLOCK_SIZE)
    else:
        abs_kernel_real[grid](x, y, n_elements, BLOCK_SIZE)

    return y


# ---------------------------------------------------------------------------
# NOTE
# ---------------------------------------------------------------------------
# This skeleton has NOT been compiled or benchmarked on any real hardware.
# It is a knowledge artifact only.