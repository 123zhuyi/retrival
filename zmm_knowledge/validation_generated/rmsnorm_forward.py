"""
Generated validation kernel: RMSNorm forward on Ascend 910B
Retrieved knowledge:
- ops/rmsnorm-rope.md           -> fp32 accumulation, contiguous access
- reference/op-class-normalization.md -> normalization patterns
- runbooks/compute-pipeline/traj-normalization-scheduling.md -> grid-stride + two-pass
"""
import triton
import triton.language as tl
import torch


@triton.jit
def rmsnorm_forward_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    n_rows,
    n_cols,
    eps: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)

    # grid-stride: each program handles multiple rows
    for row_idx in range(pid, n_rows, num_pid):
        col_offsets = tl.arange(0, BLOCK_COLS)
        mask = col_offsets < n_cols

        # Pass 1: compute sum(x^2) in fp32
        sum_sq = tl.zeros((BLOCK_COLS,), dtype=tl.float32)
        for col_start in range(0, n_cols, BLOCK_COLS):
            cols = col_start + col_offsets
            m = cols < n_cols
            x = tl.load(x_ptr + row_idx * n_cols + cols, mask=m, other=0.0).to(tl.float32)
            sum_sq += x * x
        variance = tl.sum(sum_sq, axis=0) / n_cols
        rstd = tl.rsqrt(variance + eps)

        # Pass 2: apply rstd and weight, store
        w = tl.load(w_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
        for col_start in range(0, n_cols, BLOCK_COLS):
            cols = col_start + col_offsets
            m = cols < n_cols
            x = tl.load(x_ptr + row_idx * n_cols + cols, mask=m, other=0.0).to(tl.float32)
            y = x * rstd * w
            tl.store(
                out_ptr + row_idx * n_cols + cols,
                y.to(out_ptr.dtype.element_ty),
                mask=m,
            )


def rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """x: [rows, cols]; weight: [cols]"""
    rows, cols = x.shape
    out = torch.empty_like(x)
    BLOCK_COLS = 128
    grid = (min(rows, 8192),)  # cap grid to avoid launch overhead
    rmsnorm_forward_kernel[grid](
        x, weight, out,
        rows, cols, eps,
        BLOCK_COLS=BLOCK_COLS,
        num_warps=4,
    )
    return out


if __name__ == "__main__":
    rows, cols = 1024, 4096
    x = torch.randn(rows, cols, dtype=torch.float16, device="cpu")
    w = torch.randn(cols, dtype=torch.float16, device="cpu")
    y = rmsnorm_forward(x, w)
    print("RMSNorm output shape:", y.shape)
