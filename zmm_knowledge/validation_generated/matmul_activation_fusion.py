"""
Generated validation kernel: GEMM with fused activation on Ascend 910B
Retrieved knowledge:
- runbooks/compute-pipeline/traj-matmul-split.md           -> 512B alignment, BLOCK_M/N/K tiling
- runbooks/compute-pipeline/instances/traj-exp-matmul-vec-parallel-001.md -> matmul vector parallel + activation fusion
- runbooks/compute-pipeline/traj-activation-compute.md     -> activation fusion in fp32 accumulator
"""
import triton
import triton.language as tl
import torch


def leaky_relu_custom(x):
    # placeholder: real kernel would define this inline or import dialect helper
    return tl.where(x >= 0, x, 0.01 * x)


@triton.jit
def matmul_activation_fusion_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # grouped ordering for L2 reuse
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # block pointers; K0=256 for 512B alignment on fp16
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # fused activation on fp32 accumulator before downcast
    acc = leaky_relu_custom(acc)
    c = acc.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(
        c_ptrs,
        c,
        mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N),
    )


def matmul_activation_fusion(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: [M, K], b: [K, N]"""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    matmul_activation_fusion_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=256,
        GROUP_SIZE=8,
        num_warps=4,
    )
    return c


if __name__ == "__main__":
    M, K, N = 1024, 1024, 1024
    a = torch.randn(M, K, dtype=torch.float16, device="cpu")
    b = torch.randn(K, N, dtype=torch.float16, device="cpu")
    c = matmul_activation_fusion(a, b)
    print("GEMM+activation output shape:", c.shape)
