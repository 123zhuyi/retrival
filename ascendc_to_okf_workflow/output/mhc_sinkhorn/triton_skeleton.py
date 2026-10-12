"""
mhc_sinkhorn Triton 骨架（P2 产出，pseudocode 级）

对应 Ascend C：src/mhc/mhc_sinkhorn/op_kernel/arch35/mhc_sinkhorn_arch35.h
语义：Y = Sinkhorn(X)：M0 = rowsoftmax(X)；重复 num_iters 轮 {行归一(除首轮) + 列归一}，分母加 eps。
布局决策：不做 Ascend C 的 gather/scatter 转置，直接 [BLOCK_T, 8, 8] padding tile，axis=2 行内规约、axis=1 列规约。
"""
import torch
import triton
import triton.language as tl

try:
    import triton.runtime.driver as driver
    NUM_CORE = driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"]
except Exception:
    NUM_CORE = 48  # 按目标平台 AIV 核数填写


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": 32}),
        triton.Config({"BLOCK_T": 64}),
        triton.Config({"BLOCK_T": 128}),
    ],
    key=["T", "N"],
)
@triton.jit
def mhc_sinkhorn_kernel(
    x_ptr, y_ptr, norm_ptr, sum_ptr,
    T, N: tl.constexpr,          # N ∈ {4, 6, 8}
    num_iters, eps,
    OUT_FLAG: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_N: tl.constexpr,       # = 8（nAlign）
):
    NUM_BLOCKS = tl.cdiv(T, BLOCK_T)
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)

    for blk in range(pid, NUM_BLOCKS, NUM_CORE):
        offs_t = blk * BLOCK_T + tl.arange(0, BLOCK_T)      # [BLOCK_T]
        offs_r = tl.arange(0, BLOCK_N)                       # 行（矩阵内）
        offs_c = tl.arange(0, BLOCK_N)                       # 列
        t_valid = offs_t < T
        rc_valid = (offs_r < N)[:, None] & (offs_c < N)[None, :]
        mask = t_valid[:, None, None] & rc_valid[None, :, :]

        # CopyIn：连续 load [BLOCK_T, 8, 8]，padding 位 other=0
        ptrs = (x_ptr + offs_t[:, None, None] * N * N
                + offs_r[None, :, None] * N + offs_c[None, None, :])
        m = tl.load(ptrs, mask=mask, other=0.0)              # fp32

        # ---- iter 0：行 softmax 初始化（对应 CalcSoftmax）----
        mx = tl.max(m, axis=2)                               # [BLOCK_T, 8]
        e = tl.exp(m - mx[:, :, None])
        m = e / (tl.sum(e, axis=2) + eps)[:, :, None]        # 行归一 + eps 保护

        nbase = offs_t[:, None, None] * N * BLOCK_N + offs_r[None, :, None] * BLOCK_N + offs_c[None, None, :]
        sbase = offs_t[:, None] * BLOCK_N + offs_c[None, :]

        # ---- iter 0：列归一（对应 CalcColNorm<true>）----
        col = tl.sum(m + eps, axis=1)                        # [BLOCK_T, 8] 列和
        m = m / (col + eps)[:, None, :]
        if OUT_FLAG:
            tl.store(norm_ptr + 0 * T * N * BLOCK_N + nbase, m, mask=mask)
            tl.store(sum_ptr + 0 * T * BLOCK_N + sbase, col + eps,
                     mask=t_valid[:, None] & (offs_c < N)[None, :])

        # ---- iter 1..num_iters-1：行归一 + 列归一 ----
        for it in range(1, num_iters):
            row = tl.sum(m, axis=2) + eps                    # [BLOCK_T, 8]
            m = m / row[:, :, None]
            if OUT_FLAG:
                tl.store(sum_ptr + (2 * it) * T * BLOCK_N + sbase, row,
                         mask=t_valid[:, None] & (offs_c < N)[None, :])
                tl.store(norm_ptr + (2 * it) * T * N * BLOCK_N + nbase, m, mask=mask)
            col = tl.sum(m, axis=1) + eps
            m = m / col[:, None, :]
            if OUT_FLAG:
                tl.store(norm_ptr + (2 * it + 1) * T * N * BLOCK_N + nbase, m, mask=mask)
                tl.store(sum_ptr + (2 * it + 1) * T * BLOCK_N + sbase, col,
                         mask=t_valid[:, None] & (offs_c < N)[None, :])

        # CopyOut：直接 store，无需 scatter 转置
        tl.store(y_ptr + offs_t[:, None, None] * N * N
                 + offs_r[None, :, None] * N + offs_c[None, None, :], m, mask=mask)


def mhc_sinkhorn(x: torch.Tensor, eps: float = 1e-6, num_iters: int = 20,
                 out_flag: bool = False):
    assert x.is_contiguous() and x.dtype == torch.float32
    shape = x.shape
    n = shape[-1]
    assert n in (4, 6, 8) and shape[-2] == n and len(shape) in (3, 4)
    T = shape[0] if len(shape) == 3 else shape[0] * shape[1]

    y = torch.empty_like(x)
    if out_flag:
        norm_out = torch.empty(2 * num_iters * T * n * 8, dtype=torch.float32, device=x.device)
        sum_out = torch.empty(2 * num_iters * T * 8, dtype=torch.float32, device=x.device)
    else:
        norm_out = torch.empty(1, dtype=torch.float32, device=x.device)
        sum_out = torch.empty(1, dtype=torch.float32, device=x.device)

    mhc_sinkhorn_kernel[(NUM_CORE,)](
        x, y, norm_out, sum_out, T, n, num_iters, eps,
        OUT_FLAG=1 if out_flag else 0, BLOCK_N=8,
    )
    return y, norm_out, sum_out
