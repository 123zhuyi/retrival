---
schema_version: okf.v1
kind: example
type: code_example
source_family: reference
title: Gemm Basic Skeleton
description: 骨架：GEMM（矩阵乘法）基础模板 适用：M≥256, N≥256, K≥1024, fp16/bf16, NN 布局
tags: [code_skeleton, gemm_basic]
confidence: inferred
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/pages/skeletons/gemm_basic.py
created_at: '2026-08-10T02:56:36Z'
updated_at: '2026-08-10T02:56:36Z'
---

sources:
  - url: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/pages/skeletons/gemm_basic.py
    role: primary

> 迁移备注：本卡由 `cannbot-skills_5171/ops/triton-knowledge-retrieval/knowledge/pages/skeletons/gemm_basic.py` 迁移而来，原始为代码骨架，未做内容改写，仅转 OKF frontmatter。

```python
"""
骨架：GEMM（矩阵乘法）基础模板
适用：M≥256, N≥256, K≥1024, fp16/bf16, NN 布局
经验：exp_matmul_001 (40%), skill_tiling_diagonal（M,N≥512 时）
约束检查：[hardware/constraints_cheatsheet]
"""

import torch
import triton
import triton.language as tl


# ── AutoTune 配置 ────────────────────────────────────────────────
# 满足 512B 行宽对齐（fp16: BLOCK_N≥256）
# GROUP_SIZE 至少提供 8/16/32 三档
configs = [
    triton.Config({'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': 256, 'GROUP_SIZE': gs},
                  num_stages=2, num_warps=4)
    for bm in [128, 256]
    for bn in [128, 256]
    for gs in [8, 16, 32]
]


@triton.autotune(configs=configs, key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # ── PID → (m_id, n_id)：对角映射（M,N≥512 时消除 L2 bank conflict）──
    pid = tl.program_id(0)
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    num_blocks_in_group = GROUP_SIZE * num_n_blocks
    group_id = pid // num_blocks_in_group
    first_m = group_id * GROUP_SIZE
    group_m_size = min(num_m_blocks - first_m, GROUP_SIZE)
    m_id = first_m + (pid % group_m_size)
    n_id = (pid % num_blocks_in_group) // group_m_size

    # ── 指针计算 ──────────────────────────────────────────────────
    offs_m = m_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_id * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # ── 累加（fp32） ──────────────────────────────────────────────
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # care_padding=False：当 padding 不影响结果时，避免 MTE2→Vector 同步屏障
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_K),
                    other=0.0, cache_modifier=".cg")
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_K) & (offs_n[None, :] < N),
                    other=0.0, cache_modifier=".cg")
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ── Store ─────────────────────────────────────────────────────
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "K dim mismatch"
    assert a.is_contiguous() and b.is_contiguous(), "Inputs must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # grid_size ≤ num_ai_cores 约束（TRITON_ALL_BLOCKS_PARALLEL 解决多轮调度问题）
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


# ── 扩展变体（取消注释以使用）────────────────────────────────────

# 融合激活函数（在 acc store 前施加）：
#   acc = tl.sigmoid(acc) * acc  # SiLU
#   acc = 0.5 * acc * (1 + tl.math.erf(acc * 0.7071067811865476))  # GELU

# 融合 Bias（在 acc store 前加 bias 向量）：
#   bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
#   acc += bias[None, :]

```

<!-- okf:related:start -->

# 相关

- 对应指南: [融合算子综合页](op-class-fused_ops.md) — 同属优化知识，主题相关
- 对应指南: [MatMul / GEMM 高性能算子](op-class-matmul.md) — 同属优化知识，主题相关
- 相关主题: [GEMM 操作数保持 low_dtype，fp32 累加](../runbooks/algorithm-numerics/gemm-low-dtype-fp32-accum.md) — 同属优化知识，主题相关
- 相关主题: [exp_matmul_vec_parallel_001 — matmul 向量并行 + activation fusion](../runbooks/compute-pipeline/instances/traj-exp-matmul-vec-parallel-001.md) — 同属优化知识，主题相关
- 相关主题: [算子融合模式与约束决策](../runbooks/compute-pipeline/operator-fusion-patterns.md) — 同属优化知识，主题相关
- 相关主题: [Matmul Compute 优化聚合](../runbooks/compute-pipeline/traj-matmul-compute.md) — 同属优化知识，主题相关
- 相关主题: [Matmul Scheduling 优化聚合](../runbooks/compute-pipeline/traj-matmul-scheduling.md) — 同属优化知识，主题相关
- 相关主题: [Flash Attention Causal Skeleton](skeleton-flash_attention_causal.md) — 主题相关：共享 code_skeleton
- 相关主题: [Moe Dispatch Skeleton](skeleton-moe_dispatch.md) — 主题相关：共享 code_skeleton
- 相关主题: [Rmsnorm Fused Skeleton](skeleton-rmsnorm_fused.md) — 主题相关：共享 code_skeleton

<!-- okf:related:end -->
