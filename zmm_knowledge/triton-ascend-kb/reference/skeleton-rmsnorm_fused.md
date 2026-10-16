---
schema_version: okf.v1
kind: example
type: code_example
source_family: reference
title: Rmsnorm Fused Skeleton
description: 骨架：RMSNorm（融合变体：支持 Skip Add + RMSNorm） 适用：fp16/bf16，任意 batch/seq/hidden
tags: [code_skeleton, rmsnorm_fused]
confidence: inferred
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/pages/skeletons/rmsnorm_fused.py
created_at: '2026-08-10T02:56:36Z'
updated_at: '2026-08-10T02:56:36Z'
---

sources:
  - url: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/pages/skeletons/rmsnorm_fused.py
    role: primary

> 迁移备注：本卡由 `cannbot-skills_5171/ops/triton-knowledge-retrieval/knowledge/pages/skeletons/rmsnorm_fused.py` 迁移而来，原始为代码骨架，未做内容改写，仅转 OKF frontmatter。

```python
"""
骨架：RMSNorm（融合变体：支持 Skip Add + RMSNorm）
适用：fp16/bf16，任意 batch/seq/hidden
经验：exp_rmsnorm_001(086) Persistent (35%), exp_rmsnorm_npu_001(093) 列分块 (35%)
       exp_skip_layernorm_001(118) Skip+Norm融合 (30%)
参考：ops/normalization.md, techniques/persistent_kernels.md
约束：[hardware/constraints_cheatsheet]
"""

import torch
import triton
import triton.language as tl
from triton.runtime import driver


@triton.heuristics({
    # 自动选 BLOCK_SIZE_N：尽量一次装入整行，不超 UB（fp32: ~49K elements）
    'BLOCK_SIZE_N': lambda args: min(
        triton.next_power_of_2(args['N']),
        4096  # 上限：4096 * 4B(fp32) = 16KB，远小于 192KB UB
    ),
})
@triton.jit
def rmsnorm_kernel(
    x_ptr,            # 输入 [M, N]
    residual_ptr,     # Skip 残差输入（可选，HAS_RESIDUAL=True 时使用）
    weight_ptr,       # 归一化权重 [N]
    out_ptr,          # 输出 [M, N]
    residual_out_ptr, # Skip Add 后的中间结果输出（可选）
    M, N,
    eps: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,   # 是否融合 Skip Add
    BLOCK_SIZE_N: tl.constexpr,
):
    # ── Persistent kernel：固定 grid = num_aicores，grid-stride 遍历行 ──
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # 权重只加载一次，复用于此 program 处理的所有行
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    w_mask = offs_n < N
    weight = tl.load(weight_ptr + offs_n, mask=w_mask, other=0.0).to(tl.float32)

    for row in range(pid, M, num_programs):
        x_ptrs = x_ptr + row * N + offs_n
        x = tl.load(x_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        if HAS_RESIDUAL:
            res_ptrs = residual_ptr + row * N + offs_n
            residual = tl.load(res_ptrs, mask=w_mask, other=0.0).to(tl.float32)
            x = x + residual
            # 写出 skip add 中间结果（后续层可能需要）
            tl.store(residual_out_ptr + row * N + offs_n, x, mask=w_mask)

        # ── Two-pass RMSNorm ──────────────────────────────────────
        # Pass 1: 计算 variance = mean(x^2)（fp32 累加）
        x_sq = x * x
        var = tl.sum(x_sq, axis=0) / N   # scalar

        # Pass 2: 归一化
        rstd = 1.0 / tl.sqrt(var + eps)  # rsqrt 替代 sqrt+div
        x_norm = x * rstd
        out = x_norm * weight

        tl.store(out_ptr + row * N + offs_n, out.to(tl.float16), mask=w_mask)


# ── 大 hidden dim 变体（n_cols > 4096，需列分块 two-pass）──────────
@triton.jit
def rmsnorm_large_hidden_kernel(
    x_ptr, weight_ptr, out_ptr,
    M, N,
    eps: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,   # 每次处理的列数（满足 UB 约束）
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for row in range(pid, M, num_programs):
        # Pass 1: 分块累加 sum(x^2)
        var_acc = tl.zeros([1], dtype=tl.float32)
        for col_start in range(0, N, BLOCK_SIZE_N):
            offs = col_start + tl.arange(0, BLOCK_SIZE_N)
            mask = offs < N
            x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
            var_acc += tl.sum(x * x, axis=0)
        var = var_acc / N
        rstd = 1.0 / tl.sqrt(var + eps)

        # Pass 2: 分块归一化
        for col_start in range(0, N, BLOCK_SIZE_N):
            offs = col_start + tl.arange(0, BLOCK_SIZE_N)
            mask = offs < N
            x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            out = x * rstd * w
            tl.store(out_ptr + row * N + offs, out.to(tl.float16), mask=mask)


def rmsnorm(
    x: torch.Tensor,          # [M, N]
    weight: torch.Tensor,     # [N]
    residual: torch.Tensor = None,  # [M, N]，可选 Skip Add
    eps: float = 1e-5,
):
    assert x.is_contiguous()
    M, N = x.shape
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x) if residual is not None else None

    # 获取 AI Core 数量（Persistent kernel 的 grid 大小）
    device = x.device
    try:
        props = driver.active.utils.get_device_properties(device.index)
        num_cores = props["multiprocessor_count"]
    except Exception:
        num_cores = 20  # 910b2 默认值

    grid_size = min(M, num_cores)

    if N <= 4096:
        rmsnorm_kernel[(grid_size,)](
            x, residual if residual is not None else x,
            weight, out,
            residual_out if residual_out is not None else out,
            M, N,
            eps=eps,
            HAS_RESIDUAL=(residual is not None),
        )
    else:
        # 列分块：BLOCK_SIZE_N * 4B(fp32) ≤ ~16KB
        BLOCK_N = min(triton.next_power_of_2(N), 4096)
        rmsnorm_large_hidden_kernel[(grid_size,)](
            x, weight, out,
            M, N,
            eps=eps,
            BLOCK_SIZE_N=BLOCK_N,
        )

    return (out, residual_out) if residual is not None else out

```

<!-- okf:related:start -->

# 相关

- 对应指南: [融合算子综合页](op-class-fused_ops.md) — 同属优化知识，主题相关
- 对应指南: [归一化算子优化](op-class-normalization.md) — 同属优化知识，主题相关
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../ops/mla-preprocess.md) — 同属优化知识，主题相关
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../ops/mla-prolog-v3.md) — 同属优化知识，主题相关
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](../ops/mla-prolog.md) — 同属优化知识，主题相关
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](../ops/rmsnorm-rope.md) — 同属优化知识，主题相关
- 相关主题: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../runbooks/algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 同属优化知识，主题相关
- 相关主题: [算子融合模式与约束决策](../runbooks/compute-pipeline/operator-fusion-patterns.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](../runbooks/memory-sram/exp_fused_rms_norm_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](../runbooks/memory-sram/exp_gemma_rmsnorm_001.md) — 同属优化知识，主题相关
- 相关主题: [Flash Attention Causal Skeleton](skeleton-flash_attention_causal.md) — 主题相关：共享 code_skeleton
- 相关主题: [Gemm Basic Skeleton](skeleton-gemm_basic.md) — 主题相关：共享 code_skeleton
- 相关主题: [Moe Dispatch Skeleton](skeleton-moe_dispatch.md) — 主题相关：共享 code_skeleton

<!-- okf:related:end -->
