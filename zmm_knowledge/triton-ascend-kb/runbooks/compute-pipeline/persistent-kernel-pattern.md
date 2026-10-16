---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.3"
title: persistent kernel 模式（grid-stride 摊还启动开销）
description: 当 n_rows >> num_aicores 时，用固定 grid=num_aicores + 内部 grid-stride 循环，让每核处理多行，摊还 launch 开销并复用权重。
tags: [persistent_kernel, grid_stride, num_aicores, launch_overhead, inference]
confidence: verified
status: verified
created_at: "2026-08-10T00:00:00Z"
updated_at: "2026-08-10T00:00:00Z"
---

## OPT-15 persistent kernel 模式（grid-stride 摊还启动开销）

【优化维度】：计算｜多核负载
【泛化层级】：条件性（迁移条件：n_rows >> num_aicores，权重/常量跨行复用；失效边界：n_rows ≈ 或 < num_aicores，或每行无共享权重）

<原则：普通 kernel 的 grid 等于工作量，每核处理一行就退出，launch 开销和权重加载无法摊还。Persistent kernel 把 grid 固定为 AI Core 数量，每核内部用 grid-stride 循环处理多行，权重只加载一次、launch 只发生一次，通常带来 25–35% 吞吐提升。>

**普通 vs Persistent**：

```
普通：   [launch][core0: row0] [launch][core0: row1] [launch][core0: row2] ...
Persistent: [launch] [core0: row0 → row4 → row8 → ...] （单次 launch，循环）
```

**正例（Persistent LayerNorm）**：

```python
@triton.jit
def persistent_layernorm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    n_rows, n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    num_programs = tl.num_programs(axis=0)
    pid = tl.program_id(axis=0)

    # 权重只加载一次
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    weight = tl.load(W_ptr + offs, mask=mask)
    bias = tl.load(B_ptr + offs, mask=mask)

    # grid-stride 循环
    for row in range(pid, n_rows, num_programs):
        x = tl.load(X_ptr + row * n_cols + offs, mask=mask)
        mean = tl.sum(x, axis=0) / n_cols
        var = tl.sum((x - mean) * (x - mean), axis=0) / n_cols
        x_norm = (x - mean) / tl.sqrt(var + eps)
        y = x_norm * weight + bias
        tl.store(Y_ptr + row * n_cols + offs, y, mask=mask)
```

**启动**：

```python
grid = (num_aicores,)  # 固定为 AI Core 数
persistent_layernorm_kernel[grid](...)
```

**何时使用**：

| 条件 | 是否推荐 |
|---|---|
| n_rows >> num_aicores | ✅ 强推荐 |
| n_rows ≈ num_aicores | ❌ 无收益 |
| n_rows < num_aicores | ❌ 浪费核，普通 kernel + pad |
| 每行需加载共享权重/常量 | ✅ 摊还 load |
| 每行工作量极小 | ❌ 循环开销占主导 |
| 融合 GEMM + activation | ✅ 配合 `dl.async_task` 异构并行 |

**与 `dl.async_task` 组合**：

```python
for row_block in range(pid, n_row_blocks, num_programs):
    with dl.async_task(scope="cube"):
        acc = tl.dot(a_tile, b_tile)

    with dl.async_task(scope="vector"):
        rms = tl.sqrt(tl.sum(acc * acc, axis=1) / n_cols + eps)
        result = acc / rms[:, None] * weight
        tl.store(Y_ptr + ..., result)
```

**坏实践（反例）**：
```python
grid = (n_rows,)  # n_rows 很大时 launch 开销高
```
后果：n_rows=8192, num_aicores=20 时，普通 kernel 需要 8192 次 program 调度，persistent 只需 20 次 launch + 每核 410 次循环。✅ 正解即大 batch/大行数用 persistent。

**诊断清单**：
1. 工作负载行数是否远大于核数（>5x）？
2. 权重/常量是否跨行复用？
3. 当前 kernel 是否 launch 开销占比较高？
4. 若满足，改为 `grid=(num_aicores,)` + grid-stride loop。
5. 验证精度和 throughput。

## 已知实例（P3 挂载）

### Experience YAML 反链
- `exp_ascend_grid_large_shape_001` — 负载均衡优化 （收益 15%）
  - 该策略是大shape算子（元素数≥1e6）的最优解，相比直接切分Grid的方案，负载均衡提升15%；Ascend 910B3默认VEC核心数为64，可按需调整
- `exp_flash_attn_gridstride_001` — Scheduling Optimization （收益 35%）
  - The grid stride factor (20) should be tuned based on the specific NPU's SM count and the specific sequence length to maximize occupancy…
  - 对应 kernel: `raw_experiences/extracted_kernels/012-_attn_fwd.py`
- `exp_sigmoid_grid_stride_001` — Scheduling Optimization （收益 15%）
  - The `NUM_CORES` parameter must accurately reflect the number of thread blocks launched to ensure correct stride behavior. This kernel is…
  - 对应 kernel: `sigmoid_kernel`

<!-- okf:related:start -->

# 相关

- 实践案例: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](../../ops/fused-infer-attention-score.md) — 同属优化知识，主题相关
- 实践案例: [Gdn2Fwd（varlen Gated DeltaNet 2 前向）双骨架 host dispatch 实测案例：serial persistent + chunked WY](../../ops/gdn2-fwd-chunked-wy-dispatch.md) — serial persistent 骨架即 persistent kernel 模式（含 BAR 同步地板的反例边界）
- 实践案例: [IncreFlashAttention inference-decode on Ascend NPU](../../ops/incre-flash-attention.md) — 同属优化知识，主题相关
- 实践案例: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](../../ops/kv-quant-sparse-flash-attention.md) — 主题相关：共享 inference
- 实践案例: [PromptFlashAttention inference/prefill on Ascend NPU](../../ops/prompt-flash-attention.md) — 同属优化知识，主题相关
- 实践案例: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](../../ops/quant-lightning-indexer.md) — 主题相关：共享 inference
- 实践案例: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 同属优化知识，主题相关
- 实践案例: [SparseFlashAttention inference (sparse KV) on Ascend NPU](../../ops/sparse-flash-attention.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: persistent kernel strategy](../memory-sram/exp_layernorm_fwd_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: persistent kernel grid stride](exp_entropy_backward_d_weight_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: diagonal traversal pattern](exp_launch_diagonal_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: persistent kernel grid stride loop](exp_linear_persistent_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: dynamic persistent scheduling](exp_partial_matmul_dynamic_001.md) — 同属优化知识，主题相关
- 相关主题: [负载均衡分核：grid 对齐物理核数，核内循环替代 GPU 多维 grid](load-balance-grid-sizing.md) — 主题相关：共享 persistent_kernel

<!-- okf:related:end -->
