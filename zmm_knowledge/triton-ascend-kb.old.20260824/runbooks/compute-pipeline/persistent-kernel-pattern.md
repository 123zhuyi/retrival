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
- `exp_ascend_grid_001` — 语法规范 （收益 0%）
  - Grid设置违反规范会直接导致kernel编译/启动失败，910b2/910b3平台无容错机制，需严格遵循维度和大小限制
- `exp_ascend_grid_large_shape_001` — 负载均衡优化 （收益 15%）
  - 该策略是大shape算子（元素数≥1e6）的最优解，相比直接切分Grid的方案，负载均衡提升15%；Ascend 910B3默认VEC核心数为64，可按需调整
- `exp_flash_attn_gridstride_001` — Scheduling Optimization （收益 35%）
  - The grid stride factor (20) should be tuned based on the specific NPU's SM count and the specific sequence length to maximize occupancy…
  - 对应 kernel: `raw_experiences/extracted_kernels/012-_attn_fwd.py`
- `exp_fused_matmul_rmsnorm_001` — Scheduling Optimization （收益 25%）
  - This optimization is specifically designed for the Ascend NPU architecture using the Da Vinci dialect extensions. It assumes that the…
  - 对应 kernel: `raw_experiences/extracted_kernels/052-fused_matmul_qb_and_rmsnorm_kv_kernel.py`
- `exp_rmsnorm_001` — Memory Access Optimization （收益 15%）
  - The implementation uses a two-pass strategy over the column dimension (one for reduction, one for application). While this increases memory…
  - 对应 kernel: `raw_experiences/extracted_kernels/149-_rms_norm_kernel.py`
- `exp_sigmoid_grid_stride_001` — Scheduling Optimization （收益 15%）
  - The `NUM_CORES` parameter must accurately reflect the number of thread blocks launched to ensure correct stride behavior. This kernel is…
  - 对应 kernel: `sigmoid_kernel`
- `exp_gemma_rmsnorm_001` — Memory Access Optimization （收益 35%）
  - This kernel implements the specific Gemma variant of RMSNorm where the weight is effectively added by 1.0 (w + 1.0) during scaling. It is…
  - 对应 kernel: `raw_experiences/extracted_kernels/190-add_gemma_rms_norm_kernel.py`

<!-- okf:related:start -->

# 相关

- 实践案例: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](../../ops/fused-infer-attention-score.md) — 同属优化知识，主题相关
- 实践案例: [IncreFlashAttention inference-decode on Ascend NPU](../../ops/incre-flash-attention.md) — 同属优化知识，主题相关
- 实践案例: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](../../ops/kv-quant-sparse-flash-attention.md) — 主题相关：共享 inference
- 实践案例: [PromptFlashAttention inference/prefill on Ascend NPU](../../ops/prompt-flash-attention.md) — 同属优化知识，主题相关
- 实践案例: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](../../ops/quant-lightning-indexer.md) — 主题相关：共享 inference
- 实践案例: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 同属优化知识，主题相关
- 实践案例: [SparseFlashAttention inference (sparse KV) on Ascend NPU](../../ops/sparse-flash-attention.md) — 同属优化知识，主题相关
- 开发指南: [Ascend 910B2/910B3 Hardware Reference](../../reference/ascend-910b.md) — 同属优化知识，主题相关
- 开发指南: [Ascend NPU 特有语法与模式](../../reference/ascend-patterns.md) — 同属优化知识，主题相关
- 相关主题: [向量 gather 随机地址在 Ascend 上极慢，二分/搜索类走 per-expert 标量+grid 并行](../memory-sram/gather-random-address-slow.md) — 同属优化知识，主题相关
- 相关主题: [输入进入 kernel 前必须 contiguous](../memory-sram/input-contiguous.md) — 同属优化知识，主题相关
- 相关主题: [pertensor scale 必须物化为连续 [1,N]，禁止 stride-0 广播 load](../memory-sram/pertensor-scale-materialize.md) — 同属优化知识，主题相关
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](../memory-sram/traj-general_perf-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend kernel 调试排查清单](../meta-skills/debugging-checklist.md) — 同属优化知识，主题相关
- 相关主题: [Ascend Triton 语法规范避坑聚合](../meta-skills/traj-general_ascend-syntax_spec.md) — 同属优化知识，主题相关
- 相关主题: [避免 Stride 访问：优先连续化后一维遍历](../meta-skills/traj-general_perf-perf_pitfall.md) — 同属优化知识，主题相关
- 相关主题: [按输出元素数（而非输入）分配 grid，且钳制到 num_cores](grid-by-output-count.md) — 同属优化知识，主题相关
- 相关主题: [grid 总数不超过核数](grid-cap-to-core-num.md) — 同属优化知识，主题相关
- 相关主题: [1D grid 硬上限 65535，超出用 2D grid 或增大 BLOCK](grid-limit-65535.md) — 同属优化知识，主题相关
- 相关主题: [grid_size 必须与 num_cores 入参严格相等](grid-strict-match-num-cores.md) — 同属优化知识，主题相关
- 相关主题: [Activation Scheduling 优化聚合](traj-activation-scheduling.md) — 主题相关：共享 grid_stride
- 相关主题: [Attention Scheduling 优化聚合](traj-attention-scheduling.md) — 主题相关：共享 grid_stride
- 相关主题: [Causal Conv1D 调度优化：Prefill/Update 布局分野、2D Grid 与 pad_slot 早退](traj-conv-scheduling.md) — 同属优化知识，主题相关
- 相关主题: [大词表 Entropy Loss Backward 调度融合优化](traj-loss-scheduling.md) — 主题相关：共享 grid_stride
- 相关主题: [Misc 算子 Scheduling 优化聚合](traj-misc-scheduling.md) — 主题相关：共享 persistent_kernel
- 相关主题: [Normalization Scheduling 优化聚合](traj-normalization-scheduling.md) — 主题相关：共享 grid_stride
- 相关主题: [RoPE 计算优化：cos/sin 复用、向量化旋转与多模态位置编码融合](traj-rope-compute.md) — 主题相关：共享 grid_stride
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](traj-rope-scheduling.md) — 主题相关：共享 grid_stride
- 相关主题: [Solve_Tril 调度优化：三层分级 kernel、分块递推与 Grid 并行](traj-solve_tril-scheduling.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
