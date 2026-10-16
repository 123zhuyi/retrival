---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: "1.1"
title: matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load
description: 矩阵 tile load 应使用 tl.make_block_ptr 触发硬件 block transfer；退化为多次独立标量 tl.load 会丧失预取、边界检查和合并访存，性能倒退 2–3x。
tags: [block_ptr, scalar_load, matmul, coalesced, prefetch]
confidence: verified
status: verified
created_at: "2026-08-10T00:00:00Z"
updated_at: "2026-08-10T00:00:00Z"
---

## CT-11 matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load

【优化维度】：内存｜访存效率
【泛化层级】：Triton通用（迁移条件：所有矩阵/块级 load/store；失效边界：明确只需单个元素的罕见场景）

<原则：在 Ascend NPU 上，`tl.make_block_ptr` 会映射为硬件 block transfer 指令（类 DMA），一次搬运整个 tile，并自动获得边界检查、零填充和异步预取提示。若为了手动索引把 block_ptr 退化成多次独立 `tl.load(weight_ptr + d_offs * width + offset, mask=mask_w)`，每次 load 都是独立标量地址计算，丧失全部硬件加速。>

**为什么性能差异大**：

| 特性 | `tl.make_block_ptr` | 独立标量 `tl.load` |
|---|---|---|
| 指令类型 | 硬件 block transfer | 标量地址 load |
| 边界检查 | 硬件级 `boundary_check` | 手动 mask（可能标量化） |
| 零填充 | `padding_option` 自动 | 手动处理 |
| 预取 | MTE2 可异步预取下一 tile | 无预取提示 |
| 合并访存 | 连续 burst | 离散地址，burst 合并失败 |

**正例（保留 block_ptr）**：

```python
a_ptr = tl.make_block_ptr(
    base=A_ptr,
    shape=(M, K),
    strides=(stride_am, stride_ak),
    offsets=(pid_m * BLOCK_M, k_off),
    block_shape=(BLOCK_M, BLOCK_K),
    order=(1, 0),   # row-major：最后一维连续
)
a = tl.load(a_ptr)
a_ptr = tl.advance(a_ptr, (0, BLOCK_K))  # K 循环推进
```

**坏实践（反例）**：

```python
# 为了循环展开，退化为 4 次独立标量 load
w0 = tl.load(weight_ptr + d_offs * W + 0, mask=mask_w)
w1 = tl.load(weight_ptr + d_offs * W + 1, mask=mask_w)
w2 = tl.load(weight_ptr + d_offs * W + 2, mask=mask_w)
w3 = tl.load(weight_ptr + d_offs * W + 3, mask=mask_w)
```
后果：在 D≥2048 的 Causal Conv1D Update kernel 中，可导致 2–3x 性能倒退。✅ 正解即保留 block_ptr，必要时用 `tl.extract_slice` 切分。

**混合方案（保留硬件加速 + 手动索引）**：

```python
# 推荐：block_ptr 加载整个 tile，再用 extract_slice 切分
w = tl.load(tl.make_block_ptr(weight_ptr, ..., block_shape=(D_CHK, W)))
w0 = tl.extract_slice(w, (0, 0), (D_CHK, 1), (1, 1))  # view，零数据移动
w1 = tl.extract_slice(w, (0, 1), (D_CHK, 1), (1, 1))
# ...
```

**诊断清单**：
1. profiling 中 `aiv_mte2_time` 或 `mte2_stall` 异常高。
2. 代码中存在 `tl.load(ptr + expr_i)` 的循环/展开模式，其中 `expr_i` 是运行时变量。
3. 确认该 load 本可以用 `tl.make_block_ptr` 表达。
4. 重构为 block_ptr + `tl.extract_slice`。
5. 重跑 profiling 验证 MTE2 下降。

## 已知实例（P3 挂载）

### Experience YAML 反链
- `exp_perf_002` — 内存访问优化 （收益 25%）
  - 适用于所有2D张量（如矩阵、特征图）的访存场景，tl.make_block_ptr可自动优化内存合并，相比手动计算指针减少约20%的访存开销，910b2/910b3平台效果尤为显著
- `exp_perf_011` — 内存访问优化 （收益 50%）
  - 手动指针替代block_ptr在Ascend上是一把双刃剑：它消除了boundary_check的标量开销，但要求开发者手动保证指针计算正确和mask完备。对确定性大块访存（如无边界tile），保留block_ptr仍可能获得更好的硬件预取。

<!-- okf:related:start -->

# 相关

- 实践案例: [ChunkGatedDeltaRuleFwdH（GDN/KDA chunk 间状态递推前向）Ascend C → Triton 语义转化卡](../../ops/chunk-gated-delta-rule-fwd-h.md) — 主题相关：共享 matmul
- 实践案例: [FlashAttentionScoreGrad training-backward on Ascend NPU](../../ops/flash-attention-score-grad.md) — 主题相关：共享 matmul
- 实践案例: [FlashAttentionScore training-forward on Ascend NPU](../../ops/flash-attention-score.md) — 主题相关：共享 matmul
- 实践案例: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](../../ops/fused-floyd-attention-grad.md) — 主题相关：共享 matmul
- 实践案例: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 主题相关：共享 matmul
- 实践案例: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../../ops/mla-preprocess.md) — 主题相关：共享 matmul
- 实践案例: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../../ops/mla-prolog-v3.md) — 主题相关：共享 matmul
- 实践案例: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](../../ops/mla-prolog.md) — 主题相关：共享 matmul
- 实践案例: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 主题相关：共享 matmul
- 实践案例: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](../../ops/recurrent-gated-delta-rule.md) — 主题相关：共享 matmul
- 开发指南: [MatMul / GEMM 高性能算子](../../reference/op-class-matmul.md) — 主题相关：共享 matmul
- 相关主题: [matmul core mapping 策略选择（列/行/对角）](../compute-pipeline/core-mapping-strategy-selection.md) — 主题相关：共享 matmul
- 相关主题: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](../compute-pipeline/diagonal-core-mapping-l2-bank-conflict.md) — 主题相关：共享 matmul
- 相关主题: [Scheduling Optimization: dynamic autotuning](../compute-pipeline/exp_addmm_ascend_001.md) — 主题相关：共享 matmul
- 相关主题: [Compute Optimization: block ptr tiling](../compute-pipeline/exp_block_inverse_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: block ptr utilization](../compute-pipeline/exp_block_matmul_inv_001.md) — 主题相关：共享 matmul
- 相关主题: [Compute Optimization: block ptr tiling](../compute-pipeline/exp_chunk_fwd_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: block ptr abstraction](../compute-pipeline/exp_chunk_kkt_001.md) — 同属优化知识，主题相关
- 相关主题: [标量消除: block_ptr boundary_check 与 i32 比较定向消除](../compute-pipeline/exp_kda_bwd_intra_scalar_opt_001.md) — 同属优化知识，主题相关
- 相关主题: [per-channel 参数标量 load 路径，避免向量化 int 除法降级](../compute-pipeline/scalar-load-channel-index.md) — 主题相关：共享 scalar_load
- 相关主题: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](../portability/exp_ascend_blockptr_boundary_scalar_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_causal_conv1d_bwd_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: make block ptr hardware acceleration](exp_causal_conv1d_update_003.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_paged_attn_decode_001.md) — 同属优化知识，主题相关
- 相关主题: [内存访问优化：2D 张量块指针访存（make_block_ptr）](exp_perf_002.md) — 同属优化知识，主题相关
- 相关主题: [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](exp_perf_011.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: block ptr abstraction](exp_sdpa_infer_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_topk_pad_001.md) — 同属优化知识，主题相关
- 相关主题: [preload + extract_slice 消除循环内同一数据重复 HBM 读取](preload-extract-slice-eliminate-reread.md) — 主题相关：共享 block_ptr
- 相关主题: [向量化读写与 Coalesced Memory Access](vectorized-coalesced-io.md) — 主题相关：共享 coalesced

<!-- okf:related:end -->
