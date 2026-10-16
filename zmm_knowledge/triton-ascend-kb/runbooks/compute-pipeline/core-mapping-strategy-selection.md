---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.3"
title: matmul core mapping 策略选择（列/行/对角）
description: 根据 M/N 形状为 matmul 选择默认列、默认行或对角 core mapping，避免相邻 program 访问同一 L2 bank 或同一输入切片。
tags: [core_mapping, matmul, diagonal_mapping, group_size, l2_reuse, load_balance]
confidence: verified
status: verified
created_at: "2026-08-10T00:00:00Z"
updated_at: "2026-08-10T00:00:00Z"
---

## DG-8 matmul core mapping 策略选择（列/行/对角）

【优化维度】：计算｜多核负载
【泛化层级】：条件性（迁移条件：2D block 划分的 matmul；失效边界：1D 划分、矩阵过小、无共享输入模式）

<原则：core mapping 决定相邻 program 处理哪些 (M,N) 输出块，进而决定它们共享 A/B 输入切片的模式。选错策略会让相邻 core 同时命中同一 L2 bank（bank conflict）或降低 L2 复用，损失 20–40% 性能。>

**三种策略对比**：

| 策略 | 相邻 PID 访问模式 | 最佳形状 | GROUP_SIZE | 核心效果 |
|---|---|---|---|---|
| 默认列（Column） | 同一 N 列，不同 M 行 | N >> M（高瘦） | 8/16/32 | B 矩阵复用最大 |
| 默认行（Row） | 同一 M 行，不同 N 列 | M >> N（矮胖） | 8/16/32 | A 矩阵复用最大 |
| 对角（Diagonal） | 不同 M 行且不同 N 列 | M ≥ 512, N ≥ 512 | 8/16/32 | 分散 L2 bank，消除 conflict |

**访问模式示意图（4×4 tile 网格，GROUP_SIZE=4）**：

```
Column 映射：            Row 映射：              Diagonal 映射：
PID 0 1 2 3             PID 0 1 2 3            PID 0 1 2 3
M   0 1 2 3             M   0 0 0 0            M   0 1 2 3
N   0 0 0 0             N   0 1 2 3            N   0 1 2 3
                        
PID 4 5 6 7             PID 4 5 6 7            PID 4 5 6 7
M   0 1 2 3             M   1 1 1 1            M   0 1 2 3
N   1 1 1 1             N   0 1 2 3            N   1 2 3 0
```

**形状决策树**：

```
M >= 512 AND N >= 512 ?
  YES → Diagonal (dl.grouped_launch_diagonal)
  NO  → N > M * 2 ?
          YES → Default Column
          NO  → M > N * 2 ?
                    YES → Default Row
                    NO  → Default Column (safe default)
```

**GROUP_SIZE 选择**：

| GROUP_SIZE | 适用场景 | 取舍 |
|---|---|---|
| 8 | 默认；小到中等矩阵 | 组内复用适中，负载均衡好 |
| 16 | M 或 N 在 512–2048 | 增加 K 维复用，需 grid 可被整除 |
| 32 | M 或 N ≥ 2048 | 最大组内复用，负载均衡风险上升 |

**最小 autotune 配置**：至少覆盖 3 个 GROUP_SIZE：

```python
configs = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256, 'GROUP_SIZE': 8}),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 256, 'GROUP_SIZE': 16}),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 256, 'GROUP_SIZE': 32}),
]
```

**坏实践（反例）**：
```python
pid_m = pid // num_pid_n
pid_n = pid % num_pid_n  # 纯线性映射，无视形状
```
后果：大矩阵下相邻 PID 访问同一 B 列，L2 bank conflict 严重。✅ 正解即按形状选列/行/对角策略。

> **注意**：Ascend 上 `tl.swizzle2d` 行为未定义，对角映射请使用 `triton_ascend.language.grouped_launch_diagonal`。详见 [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](diagonal-core-mapping-l2-bank-conflict.md)。

## 已知实例（P3 挂载）

### Experience YAML 反链
- `exp_matmul_ascend_diag_001` — Scheduling Optimization （收益 25%）
  - The diagonal partitioning logic is triggered only when both M and N dimensions are sufficiently large (>= 8 blocks). Otherwise, it falls…
  - 对应 kernel: `raw_experiences/extracted_kernels/031-matmul_kernel.py`
- `exp_rope_matmul_001` — Hardware Adaptation Optimization （收益 35%）
  - The 'dl' module usage indicates this is optimized for specific NPU hardware (likely Ascend series). The "dot_pad_only_k" hint is critical…
  - 对应 kernel: `raw_experiences/extracted_kernels/057-_single_rope_matmul_kernel.py`
- `exp_partial_matmul_001` — Compute Optimization （收益 25%）
  - This kernel relies on the `dl` (Dialect) module for compile hints and an external `grouped_launch_diagonal` function for block mapping. It…
  - 对应 kernel: `raw_experiences/extracted_kernels/082-partial_matmul_kernel.py`

### Extracted Kernel 反链
- `239-grouped_launch_diagonal.py` — extracted kernel（source: `raw_experiences/extracted_kernels/239-grouped_launch_diagonal.py`）

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
- 实践案例: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](../../ops/nsa-compress-attention.md) — 同属优化知识，主题相关
- 实践案例: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](../../ops/quant-lightning-indexer.md) — 同属优化知识，主题相关
- 实践案例: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 主题相关：共享 matmul
- 实践案例: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](../../ops/recurrent-gated-delta-rule.md) — 主题相关：共享 matmul
- 开发指南: [MatMul / GEMM 高性能算子](../../reference/op-class-matmul.md) — 主题相关：共享 matmul
- 相关主题: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](../memory-sram/block-ptr-vs-scalar-load.md) — 主题相关：共享 matmul
- 相关主题: [访存优化：大尺寸 matmul 的 256 分块与 L0C 复用](../memory-sram/exp_matmul_001.md) — 主题相关：共享 matmul
- 相关主题: [切分优化：A、B 均不转置 matmul 的 512B 行宽对齐分块](../memory-sram/exp_matmul_003.md) — 主题相关：共享 matmul
- 相关主题: [切分优化：A 不转置、B 转置 matmul 的 K0=256 统一分块](../memory-sram/exp_matmul_004.md) — 主题相关：共享 matmul
- 相关主题: [切分优化：双转置矩阵乘的 512B 对齐分块](../memory-sram/exp_matmul_005.md) — 主题相关：共享 matmul
- 相关主题: [pergroup BLOCK_K 必须整除 group_size，选 min(group_size, 128)](../memory-sram/pergroup-blockk-align.md) — 主题相关：共享 group_size
- 相关主题: [Cube-Vector 场景行宽必须 512B 对齐](../memory-sram/row-width-512b-align.md) — 主题相关：共享 matmul
- 相关主题: [Tiling 预算与布局驱动的块大小选择](../memory-sram/tiling-budget-and-layout-driven-selection.md) — 同属优化知识，主题相关
- 相关主题: [tl.dot vs torch.matmul 精度差异诊断](../meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [固定 CUBE_CORE_NUM 启动 + 核内循环（Cube 类算子）](cube-core-num-fixed.md) — 同属优化知识，主题相关
- 相关主题: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](diagonal-core-mapping-l2-bank-conflict.md) — 主题相关：共享 core_mapping, diagonal_mapping, matmul
- 相关主题: [动态读取 Vector Core 数量，禁止硬编码](dynamic-vector-core-num.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: dynamic autotuning](exp_addmm_ascend_001.md) — 主题相关：共享 matmul
- 相关主题: [Compute Optimization: block ptr utilization](exp_block_matmul_inv_001.md) — 主题相关：共享 matmul
- 相关主题: [Scheduling Optimization: grouped scheduling](exp_bmm_grouped_001.md) — 主题相关：共享 matmul
- 相关主题: [Compute Optimization: tiling strategy](exp_cube1_qk_gemm_001.md) — 主题相关：共享 matmul
- 相关主题: [Compute Optimization: operator fusion](exp_fused_gemm_scale_001.md) — 主题相关：共享 matmul
- 相关主题: [Scheduling Optimization: async pipeline scheduling](exp_fused_matmul_norm_sigmoid_001.md) — 主题相关：共享 matmul
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rms_001.md) — 主题相关：共享 matmul
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rmsnorm_001.md) — 主题相关：共享 matmul
- 相关主题: [计算优化: fixpipe 随路量化 + K 维循环展开 + 预取](exp_matmul_002.md) — 主题相关：共享 matmul
- 相关主题: [grid 总数不超过核数](grid-cap-to-core-num.md) — 同属优化知识，主题相关
- 相关主题: [多核负载均衡分配公式（每 core 处理 block 数差不超过 1）](load-balance-formula.md) — 主题相关：共享 load_balance
- 相关主题: [负载均衡分核：grid 对齐物理核数，核内循环替代 GPU 多维 grid](load-balance-grid-sizing.md) — 主题相关：共享 load_balance
- 相关主题: [swizzle2d 块重排提升 L2 复用](swizzle2d-block-reorder.md) — 主题相关：共享 l2_reuse

<!-- okf:related:end -->
