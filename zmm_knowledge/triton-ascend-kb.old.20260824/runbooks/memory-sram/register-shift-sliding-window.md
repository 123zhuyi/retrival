---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: "1.1"
title: 寄存器移位复用滑动窗口，消除重复 HBM 读
description: 对固定小窗口（W=2-8）的序列处理，把窗口元素驻留寄存器，每步仅加载 1 个新元素，通过寄存器移位替代 W-1 次重复 load。
tags: [register_shift, sliding_window, causal_conv1d, ssm, convolution, hbm_reuse]
confidence: verified
status: verified
created_at: "2026-08-10T00:00:00Z"
updated_at: "2026-08-10T00:00:00Z"
---

## OPT-22 寄存器移位复用滑动窗口，消除重复 HBM 读

【优化维度】：内存｜访存效率
【泛化层级】：条件性（迁移条件：固定小窗口 W≤8 的序列处理，窗口访问为严格移位；失效边界：W 大、随机窗口访问、或 seq_len 太短无法摊还初始化开销）

<原则：对卷积、移动平均、SSM 前缀处理等滑动窗口场景，若每次迭代都重新加载全部 W 个窗口元素，会造成大量冗余 HBM 读。将窗口驻留命名寄存器，每步仅加载 1 个新元素，其余 W-1 个通过寄存器间移位复用，可消除 (W-1)/W 的 HBM 读。>

**核心思想**：

```
迭代 i:   [col0, col1, col2, col3] → 计算 → store
迭代 i+1:         [col1, col2, col3, new_col] → 计算 → store
                ↑ 移位（零 load）  ↑ 仅此 1 次 load
```

**正例（W=4）**：

```python
# 初始化：一次加载 W 个窗口元素
col0 = tl.load(ptr + 0*stride, mask=mask)
col1 = tl.load(ptr + 1*stride, mask=mask)
col2 = tl.load(ptr + 2*stride, mask=mask)
col3 = tl.load(ptr + 3*stride, mask=mask)

for i in range(seq_len - W):
    acc = col0*w0 + col1*w1 + col2*w2 + col3*w3
    tl.store(out + i*stride, acc, mask=mask)
    # 寄存器移位：零 HBM 操作
    col0 = col1; col1 = col2; col2 = col3
    col3 = tl.load(ptr + (i+W) * stride, mask=mask)  # 仅 1 次新 load
```

**适用条件**：

| 条件 | 说明 |
|---|---|
| W 固定且小 | W ≤ 8，所有窗口元素可驻留寄存器 |
| 窗口访问严格移位 | 最旧丢弃，其余左移，末尾新元素 |
| seq_len 足够大 | ≥ 64，初始化的 W 次 load 可被摊还 |
| 需要滑动窗口连续性 | 卷积、移动平均、SSM 前缀处理 |

**收益**：W=4 时从每步 4 次 load 降至 1 次 load + 3 次寄存器移位，消除 75% HBM 读。

**与重叠 T-tiling 的关系**：

- 寄存器滑动窗口：用于**无 T-tiling** 的 Prefill kernel（逐 token 顺序处理）。
- 重叠 T-tiling：用于**有 T-tiling** 的 Update kernel（跨 tile 边界连续性）。

两者解决不同维度问题，可在同一 kernel 族中并存。

**坏实践（反例）**：
```python
for i in range(seq_len):
    # 每步都重新加载 W 个元素
    for w in range(W):
        x[w] = tl.load(ptr + (i+w)*stride)
```
后果：W=4 时每步 4 次 HBM 读，带宽浪费。✅ 正解即寄存器移位复用。

## 已知实例（P3 挂载）

### Experience YAML 反链
- `exp_causal_conv1d_prefill_002` — Memory Access Optimization （收益 消除 75% 的 token load，实测贡献 Prefill kernel 整体加速比中的最大份额（与 2D Grid + DIM_IS_MULTIPLE 组合达 1.60x 平均加速比，最高 1.84x））
  - 滑动窗口寄存器复用是 Prefill kernel 最核心的创新，利用了 W=4 的小卷积核特性。此模式可推广到任何固定小窗口的序列处理 kernel（如移动平均、指数平滑、small-kernel 1D conv），前提是窗口大小在编译期已知且足够小（≤8）。
- `exp_causal_conv1d_prefill_003` — Scheduling Optimization （收益 2D Grid vs 1D Grid: 25-39% 性能优势（搜索验证）; 整体 Prefill kernel: 1.60x 平均加速比）
  - 2D Grid + BTD 布局是 Prefill 阶段的架构基础决策，与 Update kernel 的 1D Grid + BDT 布局 + T-tiling 形成互补——两者针对不同的 seq_len 范围（长 vs 短）做了相反的架构选择。选择依据见 tiling.md…

<!-- okf:related:start -->

# 相关

- 实践案例: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](../../ops/nsa-compress-attention-infer.md) — 主题相关：共享 sliding_window
- 实践案例: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](../../ops/nsa-compress-attention.md) — 主题相关：共享 sliding_window
- 实践案例: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](../../ops/recurrent-gated-delta-rule.md) — 主题相关：共享 ssm
- 开发指南: [Ascend 910B2/910B3 Hardware Reference](../../reference/ascend-910b.md) — 同属优化知识，主题相关
- 开发指南: [卷积算子优化（Causal Conv1D）](../../reference/op-class-convolution.md) — 主题相关：共享 convolution, ssm
- 开发指南: [SSM / 状态空间模型算子优化](../../reference/op-class-ssm.md) — 主题相关：共享 ssm
- 相关主题: [标量循环改写为 tl.dot，提升 Cube 利用率](../compute-pipeline/scalar-loop-to-dot.md) — 主题相关：共享 ssm
- 相关主题: [Causal Conv1D Compute 优化聚合](../compute-pipeline/traj-conv-compute.md) — 主题相关：共享 causal_conv1d
- 相关主题: [Causal Conv1D Prefill/Update 跨 Kernel 优化迁移](../compute-pipeline/traj-conv-cross_kernel.md) — 主题相关：共享 causal_conv1d
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](../compute-pipeline/traj-conv-memory_access.md) — 主题相关：共享 causal_conv1d, sliding_window, ssm
- 相关主题: [Causal Conv1D 调度优化：Prefill/Update 布局分野、2D Grid 与 pad_slot 早退](../compute-pipeline/traj-conv-scheduling.md) — 主题相关：共享 causal_conv1d
- 相关主题: [Reduction Compute 优化聚合](../compute-pipeline/traj-reduction-compute.md) — 主题相关：共享 ssm
- 相关主题: [SSM Compute 优化聚合](../compute-pipeline/traj-ssm-compute.md) — 主题相关：共享 ssm
- 相关主题: [SSM 反向 kernel 标量消除：i64 降级、整数比较与 boundary_check 优化](../compute-pipeline/traj-ssm-scalar_elimination.md) — 主题相关：共享 ssm
- 相关主题: [重叠 T-tiling 保证跨 tile 滑动窗口连续性](overlapping-t-tiling.md) — 主题相关：共享 causal_conv1d, sliding_window, ssm
- 相关主题: [preload + extract_slice 消除循环内同一数据重复 HBM 读取](preload-extract-slice-eliminate-reread.md) — 同属优化知识，主题相关
- 相关主题: [Solve_Tril 内存访问优化：Preload + extract_slice 消除循环内 HBM 重读](traj-solve_tril-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [SSM 内存访问优化：转置 load 合并化、寄存器复用与分块状态矩阵](traj-ssm-memory_access.md) — 主题相关：共享 ssm

<!-- okf:related:end -->
