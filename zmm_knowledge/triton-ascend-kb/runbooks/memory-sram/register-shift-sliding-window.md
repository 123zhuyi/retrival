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

<!-- okf:related:start -->

# 相关

- 实践案例: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](../../ops/nsa-compress-attention-infer.md) — 主题相关：共享 sliding_window
- 实践案例: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](../../ops/nsa-compress-attention.md) — 主题相关：共享 sliding_window
- 开发指南: [Ascend 910B2/910B3 Hardware Reference](../../reference/ascend-910b.md) — 同属优化知识，主题相关
- 开发指南: [卷积算子优化（Causal Conv1D）](../../reference/op-class-convolution.md) — 主题相关：共享 convolution, ssm
- 相关主题: [Compute Optimization: register accumulation](../compute-pipeline/exp_hist_count_001.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend 移位 load 降级契约（大位移 ~25x 标量化，2D tile ±1 不合并）](../compute-pipeline/triton-ascend-shifted-load-degradation.md) — 主题相关：共享 sliding_window
- 相关主题: [Memory Access Optimization: sliding window register reuse](exp_causal_conv1d_prefill_002.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: register buffering](exp_moe_binned_scatter_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: register reuse intermediate](exp_rmsnorm_add_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: no-tiling register reuse with conflict-free dW accumulation](exp_rmsnorm_backward_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: preload extract slice eliminate hbm read](exp_solve_tril_002.md) — 同属优化知识，主题相关
- 相关主题: [重叠 T-tiling 保证跨 tile 滑动窗口连续性](overlapping-t-tiling.md) — 主题相关：共享 causal_conv1d, sliding_window, ssm
- 相关主题: [preload + extract_slice 消除循环内同一数据重复 HBM 读取](preload-extract-slice-eliminate-reread.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
