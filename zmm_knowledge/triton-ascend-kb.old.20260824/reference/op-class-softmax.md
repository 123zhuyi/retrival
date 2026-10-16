---
schema_version: okf.v1
kind: guide
type: programming_guide
source_family: reference
title: Softmax / LogSoftmax 算子优化
description: Softmax / LogSoftmax 算子优化
tags: [softmax, logsoftmax, online_softmax, reduction, numerical_stability, 910b2, 910b3, operator_class]
confidence: inferred
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/pages/ops/softmax.md
created_at: '2026-08-10T02:56:36Z'
updated_at: '2026-08-10T02:56:36Z'
---

sources:
  - url: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/pages/ops/softmax.md
    role: primary

> 迁移备注：本卡由 `cannbot-skills_5171/ops/triton-knowledge-retrieval/knowledge/pages/ops/softmax.md` 迁移而来，原始为算子类别设计页，未做内容改写，仅转 OKF frontmatter。

# Softmax / LogSoftmax 算子优化

## 概述

Softmax 将实值向量归一化为概率分布：`softmax(x)_i = exp(x_i) / Σ exp(x_j)`。  
LogSoftmax 输出对数概率：`log_softmax(x)_i = x_i - log(Σ exp(x_j))`，数值更稳定。

在 Triton-Ascend 中，Softmax 的核心挑战是：
1. **两趟访存开销**：标准实现需两次遍历（先求 max，再求 exp+sum）
2. **数值溢出**：fp16/bf16 下 `exp(x)` 容易溢出，需减去最大值稳定化
3. **归约通信**：跨 tile 归约需要同步，影响并行度

## 检测标准

满足以下条件时归类为 Softmax：
- PyTorch 原语：`F.softmax(x, dim=...)` 或 `F.log_softmax(x, dim=...)`
- 归约沿指定维度（inner dim 或 outer dim），输出形状与输入相同
- 不含跨 batch 归约（跨 batch 属于 Reduction）

## 适用条件

| 条件 | 说明 |
|------|------|
| 数据类型 | fp16、bf16、fp32；fp8 需特殊 UB 分片处理 |
| inner dim N | N≥64 时向量化收益显著；N 超过 UB 容量需分块 |
| outer dim M | M≥1，通常每个 program 处理一行 |

## 经验路由

| 场景 | 经验ID | 提升% | 关键技术 | 经验文件 |
|------|--------|-------|----------|----------|
| 标准 Softmax（inner dim） | exp_softmax_inner_002 | 30% | 在线 max+sum 单趟，数值稳定 | yaml |
| 非 inner dim Softmax | exp_softmax_non_inner_001 | 25% | 外层维度归约，转置后处理 | yaml |
| Online Softmax（Attention 内融合） | exp_softmax_online_001 | 50% | 运行 m_i/l_i 迭代更新，消除完整 NxN 矩阵 | yaml |
| LogSoftmax 前向 | exp_log_softmax_001 | 30% | 两趟：max+sum → log-sum-exp 单核 | yaml |
| LogSoftmax 反向 | exp_log_softmax_backward_001 | 25% | 反向梯度：`dy - softmax(x) * sum(dy)` | yaml |
| Softmax 反向（inner） | exp_softmax_backward_001 | 25% | 反向：`p * (dy - dot(p, dy))` | yaml |
| Softmax 反向（非 inner） | exp_softmax_backward_non_inner_001 | 20% | 非内层维度反向，二维 tile 处理 | yaml |

## 关键技术

### Online Softmax 算法（单趟）

核心思想：维护运行最大值 `m_i` 和运行指数和 `l_i`，一次遍历完成 max+sum：

```
m_new = max(m_old, x_tile_max)
l_new = l_old * exp(m_old - m_new) + sum(exp(x_tile - m_new))
```

最终输出：`softmax(x)_i = exp(x_i - m_final) / l_final`

```python
@triton.jit
def softmax_online_kernel(x_ptr, y_ptr, n_rows, n_cols, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * n_cols

    # 在线算法：单趟计算 max 和 sum
    m = tl.full([1], float('-inf'), dtype=tl.float32)
    l = tl.zeros([1], dtype=tl.float32)
    for col in range(0, n_cols, BLOCK_N):
        cols = col + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=float('-inf')).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x))
        l = l * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new))
        m = m_new

    # 第二趟：写出归一化结果
    for col in range(0, n_cols, BLOCK_N):
        cols = col + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        y = tl.exp(x - m) / l
        tl.store(y_ptr + row * n_cols + cols, y.to(tl.float16), mask=mask)
```

### 数值稳定化

fp16/bf16 下 `exp(x)` 在 x > 65504 时溢出。**必须**在 exp 前减去行内最大值：

```python
x_stable = x - tl.max(x, axis=0)  # 防止 exp 溢出
p = tl.exp(x_stable)
p = p / tl.sum(p)
```

参考：exp_stability_001

### UB 容量限制（N 大于 192KB 时分块）

910b2/910b3 UB 约 192KB。fp16 时单行最大元素数：`192*1024/2 = 98304`。  
超出时使用分块两趟算法（exp_softmax_non_inner_001 / exp_log_softmax_001 的策略）。

## 常见陷阱

| 陷阱 | 说明 | 解决方案 |
|------|------|----------|
| fp16 exp 溢出 | x 不减 max 直接 exp，输出 Inf | 先减行内最大值（数值稳定化） |
| outer dim 效率低 | 非 inner dim 归约需要 gather-like 访问 | 转置后处理内层，或使用 non_inner 变体 |
| 两趟访存带宽瓶颈 | 独立读两次 HBM | 使用 Online Softmax 合并为单趟 |
| Attention 中单独 Softmax | 多余 HBM 写回 | 将 Softmax 融入 Attention 内核（Flash Attention） |

## 相关页面

- [ops/attention.md](op-class-attention.md) — Flash Attention 内嵌 Online Softmax，是最重要的 Softmax 融合场景
- [ops/reduction.md](op-class-reduction.md) — Reduction 原语与 Softmax 共享归约逻辑
- [ops/loss.md](op-class-loss.md) — CrossEntropy/LogSoftmax 常与 Loss 计算融合
- techniques/online_algorithms.md — Online Softmax 算法详解
- techniques/numerical_stability.md — 数值稳定性指南

## 已知实例（P3 挂载）

### Extracted Kernel 反链
- `126-softmax_kernel_non_inner.py` — extracted kernel（source: `raw_experiences/extracted_kernels/126-softmax_kernel_non_inner.py`）
- `127-softmax_kernel_inner.py` — extracted kernel（source: `raw_experiences/extracted_kernels/127-softmax_kernel_inner.py`）
- `128-softmax_backward_kernel_non_inner.py` — extracted kernel（source: `raw_experiences/extracted_kernels/128-softmax_backward_kernel_non_inner.py`）
- `129-softmax_backward_kernel_inner.py` — extracted kernel（source: `raw_experiences/extracted_kernels/129-softmax_backward_kernel_inner.py`）
- `144-log_softmax_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/144-log_softmax_kernel.py`）
- `145-log_softmax_backward_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/145-log_softmax_backward_kernel.py`）
- `054-vec_a.py` — extracted kernel（source: `raw_experiences/extracted_kernels/054-vec_a.py`）

<!-- okf:related:start -->

# 相关

- 实现示例: [FlashAttentionScoreGrad training-backward on Ascend NPU](../ops/flash-attention-score-grad.md) — 主题相关：共享 online_softmax, softmax
- 实现示例: [FlashAttentionScore training-forward on Ascend NPU](../ops/flash-attention-score.md) — 主题相关：共享 online_softmax, softmax
- 实现示例: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../ops/fused-floyd-attention.md) — 主题相关：共享 softmax
- 实现示例: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](../ops/moe-gating-top-k-softmax-v2.md) — 主题相关：共享 softmax
- 实现示例: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](../ops/moe-gating-top-k-softmax.md) — 主题相关：共享 softmax
- 实现示例: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](../ops/nsa-compress-with-cache.md) — 主题相关：共享 reduction
- 实现示例: [PromptFlashAttention inference/prefill on Ascend NPU](../ops/prompt-flash-attention.md) — 主题相关：共享 online_softmax, softmax
- 讲解接口: [跨 tile 归约用 tl.atomic_add 作为可靠原语](../runbooks/algorithm-numerics/atomic-add-cross-tile-reduction.md) — 同属优化知识，主题相关
- 讲解接口: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../runbooks/algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 主题相关：共享 numerical_stability, softmax
- 讲解接口: [统计归约必须用 fp32 累加器，禁止 fp16/bf16 直接累加](../runbooks/algorithm-numerics/fp32-reduction-accumulator.md) — 主题相关：共享 numerical_stability, reduction
- 讲解接口: [Online Softmax / Flash Attention 单遍稳定计算](../runbooks/algorithm-numerics/online-softmax-flash-attention.md) — 主题相关：共享 numerical_stability, online_softmax
- 讲解接口: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../runbooks/meta-skills/attention-fusion-failure-diagnosis.md) — 同属优化知识，主题相关
- 配套样例: [大序列 Attention 的分块计算与 Online Softmax 数值稳定](../runbooks/algorithm-numerics/traj-attention-memory_numerical.md) — 主题相关：共享 910b2, 910b3, online_softmax
- 配套样例: [低精度数值稳定性：exp 最大偏移与 sqrt 非负保护](../runbooks/algorithm-numerics/traj-general_stability-numerical_stability.md) — 主题相关：共享 910b2, 910b3, numerical_stability, softmax
- 配套样例: [Misc 算子计算优化：向量化、微内核与阶段融合](../runbooks/compute-pipeline/traj-misc-compute.md) — 主题相关：共享 910b2, 910b3, numerical_stability
- 配套样例: [Reduction Compute 优化聚合](../runbooks/compute-pipeline/traj-reduction-compute.md) — 主题相关：共享 910b2, 910b3, reduction
- 配套样例: [Softmax 计算优化：Online Softmax、Two-Pass 向量化与数值稳定](../runbooks/compute-pipeline/traj-softmax-compute.md) — 主题相关：共享 910b2, 910b3, numerical_stability, online_softmax, softmax
- 配套样例: [Sum-and-Scale 归约内存访问优化](../runbooks/memory-sram/traj-reduction-memory_access.md) — 主题相关：共享 910b2, 910b3, reduction
- 配套样例: [Softmax Backward 内存访问优化聚合](../runbooks/memory-sram/traj-softmax-memory_access.md) — 主题相关：共享 910b2, 910b3, reduction, softmax
- 配套样例: [Attention 工作流优化：中小序列标准 Attention 的三步计算、数值稳定与维度对齐](../runbooks/meta-skills/traj-attention-workflow_opt.md) — 主题相关：共享 910b2, 910b3, softmax
- 相关主题: [Fused Attention / Flash Attention 高性能算子](op-class-attention.md) — 主题相关：共享 910b2, 910b3, operator_class, softmax
- 相关主题: [卷积算子优化（Causal Conv1D）](op-class-convolution.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Elementwise 算子优化](op-class-elementwise.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Embedding / 索引拷贝算子优化](op-class-embedding.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [融合算子综合页](op-class-fused_ops.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Gather/Scatter 算子优化](op-class-gather_scatter.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Loss 函数算子优化](op-class-loss.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [MatMul / GEMM 高性能算子](op-class-matmul.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [归一化算子优化](op-class-normalization.md) — 主题相关：共享 910b2, 910b3, numerical_stability, operator_class, softmax
- 相关主题: [归约算子优化](op-class-reduction.md) — 主题相关：共享 910b2, 910b3, numerical_stability, operator_class, reduction
- 相关主题: [RoPE（旋转位置编码）算子优化](op-class-rope.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [solve_tril 算子优化（三角矩阵求逆）](op-class-solve_tril.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [SSM / 状态空间模型算子优化](op-class-ssm.md) — 主题相关：共享 910b2, 910b3, operator_class

<!-- okf:related:end -->
