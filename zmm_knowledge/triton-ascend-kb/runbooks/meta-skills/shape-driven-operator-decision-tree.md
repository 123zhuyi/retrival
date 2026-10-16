---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: meta-skills
dimension: "6.1"
title: 形状驱动算子决策树
description: 拿到 PyTorch 算子时，按计算结构逐级路由到算子类别与对应参考卡，再下沉到具体优化 runbook。
tags: [shape_routing, decision_tree, operator_classification, pattern_detection]
confidence: verified
status: verified
created_at: '2026-08-10T00:00:00Z'
updated_at: '2026-08-10T00:00:00Z'
---

## DG-9 形状驱动算子决策树

【触发条件】：拿到一个 PyTorch 算子 / 原子分解列表 / 算子描述，不确定它属于哪类 Triton-Ascend kernel，也不确定该查哪张参考卡或哪条 runbook。

【判定/动作规则】：

### 第一步：识别计算结构

```
输入：PyTorch 算子代码 / 原子分解列表 / 算子描述

├── 含 mm / bmm / matmul / linear / einsum(ik,kj->ij) ?
│   └── YES → 进入 [MatMul 路径] ↓
│
├── 含 softmax(QK^T) @ V 结构 ?
│   └── YES → 进入 [Attention 路径] ↓
│
├── 含 mean / var / rms 后接归一化 ?
│   └── YES → 进入 [Normalization 路径] ↓
│
├── 含 sum / max / min(dim=) / logsumexp ?
│   └── YES → 进入 [Reduction 路径] ↓
│
├── 含 tensor[indices] 或 out[indices] = val ?
│   └── YES → 进入 [Gather/Scatter 路径] ↓
│
└── 输入输出同形状，无跨元素依赖 ?
    └── YES → 进入 [Elementwise 路径] ↓
```

### MatMul 路径

- 含 `softmax(QK^T / sqrt(d_k)) @ V` → 走 **Attention 路径**（优先级更高）
- 否则按 shape 选 tiling 策略：
  - `M < 128 或 N < 128 或 K < 128` → 微 tiling（16×16×16）
  - `M ≥ 512 AND N ≥ 512` → 对角 core mapping，GROUP_SIZE autotune `{8,16,32}`（见 [对角 core mapping](../compute-pipeline/diagonal-core-mapping-l2-bank-conflict.md)）
  - `M ≥ 256 AND N ≥ 256 AND K ≥ 1024` → 标准大矩阵 128×128×128 tiling
  - 中型矩阵 → 按 NN/NT/TT 转置模式分别处理
- 参考卡：[MatMul 类算子设计](../../reference/op-class-matmul.md)
- 骨架：[GEMM Basic Skeleton](../../reference/skeleton-gemm_basic.md)

### Attention 路径

- `seq_len ≤ 512` → 标准三步流程
- `seq_len > 512` → Flash Attention + Online Softmax
- 变体：GQA/MQA、分页 KV cache、稀疏 attention、SDPA 对齐
- 参考卡：[Attention 类算子设计](../../reference/op-class-attention.md)
- 骨架：[Flash Attention Causal Skeleton](../../reference/skeleton-flash_attention_causal.md)

### Normalization 路径

- RMSNorm / LayerNorm / Softmax 分别按 hidden size 与批量选择 tiling
- 前有 Skip Add 或后接量化 → 考虑融合
- 参考卡：[Normalization 类算子设计](../../reference/op-class-normalization.md)
- 骨架：[RMSNorm Fused Skeleton](../../reference/skeleton-rmsnorm_fused.md)

### Reduction 路径

- 归约轴 `K < 4096` → 单核单 pass，fp32 累加
- `K ≥ 4096` → Split-K 并行，按精度要求选 atomic add 或 two-pass
- 特殊类型：LogSumExp、CrossEntropy、ArgMax、CumSum
- 参考卡：[Reduction 类算子设计](../../reference/op-class-reduction.md)

### Gather / Scatter 路径

- MoE 路由 → tokens → experts，按 expert_capacity 与负载均衡选策略
- Token 排列 / 重组
- 单纯 gather / scatter
- 参考卡：[Gather/Scatter 类算子设计](../../reference/op-class-gather_scatter.md)
- 骨架：[MoE Dispatch Skeleton](../../reference/skeleton-moe_dispatch.md)

### Elementwise 路径

- 非连续 tensor → 先 `.contiguous()`
- BLOCK_SIZE autotune `{256, 512, 1024, 2048, 4096}`
- 含 int64 比较/加法 → 转 fp32
- 多个连续 elementwise → 合并为单 kernel
- 参考卡：[Elementwise 类算子设计](../../reference/op-class-elementwise.md)

### 进入路径后的通用校验

```
□ BLOCK_SIZE * dtype_bytes ≤ UB 预算（见 memory-sram）
□ matmul 行宽 512B 对齐（见 [Cube-Vector 行宽 512B 对齐](../memory-sram/row-width-512b-align.md)）
□ grid_size ≤ num_ai_cores（见 compute-pipeline grid 约束）
□ int64 比较/加法 → .to(tl.float32)
□ tl.swizzle2d 禁用 → grouped_launch_diagonal
□ fp16/bf16 归约 → fp32 累加器
```

【判定依据】：算子分类决定后续检索空间。错误分类会导致查错 runbook 或选错 skeleton，浪费调优轮次。

【关联优化点】：
- 分类完成后，具体 tiling/内存/数值优化点分别见 `runbooks/memory-sram/`、`runbooks/compute-pipeline/`、`runbooks/algorithm-numerics/`。
- 算子特定实现细节见 `reference/op-class-*.md` 与 `reference/skeleton-*.md`。

**坏实践（反例）**：拿到算子不分类，直接套用某个固定 skeleton（如所有 2D 运算都用 gemm），或把 attention 当普通 matmul 处理。后果：tiling 与内存访问模式错配，性能无法达标。✅ 正解即先按本决策树分类，再下沉到对应参考卡与 runbook。

**经验 ID 映射（形状/场景 → 待查 experience）**：

| 路径分支 | 触发条件 | 推荐经验/骨架 |
|---|---|---|
| MatMul 微 tiling | `M < 128` 或 `N < 128` 或 `K < 128` | `exp_matmul_micro_tiling_001`（16×16×16） |
| MatMul 大正方形 | `M ≥ 512 AND N ≥ 512` | `exp_matmul_diag_tiling_001` + 对角 core mapping；GROUP_SIZE `{8,16,32}` |
| MatMul 标准大矩阵 | `M ≥ 256 AND N ≥ 256 AND K ≥ 1024` | `exp_matmul_001`（128×128×128 tiling，40%） |
| MatMul NN | 中型矩阵 + NN 转置 | `exp_matmul_003`（512B 行宽对齐，10%） |
| MatMul NT | 中型矩阵 + NT 转置 | `exp_matmul_004`（K 宽统一分块） |
| MatMul TT | 中型矩阵 + TT 转置 | `exp_matmul_005`（双转置，32%） |
| Attention 短序列 | `seq_len ≤ 512` | `exp_fused_attention_001`（15%） |
| Attention 长序列 | `seq_len > 512` | `exp_fused_attention_002`（Flash + Online Softmax，50%） |
| Attention GQA/MQA | KV head < Q head | `exp_flash_attention_gqa_001`（40%） |
| Attention 分页 KV | 推理场景 | `exp_flash_attention_paged_001`（35%） |
| Attention 稀疏 | 部分 token 无注意力 | `exp_flash_attention_sparse_001`（30%） |
| Attention SDPA | 对齐 torch SDPA | `exp_flash_attention_sdpa_001`（35%） |
| RMSNorm 小 hidden | `n_cols ≤ 512` | 无 tiling，单次装入 |
| RMSNorm 中 hidden | `512 < n_cols ≤ 8192` | `exp_rmsnorm_001`（two-pass，fp32 累加） |
| RMSNorm 大 hidden | `n_cols > UB 上限` | `exp_rmsnorm_npu_001`（35%） |
| LayerNorm | 含 mean | `exp_layernorm_npu_001`（Welford） |
| Softmax 独立 | 独立 softmax | `exp_online_softmax_001`（35%） |
| Reduction 小 K | `K < 4096` | 单核单 pass，fp32 累加 |
| Reduction 大 K 低精度 | `K ≥ 4096`，精度要求不高 | `exp_splitk_reduction_001`（atomic add，35%） |
| Reduction 大 K 高精度 | `K ≥ 4096`，精度要求高 | `exp_splitk_two_pass_001`（two-pass，40%） |
| LogSumExp | 特殊归约 | `exp_reduction_logsumexp_001`（35%） |
| CrossEntropy | 特殊归约 | `exp_reduction_cross_entropy_001`（30%） |
| ArgMax | 特殊归约 | `exp_reduction_argmax_001`（25%） |
| CumSum | 特殊归约 | `exp_cumsum_001`（25%） |
| Gather/Scatter MoE 大容量 | `expert_capacity × num_experts ≥ 1024` | `exp_gs_bin_003`（35%） |
| Gather/Scatter MoE 负载不均 | 专家负载不均 | `exp_gs_bin_005`（work-stealing，30%） |
| Gather/Scatter Token 重组 | Token 排列/重组 | `exp_gs_bin_004`（25%） |
| Gather/Scatter 单纯 gather | 单纯 gather | `exp_gs_mem_001`（25%） |
| Gather/Scatter 单纯 scatter | 单纯 scatter | `exp_gs_mem_002`（30%） |
| Gather/Scatter 无序索引 | scatter 索引无规律 | `exp_gs_mem_005`（30%） |
| Elementwise 非连续 | 输入非连续 | `exp_elem_vec_002`（先 contiguous） |
| Elementwise BLOCK_SIZE | 通用 | `exp_elem_vec_003`（BLOCK_SIZE autotune） |
| Elementwise int64 | 含 int64 比较/加法 | `exp_elem_vec_006`（转 fp32，25%） |
| Elementwise 融合 GEMM | 前序 GEMM | `exp_elem_fusion_001`（35%） |
| Elementwise 多算子合并 | 2–3 个连续 elementwise | `exp_elem_fusion_002`（25%） |

（P3 阶段把上述 experience YAML 反链挂载到本卡对应分支）

<!-- okf:related:start -->

# 相关

- 实践案例: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](../../ops/moe-token-permute-with-routing-map.md) — 同属优化知识，主题相关
- 相关主题: [Cross-kernel Optimization Transfer: token loop vs w loop decision](../compute-pipeline/exp_causal_conv1d_update_006.md) — 同属优化知识，主题相关
- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](../memory-sram/exp_ascend_large_shape_block_001.md) — 同属优化知识，主题相关
- 相关主题: [Tiling 预算与布局驱动的块大小选择](../memory-sram/tiling-budget-and-layout-driven-selection.md) — 同属优化知识，主题相关
- 相关主题: [大 shape 用交错循环：grid 固定为核心数，kernel 内按步长遍历](../portability/exp_ascend_grid_large_shape_001.md) — 同属优化知识，主题相关
- 相关主题: [Grid/Block 配置校验：乘积 ≤65535、≤3 维 tuple、大 shape 交错循环](exp_debug_003.md) — 同属优化知识，主题相关
- 相关主题: [流水图驱动优化：先可视化找气泡，按气泡边类型分派优化手段](pipeline-diagram-driven-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
