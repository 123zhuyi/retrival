---
title: 形状驱动算子决策树
category: workflows
tags: [routing, shape, decision_tree, pattern_detection]
hardware: [910b2, 910b3]
last_updated: 2026-04-27
---

# 形状驱动算子决策树

> **LLM 使用说明**：当你拿到一个 PyTorch 算子但不确定走哪个 ops 页时，按此决策树逐级判断，最终落到具体的经验 ID 和代码骨架。

---

## 第一步：识别计算结构

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

---

## MatMul 路径

```
含 tl.dot / mm / bmm
│
├── 同时含 softmax(QK^T / sqrt(d_k)) @ V ?
│   └── YES → Attention 路径（优先级更高）
│
├── 形状约束判断：
│   ├── M < 128 或 N < 128 或 K < 128
│   │   └── → exp_matmul_micro_tiling_001 (16×16×16 微 Tiling)
│   │
│   ├── M ≥ 512 AND N ≥ 512（大正方形矩阵）
│   │   └── → exp_matmul_diag_tiling_001 + skill_tiling_diagonal (25-35%)
│   │       使用对角 core mapping，GROUP_SIZE autotune {8,16,32}
│   │
│   ├── M ≥ 256 AND N ≥ 256 AND K ≥ 1024（标准大矩阵）
│   │   └── → exp_matmul_001 (40%)，L0级 128×128×128 Tiling
│   │
│   └── 其他（中型矩阵）
│       └── → 按转置模式路由：
│           ├── NN → exp_matmul_003 (10%)，512B 行宽对齐
│           ├── NT → exp_matmul_004 (10%)，K宽统一分块
│           └── TT → exp_matmul_005 (32%)，双转置 M=256,K=256,N=128
│
├── 后接其他算子（融合机会）？
│   ├── 后接 GELU/SiLU/ReLU → ops/fused_ops.md Fusion-1 (30%)
│   ├── 后接 RMSNorm + RoPE → ops/fused_ops.md Fusion-2 (45%)
│   ├── 后接 Bias → ops/fused_ops.md Fusion-7 (35%)
│   └── 前有 Scatter（MoE） → ops/fused_ops.md Fusion-6
│
└── → 详细页面：ops/matmul.md
    骨架代码：skeletons/gemm_basic.py
```

---

## Attention 路径

```
含 softmax(QK^T / sqrt(d_k)) @ V
│
├── seq_len 判断：
│   ├── seq_len ≤ 512
│   │   └── → exp_fused_attention_001 (15%)，标准三步流程
│   │
│   └── seq_len > 512（或不确定，优先 Flash）
│       └── → exp_fused_attention_002 (50%)，Flash Attention + Online Softmax
│
├── 变体判断：
│   ├── GQA / MQA（KV head 数 < Q head 数）
│   │   └── → exp_flash_attention_gqa_001 (40%)，SIMD index select 共享 KV head
│   │
│   ├── 分页 KV cache（推理场景）
│   │   └── → exp_flash_attention_paged_001 (35%)，Block Pointer 分页访问
│   │
│   ├── 稀疏 attention（部分 token 间无注意力）
│   │   └── → exp_flash_attention_sparse_001 (30%)
│   │
│   └── SDPA（torch.nn.functional.scaled_dot_product_attention 对齐）
│       └── → exp_flash_attention_sdpa_001 (35%)
│
└── → 详细页面：ops/attention.md
    骨架代码：skeletons/flash_attention_causal.py
```

---

## Normalization 路径

```
含 mean/var/rms 归一化
│
├── 算子类型：
│   ├── RMSNorm（无均值，y = x / rms(x) * w）
│   │   ├── n_cols ≤ 512（小 hidden）→ 无 tiling，单次装入寄存器
│   │   ├── n_cols ≤ 8192（中等 hidden）→ exp_rmsnorm_001，two-pass，fp32 累加
│   │   └── n_cols > UB 上限 → exp_rmsnorm_npu_001 (35%)，列分块 two-pass
│   │
│   ├── LayerNorm（含 mean，y = (x-mean)/std * w + b）
│   │   ├── n_cols 中等 → Welford 单遍（exp_layernorm_npu_001）
│   │   └── n_cols 大 → three-pass（mean → variance → normalize）
│   │
│   └── Softmax（作为独立算子）
│       └── → exp_online_softmax_001 (35%)，log-sum-exp 单遍
│
├── 批量大（n_rows >> num_cores）？
│   └── YES → exp_rmsnorm_001(086) Persistent kernel (35%)
│
├── 后接其他算子（融合机会）？
│   ├── 前有 Skip Add → ops/fused_ops.md Fusion-3 (30%)
│   └── 后接量化 → ops/fused_ops.md Fusion-4 (50%)
│
└── → 详细页面：ops/normalization.md
    骨架代码：skeletons/rmsnorm_fused.py
```

---

## Reduction 路径

```
含 tl.sum / tl.max / tl.min(dim=) / logsumexp
│
├── 归约轴 K 的大小：
│   ├── K < 4096 → 单核单 pass 归约，fp32 累加器
│   └── K ≥ 4096 → Split-K 并行：
│       ├── 精度要求不高 → exp_splitk_reduction_001 (35%)，原子 add
│       └── 精度要求高 → exp_splitk_two_pass_001 (40%)，两阶段归约
│
├── 特殊类型：
│   ├── LogSumExp → exp_reduction_logsumexp_001 (35%)，log-sum-exp trick
│   ├── CrossEntropy → exp_reduction_cross_entropy_001 (30%)，融合 log-softmax
│   ├── ArgMax（同时需要 max 值）→ exp_reduction_argmax_001 (25%)，联合单遍
│   └── CumSum → exp_cumsum_001 (25%)，scan 原语
│
├── 数值稳定性检查（必做）：
│   └── fp16/bf16 输入 → 必须使用 fp32 累加器
│
└── → 详细页面：ops/reduction.md
```

---

## Gather / Scatter 路径

```
含 tensor[indices] 或 out[indices] = val
│
├── 场景类型：
│   ├── MoE 路由（tokens → experts）
│   │   ├── expert_capacity × num_experts ≥ 1024 → exp_gs_bin_003 (35%)
│   │   └── 专家负载不均 → exp_gs_bin_005 work-stealing (30%)
│   │
│   ├── Token 排列 / 重组 → exp_gs_bin_004 (25%)
│   │
│   ├── 单纯 gather → exp_gs_mem_001 (25%)，寄存器缓冲聚合
│   │
│   └── 单纯 scatter → exp_gs_mem_002 (30%)，向量化散射写入
│       + 若索引无规律 → exp_gs_mem_005 (30%)，排序索引提升连续性
│
├── 后接 GEMM（MoE wgrad）？
│   └── YES → scatter 后使用 ops/matmul.md 的 GEMM 经验
│
└── → 详细页面：ops/gather_scatter.md
    骨架代码：skeletons/moe_dispatch.py
```

---

## Elementwise 路径

```
输入输出同形状，每元素独立
│
├── 内存连续性检查：
│   └── 非连续 tensor → 先在 kernel 外 .contiguous()（exp_elem_vec_002）
│
├── BLOCK_SIZE 选择（autotune）：
│   └── {256, 512, 1024, 2048, 4096}（exp_elem_vec_003）
│
├── 含 int64 比较或加法？
│   └── YES → 转换为 fp32（exp_elem_vec_006，25%）
│
├── 融合机会（是否有前序 matmul）？
│   └── YES（前序是 GEMM） → 内联进 matmul kernel（exp_elem_fusion_001，35%）
│
├── 多个连续 elementwise 算子？
│   └── YES（2-3 个） → 合并为单 kernel（exp_elem_fusion_002，25%）
│
└── → 详细页面：ops/elementwise.md
```

---

## 性能约束快速校验

进入具体路径后，始终检查：

```
□ BLOCK_SIZE * dtype_bytes ≤ 192KB/3（UB 约束）
□ BLOCK_N * dtype_bytes 为 512B 整数倍（matmul 行宽对齐）
□ grid_size ≤ num_ai_cores
□ int64 比较/加法 → .to(tl.float32)
□ tl.swizzle2d 禁用 → grouped_launch_diagonal
□ fp16/bf16 归约 → fp32 累加器
```

→ 完整清单：[hardware/constraints_cheatsheet](../hardware/constraints_cheatsheet.md)

---

## 相关页面

- [workflows/optimization_flow](optimization_flow.md) — 更完整的优化工作流（包含 profiling 反馈循环）
- [experience_index](../experience_index.md) — 所有经验的扁平化速查表
- [ops/fused_ops](../ops/fused_ops.md) — 融合组合路由
- [hardware/constraints_cheatsheet](../hardware/constraints_cheatsheet.md) — 约束清单

## 被以下页面引用
- [knowledge/index](../../index.md) — 工作流入口
- [workflows/optimization_flow](optimization_flow.md) — 形状路由步骤
