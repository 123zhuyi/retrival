---
title: 经验速查索引（扁平化）
category: index
tags: [experience, index, routing]
hardware: [910b2, 910b3]
last_updated: 2026-07-07
---

# 经验速查索引

> **LLM 使用说明**：根据你的算子条件（形状/数据类型/瓶颈）在下表中匹配经验ID，然后：
> 1. 点击「详细页面」列跳转 ops wiki 查看背景技术和代码模板
> 2. 点击「经验文件」列直接读取 `experiences/<category>/<id>.yaml` 获取可用代码片段
>
> 全量经验索引（303条）：[`../../raw/experiences/_index.yaml`](../../raw/experiences/_index.yaml)

---

## MatMul / GEMM 经验

### 基础经验

| 经验 ID | 条件 | 数据类型 | 转置 | 提升% | 核心技术 | 详细页面 | 经验文件 |
|---------|------|---------|------|-------|---------|---------|---------|
| exp_matmul_001 | M≥256, N≥256, K≥1024 | fp16, bf16 | NN | 40% | L0级128×128×128 Tiling | [ops/matmul](ops/matmul.md) | [yaml](../../raw/experiences/matmul/exp_matmul_001.yaml) |
| exp_matmul_002 | M≥128, N≥128, K≥128 | fp16 | NN (NZ格式) | 35% | 量化融合+4x K循环展开 | [ops/matmul](ops/matmul.md) | [yaml](../../raw/experiences/matmul/exp_matmul_002.yaml) |
| exp_matmul_003 | M≥128, N≥256, K≥256 | fp16/bf16/fp32 | NN | 10% | 512B行宽对齐块划分 | [ops/matmul](ops/matmul.md) | [yaml](../../raw/experiences/matmul/exp_matmul_003.yaml) |
| exp_matmul_004 | M≥128, N≥128, K≥256 | fp16/bf16/fp32 | NT | 10% | K宽统一分块 | [ops/matmul](ops/matmul.md) | [yaml](../../raw/experiences/matmul/exp_matmul_004.yaml) |
| exp_matmul_005 | M≥256, N≥128, K≥256 | fp16/bf16/fp32 | TT | 32% | 双转置 M=256,K=256,N=128 | [ops/matmul](ops/matmul.md) | [yaml](../../raw/experiences/matmul/exp_matmul_005.yaml) |

### Tiling 与 Core Mapping 经验

| 经验 ID | 条件 | 提升% | 核心技术 | 详细页面 | 经验文件 |
|---------|------|-------|---------|---------|---------|
| exp_matmul_diag_tiling_001 | M≥512, N≥512, K≥256 | 25% | 对角8×8 Tiling，消除bank conflict | [ops/matmul](ops/matmul.md), [techniques/core_mapping](techniques/core_mapping.md) | [yaml](../../raw/experiences/matmul/exp_matmul_diag_tiling_001.yaml) |
| exp_matmul_small_tile_001 | M/N/K < 128（任一维小） | 20% | 小tile微矩阵乘 | [ops/matmul](ops/matmul.md) | [yaml](../../raw/experiences/matmul/exp_matmul_small_tile_001.yaml) |
| exp_matmul_ascend_diag_001 | M,N≥512，Ascend专用 | 25-35% | 对角核映射，最大化L2复用 | [techniques/core_mapping](techniques/core_mapping.md) | [yaml](../../raw/experiences/matmul/exp_matmul_ascend_diag_001.yaml) |

### 调度与融合经验

| 经验 ID | 条件 | 提升% | 核心技术 | 详细页面 | 经验文件 |
|---------|------|-------|---------|---------|---------|
| exp_fused_matmul_rms_001 | 后接RMSNorm | 30% | GEMM+RMSNorm单核 | [techniques/operator_fusion](techniques/operator_fusion.md) | [yaml](../../raw/experiences/matmul/exp_fused_matmul_rms_001.yaml) |
| exp_fused_matmul_rope_001 | 后接RoPE | 25% | GEMM+旋转位置编码融合 | [techniques/operator_fusion](techniques/operator_fusion.md) | [yaml](../../raw/experiences/matmul/exp_fused_matmul_rope_001.yaml) |
| exp_fused_matmul_rms_rope_001 | 后接RMSNorm+RoPE | 35% | GEMM+RMSNorm+RoPE三合一 | [techniques/operator_fusion](techniques/operator_fusion.md) | [yaml](../../raw/experiences/matmul/exp_fused_matmul_rms_rope_001.yaml) |
| exp_fused_matmul_rmsnorm_001 | 后接RMSNorm（另一版本） | 30% | GEMM+RMSNorm融合 | [techniques/operator_fusion](techniques/operator_fusion.md) | [yaml](../../raw/experiences/matmul/exp_fused_matmul_rmsnorm_001.yaml) |
| exp_grouped_matmul_001 | MoE grouped GEMM | 30% | 分组矩阵乘 | [ops/matmul](ops/matmul.md) | [yaml](../../raw/experiences/matmul/exp_grouped_matmul_001.yaml) |
| exp_addmm_ascend_001 | GEMM+bias（Ascend） | 20% | addmm Ascend专用路径 | [ops/matmul](ops/matmul.md) | [yaml](../../raw/experiences/matmul/exp_addmm_ascend_001.yaml) |

---

## Attention 经验

| 经验 ID | 条件 | 数据类型 | 提升% | 核心技术 | 详细页面 | 经验文件 |
|---------|------|---------|-------|---------|---------|---------|
| exp_fused_attention_001 | seq_len≥128, d_k≥64, heads≥8 | fp16/bf16/fp32 | 15% | 标准三步QK^T→Softmax→AV | [ops/attention](ops/attention.md) | [yaml](../../raw/experiences/attention/exp_fused_attention_001.yaml) |
| exp_fused_attention_002 | seq_len≥1024, d_k≥64, heads≥8 | fp16/bf16 | 50% | Flash Attention，O(S²)→O(S) | [ops/attention](ops/attention.md) | [yaml](../../raw/experiences/attention/exp_fused_attention_002.yaml) |
| exp_attn_fp8_ub_slice_001 | seq≥1024，fp8输入 | fp8 | 35% | UB溢出分片处理fp8 | [ops/attention](ops/attention.md) | [yaml](../../raw/experiences/attention/exp_attn_fp8_ub_slice_001.yaml) |
| exp_attn_ascend_dual_pipeline_001 | seq≥2048 | fp16/bf16 | 45% | 双流水线软件流水 | [ops/attention](ops/attention.md) | [yaml](../../raw/experiences/attention/exp_attn_ascend_dual_pipeline_001.yaml) |
| exp_flashattn_splitk_gqa_001 | GQA，大seq | fp16/bf16 | 40% | Split-K GQA+分页KV | [ops/attention](ops/attention.md) | [yaml](../../raw/experiences/attention/exp_flashattn_splitk_gqa_001.yaml) |
| exp_paged_attn_flash_001 | 推理，分页KV cache | fp16/bf16 | 35% | Block Pointer分页注意力 | [ops/attention](ops/attention.md) | [yaml](../../raw/experiences/attention/exp_paged_attn_flash_001.yaml) |
| exp_flashattn_gqa_001 | GQA/MQA | fp16/bf16 | 40% | SIMD index select多头复用 | [ops/attention](ops/attention.md) | [yaml](../../raw/experiences/attention/exp_flashattn_gqa_001.yaml) |
| exp_sdpa_flash_001 | PyTorch SDPA对齐 | fp16/bf16 | 35% | SDPA+Cube hints | [ops/attention](ops/attention.md) | [yaml](../../raw/experiences/attention/exp_sdpa_flash_001.yaml) |
| exp_sparse_attn_topk_001 | 稀疏注意力 prefill | fp16/bf16 | 30% | top-k稀疏mask | [ops/attention](ops/attention.md) | [yaml](../../raw/experiences/attention/exp_sparse_attn_topk_001.yaml) |
| exp_flash_attn_001 | 基础Flash Attention | fp16/bf16 | 45% | 标准FlashAttention实现 | [ops/attention](ops/attention.md) | [yaml](../../raw/experiences/attention/exp_flash_attn_001.yaml) |
| exp_flashattn_fwd_001 | Flash Attention前向 | fp16/bf16 | 45% | FA前向pass | [ops/attention](ops/attention.md) | [yaml](../../raw/experiences/attention/exp_flashattn_fwd_001.yaml) |

---

## Elementwise / 激活经验

| 经验 ID | 条件 | 提升% | 核心技术 | 详细页面 | 经验文件 |
|---------|------|-------|---------|---------|---------|
| exp_silu_fwd_001 | SiLU激活 | 25% | SiLU前向向量化 | [ops/elementwise](ops/elementwise.md) | [yaml](../../raw/experiences/activation/exp_silu_fwd_001.yaml) |
| exp_gelu_fwd_001 | GELU激活 | 25% | GELU前向向量化 | [ops/elementwise](ops/elementwise.md) | [yaml](../../raw/experiences/activation/exp_gelu_fwd_001.yaml) |
| exp_swiglu_fwd_001 | SwiGLU门控激活 | 30% | SwiGLU前向 | [ops/elementwise](ops/elementwise.md) | [yaml](../../raw/experiences/activation/exp_swiglu_fwd_001.yaml) |
| exp_swiglu_quant_001 | SwiGLU+量化 | 35% | SwiGLU+量化单核 | [ops/elementwise](ops/elementwise.md) | [yaml](../../raw/experiences/activation/exp_swiglu_quant_001.yaml) |
| exp_fused_gating_001 | 门控激活融合 | 30% | gating+激活融合 | [techniques/operator_fusion](techniques/operator_fusion.md) | [yaml](../../raw/experiences/activation/exp_fused_gating_001.yaml) |

---

## Normalization 经验

| 经验 ID | 条件 | 提升% | 核心技术 | 详细页面 | 经验文件 |
|---------|------|-------|---------|---------|---------|
| exp_fused_rms_norm_001 | Residual Add+RMSNorm | 35% | 原地residual更新同时计算方差 | [techniques/operator_fusion](techniques/operator_fusion.md) | [yaml](../../raw/experiences/normalization/exp_fused_rms_norm_001.yaml) |
| exp_fused_add_layernorm_001 | Skip连接+LayerNorm | 30% | Skip Add + LayerNorm融合单核 | [techniques/operator_fusion](techniques/operator_fusion.md) | [yaml](../../raw/experiences/normalization/exp_fused_add_layernorm_001.yaml) |
| exp_fused_norm_rope_001 | RMSNorm+RoPE | 35% | Norm+RoPE融合 | [techniques/operator_fusion](techniques/operator_fusion.md) | [yaml](../../raw/experiences/normalization/exp_fused_norm_rope_001.yaml) |
| exp_fused_qkv_rmsnorm_rope_001 | QKV分割+RMSNorm+RoPE | 40% | QKV+Norm+RoPE三合一 | [techniques/operator_fusion](techniques/operator_fusion.md) | [yaml](../../raw/experiences/normalization/exp_fused_qkv_rmsnorm_rope_001.yaml) |
| exp_gemma_rmsnorm_001 | Gemma风格(1+w)缩放 | 35% | (1+weight)融合归一化乘法 | [ops/normalization](ops/normalization.md) | [yaml](../../raw/experiences/matmul/exp_gemma_rmsnorm_001.yaml) |
| exp_fused_split_qk_norm_001 | Q和K分别做RMSNorm | 35% | Q/K RMSNorm单核融合 | [techniques/operator_fusion](techniques/operator_fusion.md) | [yaml](../../raw/experiences/normalization/exp_fused_split_qk_norm_001.yaml) |
| exp_groupnorm_001 | GroupNorm | 25% | 分组归一化 | [ops/normalization](ops/normalization.md) | [yaml](../../raw/experiences/normalization/exp_groupnorm_001.yaml) |

---

## Reduction 经验

| 经验 ID | 条件 | 提升% | 核心技术 | 详细页面 | 经验文件 |
|---------|------|-------|---------|---------|---------|
| exp_reduce_split_001 | K≥4096，可并行 | 35% | Split归约 | [ops/reduction](ops/reduction.md) | [yaml](../../raw/experiences/reduction/exp_reduce_split_001.yaml) |
| exp_backward_sum_001 | 反向传播sum归约 | 25% | 反向sum优化 | [ops/reduction](ops/reduction.md) | [yaml](../../raw/experiences/reduction/exp_backward_sum_001.yaml) |
| exp_sum_scale_001 | sum+scale组合 | 20% | sum后缩放融合 | [ops/reduction](ops/reduction.md) | [yaml](../../raw/experiences/reduction/exp_sum_scale_001.yaml) |

---

## Gather / Scatter / MoE 经验

| 经验 ID | 条件 | 提升% | 核心技术 | 详细页面 | 经验文件 |
|---------|------|-------|---------|---------|---------|
| exp_gather_001 | 标准gather | 25% | 向量化gather | [ops/gather_scatter](ops/gather_scatter.md) | [yaml](../../raw/experiences/gather_scatter/exp_gather_001.yaml) |
| exp_scatter_001 | 标准scatter | 25% | 向量化scatter | [ops/gather_scatter](ops/gather_scatter.md) | [yaml](../../raw/experiences/gather_scatter/exp_scatter_001.yaml) |
| exp_moe_binned_scatter_001 | MoE binned scatter | 30% | Bin分组scatter | [ops/gather_scatter](ops/gather_scatter.md) | [yaml](../../raw/experiences/gather_scatter/exp_moe_binned_scatter_001.yaml) |
| exp_moe_gather_001 | MoE gather | 30% | MoE输出聚合 | [ops/gather_scatter](ops/gather_scatter.md) | [yaml](../../raw/experiences/gather_scatter/exp_moe_gather_001.yaml) |
| exp_moe_scatter_001 | MoE scatter路由 | 30% | MoE dispatch scatter | [ops/gather_scatter](ops/gather_scatter.md) | [yaml](../../raw/experiences/gather_scatter/exp_moe_scatter_001.yaml) |
| exp_scatter_npu_001 | Ascend专用scatter | 25% | NPU scatter路径 | [ops/gather_scatter](ops/gather_scatter.md) | [yaml](../../raw/experiences/gather_scatter/exp_scatter_npu_001.yaml) |

---

## RoPE 经验

| 经验 ID | 条件 | 提升% | 核心技术 | 详细页面 | 经验文件 |
|---------|------|-------|---------|---------|---------|
| exp_rope_001 | 标准RoPE，head_dim≥64 | 15% | cos/sin 复用 + persistent grid-stride + constexpr INVERSE | [ops/rope](ops/rope.md) | [yaml](../../raw/experiences/rope/exp_rope_001.yaml) |
| exp_rope_npu_001 | Ascend NPU RoPE，hd≥64 | 35% | cos/sin 复用 + software pipeline + 前反向共用 | [ops/rope](ops/rope.md) | [yaml](../../raw/experiences/rope/exp_rope_npu_001.yaml) |
| exp_rope_forward_001 | RoPE前向，varlen | 25% | ALIGNED 路径单次向量加载 + do_not_specialize | [ops/rope](ops/rope.md) | [yaml](../../raw/experiences/rope/exp_rope_forward_001.yaml) |
| exp_rope_backward_001 | RoPE反向，varlen/KV cache | 25% | grid-stride + IS_VARLEN + HAS_KV_LENS | [ops/rope](ops/rope.md) | [yaml](../../raw/experiences/rope/exp_rope_backward_001.yaml) |
| exp_partial_rope_001 | Partial RoPE（Llama-3），ROPE_DIM≥32 | 25% | 跳过 NOPE_DIM + 向量化前/后半 | [ops/rope](ops/rope.md) | [yaml](../../raw/experiences/rope/exp_partial_rope_001.yaml) |
| exp_qwen2vl_mrope_001 | Multi-RoPE（Qwen2-VL），hd≥128 | 25% | t/h/w 三维融合 + head blocking 复用 | [ops/rope](ops/rope.md) | [yaml](../../raw/experiences/rope/exp_qwen2vl_mrope_001.yaml) |
| exp_rope_llama4_npu_001 | Llama-4 NPU RoPE，hd≥32 | 25% | UB tiling + FMA + interleaved 布局 | [ops/rope](ops/rope.md) | [yaml](../../raw/experiences/rope/exp_rope_llama4_npu_001.yaml) |
| exp_rope_siso_001 | SISO 布局 RoPE | 25% | 3D tiling + 非交错向量化路径 | [ops/rope](ops/rope.md) | [yaml](../../raw/experiences/rope/exp_rope_siso_001.yaml) |
| exp_rope_triton_001 | Triton RoPE | 25% | 向量化 feat_offset + fused cos/sin 旋转 | [ops/rope](ops/rope.md) | [yaml](../../raw/experiences/rope/exp_rope_triton_001.yaml) |
| exp_triton_rope_001 | Neox/非Neox 双路 RoPE，rope_dim≥2 | 25% | IS_NEOX_STYLE constexpr + partial rope_dim 支持 | [ops/rope](ops/rope.md) | [yaml](../../raw/experiences/rope/exp_triton_rope_001.yaml) |

---

## 通用优化经验（跨算子复用）

### 标量消除 / 计算优化

| 经验 ID | 条件 | 提升% | 核心技术 | 详细页面 | 经验文件 |
|---------|------|-------|---------|---------|---------|
| exp_perf_012 | 所有含 i32 LT/GT/LE/GE 比较的 kernel | 35% | i32 整数比较 → fp32 比较 | [techniques/data_types](techniques/data_types.md) | [yaml](../../raw/experiences/general_perf/exp_perf_012.yaml) |
| exp_perf_013 | 所有使用 `min()` 运行时求值的 kernel | 15% | `min()` → `tl.where` | [techniques/compute_and_branch](techniques/compute_and_branch.md) | [yaml](../../raw/experiences/general_perf/exp_perf_013.yaml) |
| exp_perf_014 | 所有使用 `tl.cdiv` 的循环边界 | 10% | `tl.cdiv` → 编译期常量 | [techniques/compute_and_branch](techniques/compute_and_branch.md) | [yaml](../../raw/experiences/general_perf/exp_perf_014.yaml) |
| exp_ascend_int_compare_scalar_001 | Ascend 910B，aiv_scalar_ratio 高 | 40% | i32/i64 比较标量降级规则与修复 | [techniques/data_types](techniques/data_types.md) | [yaml](../../raw/experiences/general_ascend/exp_ascend_int_compare_scalar_001.yaml) |
| exp_ascend_int64_address_001 | 地址计算含 int64 | 50% | i64 地址计算 → i32 | [techniques/data_types](techniques/data_types.md) | [yaml](../../raw/experiences/general_ascend/exp_ascend_int64_address_001.yaml) |

### 内存访问优化

| 经验 ID | 条件 | 提升% | 核心技术 | 详细页面 | 经验文件 |
|---------|------|-------|---------|---------|---------|
| exp_perf_011 | 使用 `tl.make_block_ptr` + `boundary_check` 且 scalar 高 | 50% | block_ptr → 手动指针 + fp32 mask | [techniques/memory_access_optimization](techniques/memory_access_optimization.md) | [yaml](../../raw/experiences/general_perf/exp_perf_011.yaml) |
| exp_perf_015 | 转置视角加载行优先张量 | 30% | 行优先 load + `tl.trans` 寄存器转置 | [techniques/memory_access_optimization](techniques/memory_access_optimization.md) | [yaml](../../raw/experiences/general_perf/exp_perf_015.yaml) |
| exp_perf_016 | barrier 前后重复加载同一数据 | 15% | 跨阶段寄存器复用 | [techniques/memory_access_optimization](techniques/memory_access_optimization.md) | [yaml](../../raw/experiences/general_perf/exp_perf_016.yaml) |
| exp_ascend_blockptr_boundary_scalar_001 | block_ptr boundary_check 导致 scalar 高 | 60% | 手动指针 + ALIGNED 快速路径 | [techniques/memory_access_optimization](techniques/memory_access_optimization.md) | [yaml](../../raw/experiences/general_ascend/exp_ascend_blockptr_boundary_scalar_001.yaml) |

### 调优配置 / 硬件约束

| 经验 ID | 条件 | 提升% | 核心技术 | 详细页面 | 经验文件 |
|---------|------|-------|---------|---------|---------|
| exp_perf_017 | autotune 配置空间窄 | 20% | 扩展 num_warps/num_stages/tile | [techniques/performance_tuning](techniques/performance_tuning.md) | [yaml](../../raw/experiences/general_perf/exp_perf_017.yaml) |
| exp_ascend_compile_hint_001 | 对 block_ptr/load 加 `compile_hint` | 25% | 移除冗余 compile_hint | [techniques/compute_and_branch](techniques/compute_and_branch.md) | [yaml](../../raw/experiences/general_ascend/exp_ascend_compile_hint_001.yaml) |

---

## SSM / Linear Attention / KDA 经验

| 经验 ID | 条件 | 提升% | 核心技术 | 详细页面 | 经验文件 |
|---------|------|-------|---------|---------|---------|
| exp_kda_bwd_wy_scalar_opt_001 | chunk_kda_bwd_wy_dqkg_fused | 233% | i64→i32、i32→fp32、min/cdiv 消除、hint 移除 | [ops/ssm](ops/ssm.md) | [yaml](../../raw/experiences/ssm/exp_kda_bwd_wy_scalar_opt_001.yaml) |
| exp_kda_bwd_wy_structure_opt_001 | chunk_kda_bwd_wy_dqkg_fused | 30% | Phase split、barrier 外提、LICM | [ops/ssm](ops/ssm.md)、[ops/fused_ops](ops/fused_ops.md) | [yaml](../../raw/experiences/ssm/exp_kda_bwd_wy_structure_opt_001.yaml) |
| exp_kda_bwd_intra_scalar_opt_001 | chunk_kda_bwd_kernel_intra | 50% | block_ptr→手动指针、fp32 mask、min→tl.where | [ops/ssm](ops/ssm.md) | [yaml](../../raw/experiences/ssm/exp_kda_bwd_intra_scalar_opt_001.yaml) |
| exp_kda_bwd_intra_memory_opt_001 | chunk_kda_bwd_kernel_intra | 35% | 转置 load 合并化、冗余 load 消除 | [ops/ssm](ops/ssm.md) | [yaml](../../raw/experiences/ssm/exp_kda_bwd_intra_memory_opt_001.yaml) |
| exp_kda_bwd_scalar_loop_to_dot_001 | 含 BC 次标量循环的 SSM 反向 | 40% | 标量循环 → `tl.dot` | [ops/ssm](ops/ssm.md)、[techniques/compute_and_branch](techniques/compute_and_branch.md) | [yaml](../../raw/experiences/ssm/exp_kda_bwd_scalar_loop_to_dot_001.yaml) |

---

## 快速条件匹配

```
M,N≥256, K≥1024, fp16/bf16          → exp_matmul_001 → experiences/matmul/exp_matmul_001.yaml (40%)
M,N≥512（正方形大矩阵）               → exp_matmul_diag_tiling_001 → experiences/matmul/exp_matmul_diag_tiling_001.yaml (25-35%)
seq_len≥1024 的 Attention             → exp_fused_attention_002 → experiences/attention/exp_fused_attention_002.yaml (50%)
GQA / MQA 多头共享 KV                 → exp_flashattn_gqa_001 → experiences/attention/exp_flashattn_gqa_001.yaml (40%)
RMSNorm + RoPE 融合                   → exp_fused_qkv_rmsnorm_rope_001 → experiences/normalization/exp_fused_qkv_rmsnorm_rope_001.yaml (40%)
RMSNorm（n_cols超UB容量）             → exp_fused_rms_norm_001 → experiences/normalization/exp_fused_rms_norm_001.yaml (35%)
GEMM 后接激活函数                      → exp_fused_matmul_rms_rope_001 → experiences/matmul/exp_fused_matmul_rms_rope_001.yaml (35%)
GEMM 后接 RMSNorm + RoPE              → 见 ops/fused_ops.md + experiences/matmul/ 目录
MoE expert dispatch                   → exp_moe_binned_scatter_001 → experiences/gather_scatter/exp_moe_binned_scatter_001.yaml (30%)
归约操作                               → experiences/reduction/ 目录（4条经验）
SwiGLU / GELU / SiLU 激活             → experiences/activation/ 目录（14条经验）
RoPE 位置编码                          → exp_rope_npu_001 → experiences/rope/exp_rope_npu_001.yaml (35%，910b 最高) → 见 ops/rope.md
aiv_scalar_ratio 高 + block_ptr       → exp_perf_011 / exp_ascend_blockptr_boundary_scalar_001 → techniques/memory_access_optimization.md
转置 load + aiv_mte2 高               → exp_perf_015 → techniques/memory_access_optimization.md
KDA/Linear Attention 反向              → exp_kda_bwd_wy_scalar_opt_001 → ops/ssm.md
```

## 经验目录速查

| 类目 | 路径 | 经验数 |
|------|------|--------|
| Attention | [experiences/attention/](../../raw/experiences/attention/) | 47 |
| MatMul / GEMM | [experiences/matmul/](../../raw/experiences/matmul/) | 38 |
| 杂项融合 | [experiences/misc/](../../raw/experiences/misc/) | 40 |
| Normalization | [experiences/normalization/](../../raw/experiences/normalization/) | 33 |
| 激活函数 | [experiences/activation/](../../raw/experiences/activation/) | 14 |
| Gather/Scatter/MoE | [experiences/gather_scatter/](../../raw/experiences/gather_scatter/) | 15 |
| RoPE | [experiences/rope/](../../raw/experiences/rope/) | 10 |
| Softmax | [experiences/softmax/](../../raw/experiences/softmax/) | 7 |
| SSM | [experiences/ssm/](../../raw/experiences/ssm/) | 12 |
| 通用性能 | [experiences/general_perf/](../../raw/experiences/general_perf/) | 16 |
| 通用 Ascend | [experiences/general_ascend/](../../raw/experiences/general_ascend/) | 16 |
| Loss | [experiences/loss/](../../raw/experiences/loss/) | 12 |
| Reduction | [experiences/reduction/](../../raw/experiences/reduction/) | 4 |
| 全量索引 | [experiences/_index.yaml](../../raw/experiences/_index.yaml) | 303 |
