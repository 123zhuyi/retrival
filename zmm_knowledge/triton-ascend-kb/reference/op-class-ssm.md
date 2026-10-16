---
schema_version: okf.v1
kind: guide
type: programming_guide
source_family: reference
title: SSM / 状态空间模型算子优化
description: SSM / 状态空间模型算子优化
tags: [ssm, mamba, linear_attention, delta_rule, chunk_scan, recurrent, 910b2, 910b3, operator_class]
confidence: inferred
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/pages/ops/ssm.md
created_at: '2026-08-10T02:56:36Z'
updated_at: '2026-08-10T02:56:36Z'
---

sources:
  - url: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/pages/ops/ssm.md
    role: primary

> 迁移备注：本卡由 `cannbot-skills_5171/ops/triton-knowledge-retrieval/knowledge/pages/ops/ssm.md` 迁移而来，原始为算子类别设计页，未做内容改写，仅转 OKF frontmatter。

# SSM / 状态空间模型算子优化

## 概述

状态空间模型（SSM）是近年快速发展的序列模型架构，代表性工作包括 Mamba、Hyena、RetNet、GLA 等。  
与 Transformer 不同，SSM 以**线性递推**方式更新隐状态，复杂度 O(T) 而非 O(T²)。

Triton-Ascend 中 SSM 的核心挑战：
1. **递推依赖**：时间步之间有数据依赖，不能简单并行化
2. **Chunkwise 分解**：将序列分块，块内并行（GEMM），块间串行（状态转移）
3. **State 寄存器复用**：隐状态 `h` 需长时间驻留寄存器，避免写回 HBM

## 检测标准

满足以下条件归类为 SSM：
- 计算模式：线性递推 `h_t = A·h_{t-1} + B·x_t`，`y_t = C·h_t`
- 或 Chunkwise 线性注意力：`O = (Q·K^T ⊙ M)·V`（因果掩码 M）
- 或 Delta Rule 更新：`h_new = h - k^T·h·v^T + k^T·v`

## 适用条件

| 条件 | 说明 |
|------|------|
| 数据类型 | fp16、bf16；gating 部分通常 fp32 中间计算 |
| 序列长度 T | ≥128，chunkwise 时 chunk size BT≥16 |
| 状态维度 K/V | ≥64 |
| 模型架构 | Mamba/Mamba2、GLA、RetNet、Hyena |

## 经验路由

| 场景 | 经验ID | 关键技术 | 经验文件 |
|------|--------|----------|----------|
| Chunkwise 线性注意力前向 | exp_chunk_fwd_001 | block_ptr tiling + gating masking，Q/K/V/H 2D块加载 | yaml |
| Chunk CumSum（前缀和） | exp_chunk_cumsum_001 | 块内累积和，用于 gating 因子计算 | yaml |
| Chunk Gated Delta | exp_chunk_gated_delta_001 | 带门控的 delta rule，sigmoid gating + 状态更新 | yaml |
| Chunk KKT 计算 | exp_chunk_kkt_001 | K·K^T 内积矩阵，用于状态压缩 | yaml |
| Delta Rule 前向（标准） | exp_fused_delta_rule_001 | sigmoid gating + softplus + delta rule 三合一融合，状态驻寄存器 | yaml |
| Delta Rule 前向（NPU 专用） | exp_fused_delta_rule_npu_001 | Ascend 910b2/b3 路径优化版 | yaml |
| SSM Gated Delta | exp_ssm_gated_delta_001 | SSM + gated delta rule 融合 | yaml |

### 反向传播优化（KDA / Linear Attention）

| 场景 | 经验ID | 提升 | 关键技术 |
|------|--------|------|----------|
| wy_dqkg_fused 标量消除 | exp_kda_bwd_wy_scalar_opt_001 | 3.33x | i64→i32、i32→fp32 比较、tl.cdiv/min 消除、compile_hint 移除 |
| wy_dqkg_fused 结构优化 | exp_kda_bwd_wy_structure_opt_001 | 30% | Phase split、barrier 外提、LICM、K%BK 快速路径 |
| intra 标量消除 | exp_kda_bwd_intra_scalar_opt_001 | 50% | block_ptr→手动指针、fp32 mask、min→tl.where |
| intra 内存优化 | exp_kda_bwd_intra_memory_opt_001 | 35% | 转置 load 合并化、跨阶段冗余 load 消除 |
| 标量循环 → 矩阵乘 | exp_kda_bwd_scalar_loop_to_dot_001 | 40% | 非 SAFE_GATE 对角块循环改为 tl.dot |

## 关键技术

### Chunkwise 并行化

将长序列 T 分为 `T/BT` 个 chunk，每个 chunk 内部可做 GEMM 并行：

```
对每个 chunk c:
  intra_chunk = Q[c] @ K[c]^T ⊙ causal_mask  (GEMM，块内并行)
  inter_chunk = Q[c] @ H[c-1]                  (状态转移，串行)
  O[c] = (intra_chunk + inter_chunk) @ V[c]
  H[c] = update_state(H[c-1], K[c], V[c])
```

关键约束：BT（chunk size）通常选 16/32/64，过大超出 UB；状态 H 维度 K×V 需适配 UB。

参考：exp_chunk_fwd_001

### 隐状态寄存器驻留

Delta Rule 更新：`h_new = h - (k^T @ h) * v^T + k * v`  
将 `h`（维度 K×V）存于寄存器，避免每步写回 HBM：

```python
@triton.jit
def delta_rule_kernel(..., BK: tl.constexpr, BV: tl.constexpr):
    # h 在寄存器内迭代更新
    b_h = tl.zeros([BK, BV], dtype=tl.float32)  # 隐状态驻寄存器
    for i in tl.static_range(T):
        b_k = tl.load(k_ptr + i * K)
        b_v = tl.load(v_ptr + i * V)
        b_gate = tl.sigmoid(tl.load(g_ptr + i))
        # delta rule 更新（无 HBM 写回）
        b_h = b_gate * b_h + tl.outer(b_k, b_v)
        tl.store(o_ptr + i * V, b_h @ b_q)
```

参考：exp_fused_delta_rule_001

### Gating 融合

sigmoid gating、softplus 激活与主体计算融合到同一 kernel，消除独立激活 kernel 的 HBM 开销。  
参考：exp_chunk_gated_delta_001

---

## KDA / Linear Attention 反向优化

`chunk_kda_bwd_kernel_wy_dqkg_fused` 和 `chunk_kda_bwd_kernel_intra` 是 Linear Attention 类模型的反向核心。在 Ascend 910B 上的优化重点与标准 SSM 前向不同：

### 1. 标量消除（最高优先级）

反向 kernel 通常有大量小 tile load/store 和 mask 计算，容易触发 Ascend 标量降级：

```python
# block_ptr + boundary_check → 手动指针 + fp32 mask
o_i = tl.arange(0, BC)
o_k = tl.arange(0, BK)
m_ti = (i_ti + o_i).to(tl.float32) < T_local.to(tl.float32)
m_k  = o_k.to(tl.float32) < K.to(tl.float32)
mask_2d = m_ti[:, None] & m_k[None, :]
```

### 2. 转置 load 合并化

`dAqk` / `dAkk` 在 Phase 2 中以转置视角加载，直接 block_ptr 会导致跨大步访存。改为行优先加载 + `tl.trans`：

```python
row_idx = i_t*BT + i_j*BC + o_i[:, None]
col_idx = i_i*BC + o_i[None, :]
b_dAqk_row = tl.load(dAqk_local + row_idx*H*BT + col_idx, mask=mask, other=0.0)
b_dAqk = tl.trans(b_dAqk_row)
```

### 3. 标量循环 → 矩阵乘

非 `SAFE_GATE` 路径的对角块计算使用 `for j in range(BC)` 逐行循环，改为 `tl.dot`：

```python
b_dAqk_diag = tl.load(p_dAqk_diag, mask=dA_diag_mask, other=0.0)
b_dq2 += tl.dot(b_dAqk_diag, b_k_exp) * b_exp_g
```

### 4. Kernel 结构优化

- **Phase split**：将 `i_k==0` 分支（dA/dv2/db）外提为独立 V-only 循环
- **barrier 外提**：从 `i_v` 循环内移到循环后
- **LICM**：预计算 `BT_arange[:, None] * V`、`H*K`、`H*V` 等不变量

### 5. 调优配置

```python
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=['BK', 'NC', 'BT'],
)
```

实测 `wy_dqkg_fused` 最优：`BK=128, BV=64, num_warps=8, num_stages=2`。

### 优化效果

| Kernel | 优化前 | 优化后 | 加速比 |
|--------|--------|--------|--------|
| chunk_kda_bwd_kernel_wy_dqkg_fused | ~169,000 μs | ~50,700 μs | 3.33x |
| chunk_kda_bwd_kernel_intra | AIV scalar 41.4% / MTE2 37.9% | 预期 scalar <20% / MTE2 <25% | — |

## 常见陷阱

| 陷阱 | 说明 | 解决方案 |
|------|------|----------|
| 状态过大超 UB | K×V 状态矩阵超出 192KB | 分块 BK/BV，多次 tile 更新状态 |
| 递推串行化 | 跨 chunk 状态转移无法并行 | Chunkwise 分解，块内 GEMM 并行 |
| gating 精度 | sigmoid/softplus 在 fp16 数值不稳定 | 中间计算升 fp32 |
| block_ptr 对齐 | K/V 维度非 512B 倍数 | 使用 padding 或调整 BK/BV |

## 相关页面

- [ops/attention.md](op-class-attention.md) — Chunkwise 线性注意力与 Flash Attention 结构类似
- [ops/matmul.md](op-class-matmul.md) — 块内 GEMM 使用矩阵乘优化
- [ops/fused_ops.md](op-class-fused_ops.md) — 多阶段融合与 Phase split 模式
- techniques/persistent_kernels.md — 状态驻留与 persistent kernel
- techniques/tiling.md — BT/BK/BV tile 大小选择
- techniques/data_types.md — i32→fp32 比较避免 scalar fallback
- techniques/memory_access_optimization.md — 转置 load 合并化、跨阶段冗余 load 消除
- techniques/compute_and_branch.md — min()/tl.cdiv 消除、标量循环→tl.dot
- techniques/parallelism_and_loop.md — 外层循环 grid 并行化
- techniques/performance_tuning.md — autotune 配置空间扩展

## 已知实例（P3 挂载）

### Extracted Kernel 反链
- `151-fused_recurrent_gated_delta_rule_fwd_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/151-fused_recurrent_gated_delta_rule_fwd_kernel.py`）
- `158-chunk_scaled_dot_kkt_fwd_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/158-chunk_scaled_dot_kkt_fwd_kernel.py`）
- `159-chunk_fwd_kernel_o.py` — extracted kernel（source: `raw_experiences/extracted_kernels/159-chunk_fwd_kernel_o.py`）
- `160-chunk_gated_delta_rule_fwd_kernel_h_blockdim64.py` — extracted kernel（source: `raw_experiences/extracted_kernels/160-chunk_gated_delta_rule_fwd_kernel_h_blockdim64.py`）

<!-- okf:related:start -->

# 相关

- 实现示例: [ChunkGatedDeltaRuleFwdH（GDN/KDA chunk 间状态递推前向）Ascend C → Triton 语义转化卡](../ops/chunk-gated-delta-rule-fwd-h.md) — 同属 SSM/线性注意力递推算子类：FwdH 即 chunk 间状态递推
- 实现示例: [GDN 线性注意力三阶段优化案例（FLA 移植 → Triton 体系化 → AscendC 重写）](../ops/gdn-chunked-linear-attention.md) — 主题相关：共享 linear_attention
- 实现示例: [Gdn2Fwd（varlen Gated DeltaNet 2 前向）双骨架 host dispatch 实测案例：serial persistent + chunked WY](../ops/gdn2-fwd-chunked-wy-dispatch.md) — GDN/线性注意力属 SSM 算子类
- 实现示例: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](../ops/recurrent-gated-delta-rule.md) — 主题相关：共享 linear_attention, recurrent, ssm
- 讲解接口: [layout (B,T,H,D)→(B,H,T,D) 消除跨 Head 离散访存](../runbooks/memory-sram/layout-bthd-to-bhtd.md) — 主题相关：共享 linear_attention
- 相关主题: [Fused Attention / Flash Attention 高性能算子](op-class-attention.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [卷积算子优化（Causal Conv1D）](op-class-convolution.md) — 主题相关：共享 910b2, 910b3, mamba, operator_class, ssm
- 相关主题: [Elementwise 算子优化](op-class-elementwise.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Embedding / 索引拷贝算子优化](op-class-embedding.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [融合算子综合页](op-class-fused_ops.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Gather/Scatter 算子优化](op-class-gather_scatter.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Loss 函数算子优化](op-class-loss.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [MatMul / GEMM 高性能算子](op-class-matmul.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [归一化算子优化](op-class-normalization.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [归约算子优化](op-class-reduction.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [RoPE（旋转位置编码）算子优化](op-class-rope.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Softmax / LogSoftmax 算子优化](op-class-softmax.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [solve_tril 算子优化（三角矩阵求逆）](op-class-solve_tril.md) — 主题相关：共享 910b2, 910b3, operator_class

<!-- okf:related:end -->
