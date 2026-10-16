---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.1"
title: 标量循环改写为 tl.dot，提升 Cube 利用率
description: 当 kernel 中出现固定次数的逐行标量循环且计算可表达为矩阵乘时，改为 tl.dot 可让 Cube 单元执行，避免 scalar 循环开销。
tags: [scalar_loop, tl_dot, cube_utilization, matrix_multiply, kda, ssm]
confidence: verified
status: verified
created_at: "2026-08-10T00:00:00Z"
updated_at: "2026-08-10T00:00:00Z"
---

## OPT-16 标量循环改写为 tl.dot，提升 Cube 利用率

【优化维度】：计算｜Scalar 削减
【泛化层级】：条件性（迁移条件：固定次数小循环，循环体内是标量×向量 broadcast 乘；失效边界：循环次数大、数值不可合并为矩阵乘、或内存访问模式不适合）

<原则：Ascend Cube 单元擅长矩阵乘。当 kernel 中出现 `for j in range(BC)` 的逐行标量循环，且循环体内是 `b_dAqk[:, None] * b_kj[None, :]` 形式的 broadcast 乘法时，应把循环改写为 `tl.dot(b_dA[BC,BC], b_k[BC,BK])`。这样 Cube 单元一次性完成计算，避免 scalar 循环的逐元素调度和地址计算开销。>

**反模式：BC 次标量循环**：

```python
for j in range(BC):
    b_dAqk = tl.load(dAqk_local + o_dA + j, mask=m_dA, other=0)
    b_kj   = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
    m_i = o_i[:, None] >= j
    b_dq2 += tl.where(m_i, b_dAqk[:, None] * b_kj[None, :] * b_gqk, 0.)
    p_kj += H*K
```

**正模式：2 次 tl.dot**：

```python
b_dAqk_diag = tl.load(p_dAqk_diag, mask=dA_diag_mask, other=0.0)
b_k_exp = b_k * b_exp_neg_g
b_dq2 += tl.dot(b_dAqk_diag, b_k_exp) * b_exp_g
```

**适用条件**：

| 条件 | 说明 |
|---|---|
| 循环次数固定且小 | 如 BC=16，编译期可确定 |
| 计算可表达为矩阵乘 | `b_dA[BC,BC] @ b_k[BC,BK]` |
| 数值稳定性允许 | gate 归一化点保持一致 |
| 内存访问模式匹配 | 可把所需数据一次性加载为矩阵块 |

**收益**：16 次标量循环 → 2 次 `tl.dot`，AIC MAC ratio 从 2.1% 量级显著提升。

**坏实践（反例）**：微小矩阵（M=1 或 N=1 或 K≤16）强行用 `tl.dot`，CUBE 调度开销反而大于计算收益。此时应使用 VEC 广播乘 + `tl.sum` 归约。详见 [compute_and_branch.md §4.3](https://source) 或对应 runbook。

## 已知实例（P3 挂载）

### Experience YAML 反链
- `exp_kda_bwd_wy_scalar_opt_001` — 标量消除 （收益 233%）
  - 此bundle针对wy_dqkg_fused反向kernel。关键原则是保留tl.dot架构（不逐元素展开），定向消除导致标量降级的i64/i32操作。逐元素展开虽能降低scalar ratio，但会丧失Cube加速，性能严重回退。
- `exp_kda_bwd_intra_scalar_opt_001` — 标量消除 （收益 50%）
  - intra kernel与wy_dqkg_fused的差异：intra已经正确使用i32（无i64问题），但标量占比仍极高，说明block_ptr boundary_check和i32比较是主要根因。优化时需特别注意转置dAqk/dAkk加载的mask布局。
- `exp_kda_bwd_scalar_loop_to_dot_001` — 计算优化 （收益 40%）
  - 此优化将标量×向量broadcast改为矩阵乘，是提升Cube利用率的关键。但需注意gate归一化点的数值稳定性，建议与SAFE_GATE路径对比精度。

<!-- okf:related:start -->

# 相关

- 实践案例: [ChunkGatedDeltaRuleFwdH（GDN/KDA chunk 间状态递推前向）Ascend C → Triton 语义转化卡](../../ops/chunk-gated-delta-rule-fwd-h.md) — kda/ssm 标量循环改 tl.dot，对应 FwdH 的 kg^T 矩阵乘表达
- 相关主题: [GEMM 操作数保持 low_dtype，fp32 累加](../algorithm-numerics/gemm-low-dtype-fp32-accum.md) — 同属优化知识，主题相关
- 相关主题: [tl.dot vs torch.matmul 精度差异诊断](../meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [负优化排除清单：微小矩阵禁 tl.dot、constexpr 展开优先、preload 按 grid 规模决策](../meta-skills/exp_solve_tril_006.md) — 同属优化知识，主题相关
- 相关主题: [计算优化: 对角块标量循环改 tl.dot](exp_kda_bwd_scalar_loop_to_dot_001.md) — 主题相关：共享 ssm

<!-- okf:related:end -->
