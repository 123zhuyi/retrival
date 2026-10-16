---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.1"
title: 整数比较/加法必须转 float32，避免 Vector 标量降级
description: Ascend 910B 上 int64 的 ADD/CMP/MUL 和 int32 的 LT/GT/LE/GE 比较会触发 scalar fallback；生成 mask 或做整数比较前先 .to(tl.float32)。
tags: [scalar_fallback, int64, int32, float32, comparison, mask]
confidence: verified
status: verified
created_at: "2026-08-10T00:00:00Z"
updated_at: "2026-08-10T00:00:00Z"
---

## CT-13 整数比较/加法必须转 float32，避免 Vector 标量降级

【优化维度】：计算｜Scalar 削减
【泛化层级】：Triton通用（迁移条件：所有在 kernel 内做整数比较/加法的场景；失效边界：输入范围明确在 int32 Vector 支持内且无降级时）

<原则：Ascend Vector 单元对 int64 的 ADD/CMP/MUL 和 int32 的 LT/GT/LE/GE 比较不支持，会降级到标量单元执行。`tl.arange(0, N)` 默认产生 int64，因此 `mask = offs < N` 是最常见的 scalar fallback 来源。统一先 `.to(tl.float32)` 再比较/加法，可让 Vector 单元执行。>

**标量降级类型矩阵**：

| 操作 | int64 | int32 | fp16/bf16/fp32 |
|---|---|---|---|
| ADD | 标量降级 | Vector | Vector |
| CMP (LT/GT/LE/GE) | 标量降级 | 标量降级 | Vector |
| MUL | 标量降级 | Vector | Vector |

**最常见错误：mask 生成**：

```python
# 错误：int64 比较 → scalar fallback
offs = tl.arange(0, BLOCK_N)   # int64
mask = offs < N

# 正确：转 fp32 后比较 → Vector
offs = tl.arange(0, BLOCK_N)
mask = offs.to(tl.float32) < N
```

**2D mask 同样处理**：

```python
o_i = tl.arange(0, BC)
o_k = tl.arange(0, BK)

# 错误
mask = (o_i[:, None] < T) & (o_k[None, :] < K)

# 正确
m_t = o_i[:, None].to(tl.float32) < T.to(tl.float32)
 m_k = o_k[None, :].to(tl.float32) < K.to(tl.float32)
mask = m_t & m_k
```

**block_ptr boundary_check 的隐式 int32 比较**：

```python
# 高标量开销：每个元素都触发 i32 边界比较
p = tl.make_block_ptr(base, (T, K), (H*K, 1),
                      (i_ti, i_k*BK), (BC, BK), (1, 0))
x = tl.load(p, boundary_check=(0, 1))

# 优化：手动指针 + fp32 mask
o_i = tl.arange(0, BC)
o_k = tl.arange(0, BK)
m_ti = (i_ti + o_i).to(tl.float32) < T_local.to(tl.float32)
m_k  = o_k.to(tl.float32) < K.to(tl.float32)
mask = m_ti[:, None] & m_k[None, :]
p = base + (i_ti + o_i[:, None]) * H * K + o_k[None, :]
x = tl.load(p, mask=mask, other=0.0)
```

**诊断阈值**：

| `aiv_scalar_ratio` | 解读 |
|---|---|
| < 5% | 正常 |
| 5–10% | 中等，值得关注 |
| > 10% | 明显标量降级，需修复 |
| > 30% | 严重，必须立即修复 |

**坏实践（反例）**：
```python
offset = row_id * stride_n + col_id  # 若 row_id 为 int64，乘法降级
```
后果：`aiv_scalar_ratio` 飙升，同样数据量慢 32–128x。✅ 正解即索引转 int32 或比较/加法转 fp32。

## 已知实例（P3 挂载）

### Experience YAML 反链
- `exp_ascend_blockptr_boundary_scalar_001` — 硬件约束 （收益 60%）
  - 不要无条件替换所有block_ptr。对大块、确定性访存，block_ptr仍可能提供更好的预取和更简洁的代码。优化重点是小tile、边界检查密集的场景。
- `exp_ascend_int_compare_scalar_001` — 硬件约束 （收益 40%）
  - 即使将tl.arange转为int32，LT/GT/LE/GE仍会标量降级。必须转fp32或消除比较。这是Ascend 910B特有的行为，与GPU不同。
- GDN 线性注意力优化实践（2026-03，内部社区文章）— 硬件约束
  - A2/A3 平台同样存在本约束：`tl.arange` 默认 int64 索引与边界比较生成 mask 时退化为标量计算。该实践将 mask 索引显式转 fp32 后比较作为计算流优化首查项；仅比较操作数需转 fp32，寻址偏移保持整型（int32）。注意索引值域 > 2^24（约 16.7M）时 fp32 不再精确，超长序列全局偏移需分段处理。

<!-- okf:related:start -->

# 相关

- 实践案例: [BlockSparseAttention (block-sparse mask) on Ascend NPU](../../ops/block-sparse-attention.md) — 同属优化知识，主题相关
- 实践案例: [Elementwise add with a tail mask](../../ops/elementwise-add.md) — 同属优化知识，主题相关
- 实践案例: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](../../ops/moe-compute-expert-tokens.md) — 主题相关：共享 int32
- 相关主题: [坐标比较必须先转 float32](../algorithm-numerics/coord-compare-float32.md) — 主题相关：共享 float32
- 相关主题: [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](../memory-sram/exp_perf_011.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: vectorized comparison](../memory-sram/exp_spec_dec_verify_001.md) — 同属优化知识，主题相关
- 相关主题: [UB 打满策略：调大分块吃满 192KB，微超则把 mask/exp 计算外提到调用侧](../memory-sram/ub-fill-and-compute-hoist.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend kernel 调试排查清单](../meta-skills/debugging-checklist.md) — aiv_scalar_ratio 高排查项映射到本卡
- 相关主题: [内存访问三查：mask 防越界、stride 匹配布局、索引上限判断](../meta-skills/exp_debug_001.md) — 同属优化知识，主题相关
- 相关主题: [控制流避坑：禁 return/break/continue，复杂条件分步 mask 组合](../meta-skills/exp_debug_002.md) — 同属优化知识，主题相关
- 相关主题: [if/else 分支 dtype 不一致：改 masked load，bool 张量 host 侧转 int32](../meta-skills/exp_debug_triton_branch_type_mismatch_001.md) — 同属优化知识，主题相关
- 相关主题: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](../portability/exp_ascend_blockptr_boundary_scalar_001.md) — 同属优化知识，主题相关
- 相关主题: [地址计算全程保持 int32，避免 int64 标量降级](../portability/exp_ascend_int64_address_001.md) — 同属优化知识，主题相关
- 相关主题: [标签索引转 int32 比较 + GROUP_SIZE_M 分组调度 + 大 K tile](../portability/exp_entropy_backward_npu_001.md) — 同属优化知识，主题相关
- 相关主题: [反向 Q kernel：CV 均衡 + 双缓冲 + 代数掩码（(mask-1)*1e6）+ 64×128 三角遍历](../portability/exp_sda_bwd_q_001.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend mask 纪律与编译期陷阱（fp32 mask 污染 / stride-0 memref / 禁 % 运算符 / 2D mask store 丢格）](../portability/triton-ascend-mask-codegen-traps.md) — mask dtype 契约交点：int32 mask 纪律与 fp32 比较转换互补
- 相关主题: [Compute Optimization: dim is multiple mask elision](exp_causal_conv1d_prefill_007.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: dim is multiple mask elimination](exp_causal_conv1d_update_005.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: redundant mask elimination by matrix property](exp_solve_tril_001.md) — 同属优化知识，主题相关
- 相关主题: [kernel 内索引必须转 int32，避免 int64 降级](int32-index-cast.md) — 同属优化知识，主题相关
- 相关主题: [per-channel 参数标量 load 路径，避免向量化 int 除法降级](scalar-load-channel-index.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
