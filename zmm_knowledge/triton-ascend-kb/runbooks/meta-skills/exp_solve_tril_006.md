---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: meta-skills
dimension: 6.1
title: 负优化排除清单：微小矩阵禁 tl.dot、constexpr 展开优先、preload 按 grid 规模决策
description: 910b2/910b3/91093/91095 + fp16/bf16/fp32 下三角求解类小矩阵（16/32/64）优化决策：1×16×16×16 用 VEC 广播乘+sum 比 tl.dot 快 14-20x；99% tile 走同路径时保留 constexpr 双分支；preload 按 grid 规模（B≥4,T≥1024 启用）；off-diagonal 按需 HBM 加载防寄存器 spill；逐行递推 O(BT²) 依赖是算法根本限制
tags: [瓶颈诊断树, linear-algebra-operations, solve_tril]
confidence: provisional
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 负优化排除清单：微小矩阵禁 tl.dot、constexpr 展开优先、preload 按 grid 规模决策

【优化维度】：瓶颈诊断树
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：14-20x 退化数据针对 1×16×16×16 级别的微小矩阵，矩阵尺寸放大后 CUBE 调度开销被摊薄、tl.dot 重新占优（具体拐点未测定）；VEC 广播乘+sum 的替代仅对 M=1 或 N=1 或 K≤16 成立；逐行递推 O(BT²) 依赖是三角求解类算法的固有结构限制，换算法（如分块求逆）才可能突破；触发条件：对小尺寸矩阵 kernel 做优化选型时先查本卡排除已知无效方向）

### 原则
- **tl dot vs tl sum broadcast decision**：对 1×16 × 16×16 的微小矩阵乘法，使用 `tl.dot`（CUBE 单元）比 `tl.sum(b_a[:, None] * b_A, axis=0)`（VEC 单元广播乘+归约）慢 14-20x。CUBE 单元的调度和启动开销在微小尺寸下远大于 VEC 直接计算。**决策原则**：微小矩阵乘法（M=1 或 N=1 或 inputB≤16）优先用 VEC 广播乘 + sum 归约，不要强行用 tl.dot。
- **constexpr unroll vs runtime loop decision**：去掉 if/else 统一用运行时循环（V3）会导致失去 constexpr 展开优化，性能回退到 V1 水平。constexpr 展开使 extract_slice 的 offset 参数为编译期常量，生成静态偏移指令（零 ALU）。**决策原则**：当 99% 的 tile 走同一代码路径时，保留 constexpr 双分支比代码简洁性更重要。
- **register vs hbm tradeoff**：preload 整个 16×16 块会增加寄存器占用（256 elements fp32 = 1KB）。对小 grid 场景（B=1, T=512），额外的寄存器操作开销可能与省下的 global load 抵消（实测 1.01x）。**决策原则**：大 grid 场景（B≥4, T≥1024）无条件启用 preload；小 grid 场景可考虑保留 HBM load（代码略简），但统一使用 extract_slice 的代码一致性通常优先。
- **off diagonal preload excluded**：提前加载 off-diagonal block 会增加寄存器压力（64×64 kernel 已有 10 个 16×16 块），可能导致寄存器 spill。**决策原则**：寄存器中仅保留对角块（基础计算单元），off-diagonal block 按需从 HBM 加载——逐层计算模式下每层只用到特定的 off-diagonal 块，全部预加载造成浪费。
- **algorithmic bottleneck**：16×16 kernel 的逐行递推存在 O(BT²) 次迭代内的数据依赖——第 i 行严格依赖前 i-1 行的计算结果，无法并行化或流水线化。**这是算法的根本限制**，非代码优化可克服。32×32 和 64×64 merge kernel 的计算瓶颈在 16×16 tl.dot 矩阵乘法（CUBE 单元已饱和）。
- **exhaustive audit result**：对照经验库全部 20+ 条经验审计后确认：当前算子已达到局部最优。已应用：block_ptr 访存、fp32 中间计算、contiguous 输入、empty_like 分配、冗余消除、extract_slice 寄存器读取、constexpr 展开。不适用或无效：块大小优化（BT 是算子语义决定）、TMA 异步拷贝（CANN 8.5.0 不支持）、tl.dot 替代 VEC 广播乘（14-20x 退化）。

### 代码/骨架出处
- `dot_vs_sum`:
```python
# 反模式（14-20x 退化）：微小矩阵用 tl.dot
b_a = tl.reshape(b_a, (1, 16))
b_a = tl.dot(b_a, b_A)  # CUBE 调度开销 >> 计算量

# 正模式（高效）：VEC 广播乘 + 归约
b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)  # 1×16 × 16×16 → VEC
```

**坏实践（反例）**：此经验记录优化尝试中被排除的方向及其根因，避免未来重复尝试无效优化

**已知实例**：solve_tril §raw (`experiences/solve_tril/exp_solve_tril_006.yaml`)
- 声称性能收益：N/A（排除无效优化方向，避免负优化）
- 备注：此经验记录优化尝试中被排除的方向及其根因，避免未来重复尝试无效优化。所有尝试均在 solve_tril_analyze.md §9.6 中有详细分析。

<!-- okf:related:start -->

# 相关

- 开发指南: [Ascend NPU 特有语法与模式](../../reference/ascend-patterns.md) — 同属优化知识，主题相关
- 开发指南: [solve_tril 算子优化（三角矩阵求逆）](../../reference/op-class-solve_tril.md) — 主题相关：共享 solve_tril
- 相关主题: [GEMM 操作数保持 low_dtype，fp32 累加](../algorithm-numerics/gemm-low-dtype-fp32-accum.md) — 同属优化知识，主题相关
- 相关主题: [用 tl.constexpr 编译期消除运行时分支](../compute-pipeline/constexpr-branch-elimination.md) — 同属优化知识，主题相关
- 相关主题: [依赖无关输入预加载（load 提前于等待边）与带宽仲裁意识](../compute-pipeline/cv-preload-and-bandwidth-arbitration.md) — 同属优化知识，主题相关
- 相关主题: [计算优化: 对角块标量循环改 tl.dot](../compute-pipeline/exp_kda_bwd_scalar_loop_to_dot_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: redundant mask elimination by matrix property](../compute-pipeline/exp_solve_tril_001.md) — 主题相关：共享 linear-algebra-operations, solve_tril
- 相关主题: [Compute Optimization: constexpr dual path full vs boundary](../compute-pipeline/exp_solve_tril_003.md) — 主题相关：共享 linear-algebra-operations, solve_tril
- 相关主题: [Scheduling Optimization: three tier hierarchical dispatch](../compute-pipeline/exp_solve_tril_004.md) — 主题相关：共享 linear-algebra-operations, solve_tril
- 相关主题: [标量循环改写为 tl.dot，提升 Cube 利用率](../compute-pipeline/scalar-loop-to-dot.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: constexpr strides](../memory-sram/exp_causal_conv1d_varlen_opt_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: preload extract slice eliminate hbm read](../memory-sram/exp_solve_tril_002.md) — 主题相关：共享 linear-algebra-operations, solve_tril
- 相关主题: [BLOCK 向上取 2 的幂并声明为 constexpr](../memory-sram/pow2-block-constexpr.md) — 同属优化知识，主题相关
- 相关主题: [preload + extract_slice 消除循环内同一数据重复 HBM 读取](../memory-sram/preload-extract-slice-eliminate-reread.md) — 同属优化知识，主题相关
- 相关主题: [关键长度参数必须 tl.constexpr 静态化触发启动级特化](../portability/constexpr-launch-specialization.md) — 同属优化知识，主题相关
- 相关主题: [静态上限循环：while 改 for range（上限须为 tl.constexpr）](../portability/exp_api_003.md) — 同属优化知识，主题相关
- 相关主题: [tl.constexpr 仅限内核参数声明，Host 侧禁用](../portability/exp_constexpr_001.md) — 同属优化知识，主题相关
- 相关主题: [迁移 kernel 须按 CANN 版本审计 launch 参数（8.5.0 删 no-limit/sync_solver/num_stages）](../portability/exp_solve_tril_005.md) — 主题相关：共享 linear-algebra-operations, solve_tril
- 相关主题: [tl.dot vs torch.matmul 精度差异诊断](dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
