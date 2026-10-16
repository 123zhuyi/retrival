---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: autotuning
dimension: 5.1
title: autotune 配置空间须覆盖 num_warps(2/4/8)×num_stages(2/3/4) 与 tile 组合
description: 910b2/910b3 上单一 autotune 配置无法覆盖不同 shape/寄存器压力的最优调度，配置空间应含 num_warps 2/4/8 × num_stages 2/3/4 及不同 tile（BK/BV）；寄存器压力大的 kernel 须剪掉高 warp 配置防 spill，tuned 后额外加速 10-20%
tags: [调优空间设计, general, general_perf]
confidence: provisional
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## autotune 配置空间须覆盖 num_warps(2/4/8)×num_stages(2/3/4) 与 tile 组合

【优化维度】：调优空间设计
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：配置空间越大编译时间越长，开发阶段应用 2-3 个配置的小集合快速迭代、最终版本才全量搜索；可执行剪枝判据：① kernel 含多个 BC×BK 级累加器（寄存器压力大）时剪掉 num_warps=8（spill 风险）；② num_stages 依赖异步拷贝能力，Ascend 上 USE_TMA=False 时 num_stages>2 的流水收益存疑（见 exp_solve_tril_005），可优先试 2/3；③ autotune 的 key 必须包含所有影响 tile 形状的参数（BK/BV/BT 等），漏 key 会导致错误配置复用；经验值 BK=128/BV=64/num_warps=8/num_stages=2 来自特定 fused kernel，仅作起点）

### 原则
- **问题根因**：单一autotune配置（如num_warps=2, num_stages=3）无法覆盖不同shape/register pressure下的最优调度
- **核心方案**：扩展autotune配置空间，包含不同num_warps（2/4/8）和num_stages（2/3/4）组合，以及不同tile大小（BK/BV）
- **经验值**：chunk_kda_bwd_kernel_wy_dqkg_fused在Ascend上最优为BK=128, BV=64, num_warps=8, num_stages=2；更多warp可更好隐藏标量延迟
- **注意**：对寄存器压力大的kernel（多个BC×BK累加器），高warp数可能导致spill，需实测
- **典型收益**：在wy_dqkg_fused上从单一配置到tuned配置贡献约10-20%额外加速

### 代码/骨架出处
- `Triton`:
```python
# 单一配置（空间窄）
@triton.autotune(
    configs=[triton.Config({}, num_warps=2, num_stages=3)],
    key=['BK', 'NC', 'BT'],
)

# 扩展配置空间
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=2, num_stages=4),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=['BK', 'NC', 'BT'],
)
```

**坏实践（反例）**：只挂一个"经验配置"（如 num_warps=2, num_stages=3）就上线 → shape 或寄存器压力变化后调度严重次优，实测与 tuned 配置差 10-20% → ✅ 正解即本条：配置空间覆盖 num_warps 2/4/8 × num_stages 2/3/4 × 关键 tile 档位，并按上列判据剪枝；另一个常见错误是 autotune key 漏掉 tile 参数，导致不同 tile 复用同一配置

**已知实例**：general_perf §raw (`experiences/general_perf/exp_perf_017.yaml`)
- 声称性能收益：20%
- 备注：autotune会增大编译时间，但对重复运行收益显著。建议在最终版本启用完整搜索，在开发阶段可用小配置集快速迭代。

<!-- okf:related:start -->

# 相关

- 相关主题: [Triton API Quick Reference for Ascend Development](../../reference/triton-api-cheatsheet.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: autotune block m](../compute-pipeline/exp_gelu_bwd_001.md) — 同属优化知识，主题相关
- 相关主题: [性能避坑: 避免多维 stride 访问](../compute-pipeline/exp_perf_007.md) — 主题相关：共享 general, general_perf
- 相关主题: [算子拆分优化: 按计算阶段拆分为简单 kernel](../compute-pipeline/exp_perf_010.md) — 主题相关：共享 general_perf
- 相关主题: [标量消除: i32 比较转 fp32 走向量单元](../compute-pipeline/exp_perf_012.md) — 主题相关：共享 general, general_perf
- 相关主题: [标量消除: min() 运行时分支改 tl.where](../compute-pipeline/exp_perf_013.md) — 主题相关：共享 general, general_perf
- 相关主题: [标量消除: tl.cdiv 编译期循环边界](../compute-pipeline/exp_perf_014.md) — 主题相关：共享 general, general_perf
- 相关主题: [for-loop tiling 启用双缓冲/多缓冲流水线](../compute-pipeline/for-loop-tiling-ilp.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: constexpr strides](../memory-sram/exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 general
- 相关主题: [Add算子块大小动态调整优化经验](../memory-sram/exp_elem_add_001_20260526.md) — 主题相关：共享 general
- 相关主题: [带 alpha 缩放的逐元素 add 访存与块大小优化](../memory-sram/exp_elemwise_add_001_20260526.md) — 主题相关：共享 general
- 相关主题: [内存访问优化：2D 张量块指针访存（make_block_ptr）](../memory-sram/exp_perf_002.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存访问优化：非连续张量连续化 + 一维遍历](../memory-sram/exp_perf_003.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存访问优化：stride 参数一律传 tensor.stride() 原生值](../memory-sram/exp_perf_004.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存访问优化：contiguous() 一次性拷贝替代 stride 访问](../memory-sram/exp_perf_006.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存初始化优化：输出张量创建优选 torch.empty_like](../memory-sram/exp_perf_008.md) — 主题相关：共享 general, general_perf
- 相关主题: [冗余操作优化：同形 element-wise 算子免 reshape](../memory-sram/exp_perf_009.md) — 主题相关：共享 general_perf
- 相关主题: [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](../memory-sram/exp_perf_011.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存访问优化：行优先连续加载 + tl.trans 寄存器转置](../memory-sram/exp_perf_015.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存访问优化：跨阶段寄存器复用消除重复 load](../memory-sram/exp_perf_016.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存初始化优化：输出张量创建 API 选择](../memory-sram/exp_tensor_create_001.md) — 主题相关：共享 general
- 相关主题: [Tiling 预算与布局驱动的块大小选择](../memory-sram/tiling-budget-and-layout-driven-selection.md) — 同属优化知识，主题相关
- 相关主题: [if/else 分支 dtype 不一致：改 masked load，bool 张量 host 侧转 int32](../meta-skills/exp_debug_triton_branch_type_mismatch_001.md) — 主题相关：共享 general
- 相关主题: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../meta-skills/quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
- 相关主题: [Ascend 后端禁用语法及替代写法（控制流/lambda/链式布尔/直接索引/负偏移）](../portability/exp_api_001.md) — 主题相关：共享 general
- 相关主题: [复杂嵌套条件下 tl.where 改 if-else 分支](../portability/exp_api_002.md) — 主题相关：共享 general
- 相关主题: [静态上限循环：while 改 for range（上限须为 tl.constexpr）](../portability/exp_api_003.md) — 主题相关：共享 general
- 相关主题: [动态上限循环：for + if + MAX_ITERS 编译期上界替代 while](../portability/exp_api_004.md) — 主题相关：共享 general
- 相关主题: [禁止用 tl.where 动态计算内存偏移，改 if-else 静态分支](../portability/exp_ascend_where_001.md) — 主题相关：共享 general
- 相关主题: [波前调度 MatMul：对角遍历 + dot_pad_only_k + autotune tile 长宽比](../portability/exp_matmul_ascend_wavefront_001.md) — 同属优化知识，主题相关
- 相关主题: [禁用 Python 切片语法，改用 get_element/extract_slice/insert_slice](../portability/exp_slice_001.md) — 主题相关：共享 general
- 相关主题: [禁止对 tl.arange 索引表达式调用 get_element](../portability/exp_slice_002.md) — 主题相关：共享 general
- 相关主题: [迁移 kernel 须按 CANN 版本审计 launch 参数（8.5.0 删 no-limit/sync_solver/num_stages）](../portability/exp_solve_tril_005.md) — 同属优化知识，主题相关
- 相关主题: [累加/原地修改算子 autotune 必须每次重置输出 buffer](autotune-debug.md) — 同属优化知识，主题相关
- 相关主题: [autotune 必须探索 num_warps 与 num_stages](autotune-num-warps-stages.md) — 同属优化知识，主题相关
- 相关主题: [块大小选型：2 的幂次 128/256/512 起调，blockSize×dtypeBytes 对齐 32B 且不超 UB](exp_perf_001.md) — 主题相关：共享 general, general_perf, 调优空间设计
- 相关主题: [Tiling 候选须过编译成本闸：超大分块档位编译爆炸直接剔除，host 公式分档可替代 autotune](tiling-candidate-compile-cost-gate.md) — 调优空间设计主题：tile/编译成本同为候选预过滤变量
- 相关主题: [µs 级 kernel 禁用 autotune；num_stages/num_warps 在 Triton-Ascend 不生效](triton-ascend-microkernel-autotune-noise.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
