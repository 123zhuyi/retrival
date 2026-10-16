---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: autotuning
dimension: "5.1"
title: autotune 必须探索 num_warps 与 num_stages
description: Ascend 上单一 autotune 配置无法覆盖所有 shape/register pressure，必须探索 num_warps={2,4,8} 和 num_stages={2,3,4} 以找到最优流水线并行度。
tags: [autotune, num_warps, num_stages, register_pressure, pipeline, scalar_hide]
confidence: verified
status: verified
created_at: "2026-08-10T00:00:00Z"
updated_at: "2026-08-10T00:00:00Z"
---

## OPT-26 autotune 必须探索 num_warps 与 num_stages

【优化维度】：调优｜autotune
【泛化层级】：条件性（迁移条件：所有性能关键 kernel；失效边界：极小 kernel 或硬件固定参数无需调优）

<原则：Ascend 上 `num_warps` 和 `num_stages` 对性能影响显著。标量占比高的 kernel 需要更多 warp 隐藏延迟；寄存器压力大的 kernel 高 warp 会导致 spill；内存瓶颈的 kernel 通常 `num_stages=2/3` 最优。单一配置无法覆盖所有 shape 和 register pressure，必须在 autotune 搜索空间中包含这两个维度。>

**推荐搜索空间**：

```python
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

**参数含义与经验值**：

| 参数 | 作用 | 经验选择 |
|---|---|---|
| `num_warps` | warp 数，影响标量延迟隐藏和寄存器压力 | 标量高 → 8；寄存器压力大 → 2/4 |
| `num_stages` | 软件流水级数，影响 load/compute 重叠 | 内存瓶颈 → 2/3；长延迟 load → 4 |

**调优方向**：

- `aiv_scalar_ratio` 高、大量标量指令 → 优先尝试 `num_warps=8`。
- `PipeUtilization` 低、MTE2 stall 高 → 优先尝试 `num_stages=3/4`。
- 编译报错 spill 或 UB 超限 → 降低 `num_warps` 或 `num_stages`。

**坏实践（反例）**：
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}),
    ],
    key=['M', 'N', 'K'],
)
```
后果：错过最优 warp/stages 组合，性能可能差 20–50%。✅ 正解即至少探索 2/4/8 warps × 2/3 stages。

**KDA 反向调优实例**：

| 轮次 | 优化 | Task Duration | 关键发现 |
|---|---|---|---|
| R1 | BK/BV autotune + barrier 外提 | 127,168 μs | 更大 tile + 更少 barrier |
| R6 | num_warps=8 | 50,867 μs | 更多 warp 隐藏标量延迟 |
| R7 | Config tuning | 50,700 μs | BK=128, BV=64, num_warps=8 |

## 已知实例（P3 挂载）

### Experience YAML 反链
- `exp_perf_017` — 调优配置 （收益 20%）
  - autotune会增大编译时间，但对重复运行收益显著。建议在最终版本启用完整搜索，在开发阶段可用小配置集快速迭代。

<!-- okf:related:start -->

# 相关

- 相关主题: [Triton API Quick Reference for Ascend Development](../../reference/triton-api-cheatsheet.md) — 主题相关：共享 autotune
- 相关主题: [dl.async_task 实现 Cube/Vector 异构并行](../compute-pipeline/async-task-cube-vector.md) — 主题相关：共享 pipeline
- 相关主题: [care_padding=False 消除 MTE2→Vector 同步屏障](../compute-pipeline/care-padding-false-ilp.md) — 主题相关：共享 pipeline
- 相关主题: [固定 CUBE_CORE_NUM 启动 + 核内循环（Cube 类算子）](../compute-pipeline/cube-core-num-fixed.md) — 同属优化知识，主题相关
- 相关主题: [动态读取 Vector Core 数量，禁止硬编码](../compute-pipeline/dynamic-vector-core-num.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: autotune block m](../compute-pipeline/exp_gelu_bwd_001.md) — 同属优化知识，主题相关
- 相关主题: [for-loop tiling 启用双缓冲/多缓冲流水线](../compute-pipeline/for-loop-tiling-ilp.md) — 主题相关：共享 num_stages, pipeline
- 相关主题: [grid 总数不超过核数](../compute-pipeline/grid-cap-to-core-num.md) — 同属优化知识，主题相关
- 相关主题: [grid_size 必须与 num_cores 入参严格相等](../compute-pipeline/grid-strict-match-num-cores.md) — 同属优化知识，主题相关
- 相关主题: [Tiling 预算与布局驱动的块大小选择](../memory-sram/tiling-budget-and-layout-driven-selection.md) — 主题相关：共享 autotune
- 相关主题: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../meta-skills/quant-matmul-autotune-strategy.md) — 主题相关：共享 autotune
- 相关主题: [波前调度 MatMul：对角遍历 + dot_pad_only_k + autotune tile 长宽比](../portability/exp_matmul_ascend_wavefront_001.md) — 同属优化知识，主题相关
- 相关主题: [迁移 kernel 须按 CANN 版本审计 launch 参数（8.5.0 删 no-limit/sync_solver/num_stages）](../portability/exp_solve_tril_005.md) — 同属优化知识，主题相关
- 相关主题: [累加/原地修改算子 autotune 必须每次重置输出 buffer](autotune-debug.md) — 主题相关：共享 autotune
- 相关主题: [autotune 配置空间须覆盖 num_warps(2/4/8)×num_stages(2/3/4) 与 tile 组合](exp_perf_017.md) — 同属优化知识，主题相关
- 相关主题: [Tiling 候选须过编译成本闸：超大分块档位编译爆炸直接剔除，host 公式分档可替代 autotune](tiling-candidate-compile-cost-gate.md) — autotune 配置空间主题：本卡讲空间被编译成本封死/不适用时的替代
- 相关主题: [µs 级 kernel 禁用 autotune；num_stages/num_warps 在 Triton-Ascend 不生效](triton-ascend-microkernel-autotune-noise.md) — 主题相关：共享 autotune, num_stages, num_warps

<!-- okf:related:end -->
