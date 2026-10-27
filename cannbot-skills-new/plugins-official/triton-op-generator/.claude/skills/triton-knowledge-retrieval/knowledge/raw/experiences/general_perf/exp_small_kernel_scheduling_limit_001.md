---
id: exp_small_kernel_scheduling_limit_001
category: general_perf
operator_type: fused_normalization_rope
dtype: bfloat16
hardware: ascend910b2
source: D
created: 2026-05-06
---

# 小算子调度下限经验：HEAD_DIM=128 轻量级 kernel 优化上限

## 现象
- grid=576（bsz=12 × q_heads=48），每 program 处理 HEAD_DIM=128 个 bf16 元素
- Phase 3 基线已达 3.306x speedup，Phase 4 三轮优化（constexpr、int32 offset、load order、V prefetch）均未超过 1.05x
- 预判瓶颈为 VecUtil_low，实际为 memory_bound_small_kernel（调度开销主导）

## 根因分析
- 每 program 工作量约 1KB（128 × 2B bf16），极小
- Ascend 910B2 单 core 调度开销与计算时间接近，优化空间被调度噪声覆盖
- constexpr 化和 int32 offset 对此规模效果不显著（编译器已在小 kernel 上做了常量折叠）
- 瓶颈不是 VecUtil 或 MTE2，而是 kernel launch overhead + 小 tile 调度

## 适用条件
- HEAD_DIM ≤ 128，grid ≤ 1024，每 program 数据量 ≤ 2KB
- 基线 speedup 已 ≥ 3x 的融合算子

## 建议
- 对此类小算子，Phase 4 优化收益极为有限，不必强行尝试 3 轮
- 可考虑提升 bsz 或 seq_len 以增大 grid，从而改善调度效率
- 更大 HEAD_DIM（如 256）或更大 bsz 场景下，上述优化策略仍有效
