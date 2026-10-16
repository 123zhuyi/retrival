---
kind: index
type: section_index
title: runbooks/compute-pipeline/（占位）
description: 分类2 计算与流水优化点库（OPT-*/AP-*/CT-*，跨算子单一共享、增量合并）。
---

# 分类2 计算与流水

覆盖二级维度：2.1 单元利用/Scalar 削减 · 2.2 流水掩盖 · 2.3 多核负载/coreDim · 2.4 融合。

## 新增条目（P2 pilot + 第1批 + 第2批 + 剩余）

- [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](diagonal-core-mapping-l2-bank-conflict.md) — OPT-19，分散 L2 bank 访问降低 aiv_mte2_time
- [matmul core mapping 策略选择（列/行/对角）](core-mapping-strategy-selection.md) — DG-8，按 M/N 形状选择 mapping 策略
- [用 tl.constexpr 编译期消除运行时分支](constexpr-branch-elimination.md) — OPT-13，配置级开关编译期分支消除
- [care_padding=False 消除 MTE2→Vector 同步屏障](care-padding-false-ilp.md) — OPT-14，消除 load/compute 同步屏障提升 ILP
- [算子融合模式与约束决策](operator-fusion-patterns.md) — DG-11，常见融合模式与 UB/grid/多消费者约束
- [整数比较/加法必须转 float32，避免 Vector 标量降级](int-compare-cast-float32.md) — CT-13，int64/int32 比较标量降级修复
- [persistent kernel 模式（grid-stride 摊还启动开销）](persistent-kernel-pattern.md) — OPT-15，n_rows >> num_cores 时的 persistent 模式
- [标量循环改写为 tl.dot，提升 Cube 利用率](scalar-loop-to-dot.md) — OPT-16，固定小循环改矩阵乘
- [循环不变量外提（LICM）减少重复地址计算](loop-invariant-code-motion.md) — OPT-17，循环外预计算地址/mask/stride
- [for-loop tiling 启用双缓冲/多缓冲流水线](for-loop-tiling-ilp.md) — OPT-20，小 tile 循环重叠 MTE2/Vector/MTE3
- [dl.async_task 实现 Cube/Vector 异构并行](async-task-cube-vector.md) — OPT-21，融合 kernel 中 Cube/Vector 并发
- [Work-Stealing 动态负载均衡（不均匀任务分发）](work-stealing-dynamic-load-balance.md) — OPT-25，原子计数器动态认领任务
- [pad slot 单次检查提前退出，跳过无效填充序列](pad-slot-early-exit.md) — OPT-18，变长 batch 中 padding 序列快速跳过

> 其余条目由摄入流程增量合并产生。

## trajectory 实例（P3/P5 提取）

- [activation Compute 实践聚合](traj-activation-compute.md) — 5 条 activation 计算优化 experience 聚合
- [activation Scheduling 实践聚合](traj-activation-scheduling.md) — 7 条 activation 调度优化 experience 聚合
- [attention Scheduling 实践聚合](traj-attention-scheduling.md) — 7 条 attention 调度优化 experience 聚合
- [conv Compute 实践聚合](traj-conv-compute.md) — 5 条 conv 计算优化 experience 聚合
- [conv Cross Kernel 实践聚合](traj-conv-cross_kernel.md) — 1 条 conv 跨 kernel 优化 experience 聚合
- [conv Memory Access 实践聚合](traj-conv-memory_access.md) — 8 条 conv 访存优化 experience 聚合
- [conv Scheduling 实践聚合](traj-conv-scheduling.md) — 3 条 conv 调度优化 experience 聚合
- [embedding Scheduling 实践聚合](traj-embedding-scheduling.md) — 1 条 embedding 调度优化 experience 聚合
- [gather_scatter Compute 实践聚合](traj-gather_scatter-compute.md) — 1 条 gather_scatter 计算优化 experience 聚合
- [gather_scatter Scheduling 实践聚合](traj-gather_scatter-scheduling.md) — 1 条 gather_scatter 调度优化 experience 聚合
- [general_ascend Load Balance 实践聚合](traj-general_ascend-load_balance.md) — 2 条 general_ascend 负载均衡 experience 聚合
- [general_perf Block Size 实践聚合](traj-general_perf-block_size.md) — 1 条 general_perf 块大小优化 experience 聚合
- [general_perf Op Split 实践聚合](traj-general_perf-op_split.md) — 1 条 general_perf 算子拆分优化 experience 聚合
- [general_perf Redundancy Elim 实践聚合](traj-general_perf-redundancy_elim.md) — 1 条 general_perf 冗余消除 experience 聚合
- [general_perf Scalar Elimination 实践聚合](traj-general_perf-scalar_elimination.md) — 3 条 general_perf 标量消除 experience 聚合
- [loss Compute 实践聚合](traj-loss-compute.md) — 8 条 loss 计算优化 experience 聚合
- [loss Scheduling 实践聚合](traj-loss-scheduling.md) — 3 条 loss 调度优化 experience 聚合
- [matmul Compute 实践聚合](traj-matmul-compute.md) — 9 条 matmul 计算优化 experience 聚合
- [matmul Scheduling 实践聚合](traj-matmul-scheduling.md) — 19 条 matmul 调度优化 experience 聚合
- [matmul Split 实践聚合](traj-matmul-split.md) — 3 条 matmul 切分优化 experience 聚合
- [misc Compute 实践聚合](traj-misc-compute.md) — 16 条 misc 计算优化 experience 聚合
- [misc Memory Access 实践聚合](../memory-sram/traj-misc-memory_access.md) — 13 条 misc 访存优化 experience 聚合
- [misc Scheduling 实践聚合](traj-misc-scheduling.md) — 7 条 misc 调度优化 experience 聚合
- [moe Compute 实践聚合](traj-moe-compute.md) — 1 条 moe 计算优化 experience 聚合
- [normalization Compute 实践聚合](../algorithm-numerics/traj-normalization-compute.md) — 7 条 normalization 计算优化 experience 聚合
- [normalization Scheduling 实践聚合](traj-normalization-scheduling.md) — 2 条 normalization 调度优化 experience 聚合
- [reduction Compute 实践聚合](traj-reduction-compute.md) — 3 条 reduction 计算优化 experience 聚合
- [rope Compute 实践聚合](traj-rope-compute.md) — 6 条 rope 计算优化 experience 聚合
- [rope Scheduling 实践聚合](traj-rope-scheduling.md) — 1 条 rope 调度优化 experience 聚合
- [softmax Compute 实践聚合](traj-softmax-compute.md) — 5 条 softmax 计算优化 experience 聚合
- [solve_tril Compute 实践聚合](traj-solve_tril-compute.md) — 2 条 solve_tril 计算优化 experience 聚合
- [solve_tril Scheduling 实践聚合](traj-solve_tril-scheduling.md) — 1 条 solve_tril 调度优化 experience 聚合
- [ssm Compute 实践聚合](traj-ssm-compute.md) — 8 条 ssm 计算优化 experience 聚合
- [ssm Scalar Elimination 实践聚合](traj-ssm-scalar_elimination.md) — 2 条 ssm 标量消除 experience 聚合
- [exp_matmul_vec_parallel_001 — matmul 向量并行 + activation fusion](instances/traj-exp-matmul-vec-parallel-001.md) — 独立 trajectory 实例
