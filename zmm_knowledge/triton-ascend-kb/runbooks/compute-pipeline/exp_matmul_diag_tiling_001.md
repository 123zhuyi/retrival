---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: 'Scheduling Optimization: diagonal tiling strategy'
description: 'matmul 场景下，Scheduling Optimization: diagonal tiling strategy，收益 25%'
tags: [多核并行与负载均衡, matrix-operations, matmul]
confidence: provisional
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Scheduling Optimization: diagonal tiling strategy

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：M、N 方向块数低于 BLOCK_TRESHHOLD（4-8）时回退顺序分核；右矩阵不超 L2 容量时不存在 thrashing，对角收益减小；绑定 910B 与两组 512B 对齐 tile 配置，换硬件需重调搜索空间）

### 原则
- **diagonal tiling strategy**：Replaces the traditional row-major (horizontal) core splitting with an 8x8 diagonal tiling strategy. The kernel calculates task indices (`task_m_idx`, `task_n_idx`) such that program IDs are mapped to blocks along the diagonal of the MxN grid, rather than filling rows sequentially.
- **bank conflict mitigation**：In horizontal splitting, multiple cores simultaneously access the same row of the left matrix (mat_a), causing severe bank conflicts. Diagonal tiling distributes these accesses across different memory banks, significantly improving memory access bandwidth and efficiency.
- **l2 cache locality optimization**：Horizontal splitting requires loading the entire right matrix (mat_b) for each row of the output, often exceeding L2 cache capacity and causing thrashing. Diagonal tiling reuses active blocks of mat_b across adjacent diagonal steps, greatly increasing L2 cache hit rates and reducing global memory access latency.
- **adaptive thresholding**：Introduces a `BLOCK_TRESHHOLD` parameter (tuned between 4-8) to conditionally enable diagonal tiling. This ensures the optimization is only applied when the matrix dimensions are large enough to justify the scheduling overhead, falling back to standard tiling for smaller shapes.
- **diagonal tiling strategy**：将传统行主序（水平）分核替换为 8×8 对角线分块策略。Kernel 通过计算 `task_m_idx`、`task_n_idx`，将 program ID 映射到 MxN 网格的对角线方向， 而非逐行顺序填充。具体实现中，利用当前块在局部 BLOCK_TRESHHOLD×BLOCK_TRESHHOLD 子格内的相对位置（`localRelativeBlock`），结合最小公倍数（lcm）求解 N 轴坐标， 确保同一时刻各核心访问的任务块在 M、N 方向均匀分布。

- **bank conflict mitigation**：水平分核时，同一时刻大量核心并发访问左矩阵（mat_a）的同一行数据，引发严重 Bank 冲突，硬件访问效率显著下降。对角线分块将各核心的访问分散至不同内存 Bank， 从根本上消除冲突热点，大幅提升内存访问带宽与效率。

- **l2 cache locality optimization**：水平分核完成一整行 mat_c 的计算后，已将右矩阵（mat_b）全部数据消费完毕； 当 mat_b 较大时将超出 L2 Cache 容量，导致频繁换入换出（Cache Thrashing）， 后续每行均产生大量 Cache Miss。对角线分块使相邻对角步骤间可复用 mat_b 的活跃 数据块，显著提升 L2 Cache 命中率，降低全局内存访问延迟。

- **adaptive thresholding**：引入可调参数 `BLOCK_TRESHHOLD`（通过 autotune 在 4～8 范围内搜索最优值）， 仅当 M、N 方向的块数均不低于该阈值时（即 `NUM_BLOCKS_M >= BLOCK_TRESHHOLD and NUM_BLOCKS_N >= BLOCK_TRESHHOLD`）才启用对角线分块；否则回退至传统顺序 分核，避免小矩阵场景下调度开销得不偿失。

- **block size tuning**：autotune 搜索空间覆盖两组基础分块配置： （1）blockM=128, blockN=256, blockK=256； （2）blockM=256, blockN=128, blockK=256。 两组配置均与 NPU 芯片亲和的 512B 对齐场景匹配，结合 `BLOCK_TRESHHOLD` 共同 构成 10 个候选 Config，以 [M, N, K] 为 key 进行离线调优。

- **instruction level hint**：对 `mat_a_block` 与 `mat_b_block` 均调用 `tl.compile_hint(tensor, "dot_pad_only_k")`， 指示编译器仅在 K 轴方向进行 Pad 补齐，避免不必要的 M/N 轴 Pad，降低无效计算量， 提升指令执行效率。

- **core parallel launch**：以硬件实际 AI Core 数（`num_aicore`，通过 `driver.active.utils.get_device_properties` 动态获取）作为 grid size 启动 Kernel，每个 Core 通过步长为 `num_cores` 的循环 处理多个任务块，实现负载均衡与全核心利用。

- **autotune config**：{'search_space': {'blockM': [128, 256], 'blockN': [256, 128], 'blockK': [256], 'BLOCK_TRESHHOLD': [4, 5, 6, 7, 8]}, 'total_configs': 10, 'tune_key': ['M', 'N', 'K']}

**坏实践（反例）**：水平（行主序）分核 → 同 wave 核心并发访问左矩阵同一行引发严重 bank 冲突；且算完一整行已耗尽右矩阵，大右矩阵超出 L2 容量导致 cache thrashing → ✅ 正解即本条：8×8 对角线分块 + 阈值门控回退 + 按物理核数动态启动。

**已知实例**：matmul §raw (`experiences/matmul/exp_matmul_diag_tiling_001.yaml`)
- 声称性能收益：25%
- 备注：本优化专为昇腾 910B 大规模矩阵场景设计，核心目标是缓解内存带宽压力与 L2 Cache 容量 瓶颈。对角线分核在 M、N 方向均超过 BLOCK_TRESHHOLD 个基本块时生效，右矩阵大小超过 L2 Cache 容量时优化效果尤为显著。输出结果以 bfloat16 写回（`mat_c_block.to(tl.bfloat16)`）， 中间累加保持 float32 精度以保证数值稳定性。

<!-- okf:related:start -->

# 相关

- 相关主题: [异构并行融合：MatMul 派 Cube、归一化+位置编码派 Vector，dl.async_task 异步掩盖](../portability/exp_fused_matmul_rms_rope_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [MatMul 调度：对角遍历 grouped_launch + dot_pad_only_k hint + 寄存器原位累加](../portability/exp_matmul_ascend_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](diagonal-core-mapping-l2-bank-conflict.md) — 主题相关：共享 matmul
- 相关主题: [Scheduling Optimization: dynamic autotuning](exp_addmm_ascend_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: block ptr utilization](exp_block_matmul_inv_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: grouped scheduling](exp_bmm_grouped_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: tiling strategy](exp_cube1_qk_gemm_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: kernel fusion and recomputation](exp_entropy_backward_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: persistent kernel grid stride](exp_entropy_backward_d_weight_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_entropy_backward_fused_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_flash_attn_gridstride_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_attn_preprocess_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: operator fusion](exp_fused_gemm_scale_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: async pipeline scheduling](exp_fused_matmul_norm_sigmoid_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rms_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rmsnorm_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rope_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: hardware unit isolation](exp_fused_norm_rope_matmul_001.md) — 主题相关：共享 matmul, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic work stealing](exp_grouped_gemm_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_gemm_bkmajor_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_gemm_bnmajor_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_matmul_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dynamic load balancing](exp_grouped_matmul_bkmajor_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: activation recomputation](exp_hstu_attn_bwd_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: static task scheduling](exp_hstu_attn_sched_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: diagonal traversal pattern](exp_launch_diagonal_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: diagonal partitioning](exp_matmul_ascend_diag_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: block tiling strategy](exp_matmul_bias_fused_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: hardware specific instruction](exp_matmul_qn_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: micro tiling strategy](exp_matmul_small_tile_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: block swizzling](exp_matmul_swizzle_hardware_hint_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: small tile strategy](exp_matmul_triton_base_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: vector parallelism strategy](exp_matmul_vec_parallel_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: kernel fusion](exp_mhc_fused_gemm_sqrsum_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: pid grouping strategy](exp_mm_grouped_reorder_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: grid stride scheduling](exp_partial_matmul_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: dynamic persistent scheduling](exp_partial_matmul_dynamic_001.md) — 主题相关：共享 matmul, matrix-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: fused rope computation](exp_rope_matmul_fused_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: block tiling strategy](exp_sdpa_bwd_kv_001.md) — 主题相关：共享 matrix-operations, 多核并行与负载均衡

<!-- okf:related:end -->
