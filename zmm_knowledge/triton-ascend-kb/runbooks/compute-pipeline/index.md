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

## 已知实例 append（2026-09-21）

- [for-loop-tiling-ilp](for-loop-tiling-ilp.md) — append 已知实例：SelfAttention 投影 GEMM（ascend910b）K 归约迭代数独立杠杆（half BK 128 / fp32 BK 64）+0.3%~4.4%

## 新增条目（raw experiences 导入，待 review）

- [Scheduling Optimization: dynamic autotuning](exp_addmm_ascend_001.md) — matmul 场景下，Scheduling Optimization: dynamic autotuning，收益 25%
- [Scheduling Optimization: dual task pipelining](exp_attn_ascend_dual_pipeline_001.md) — attention 场景下，Scheduling Optimization: dual task pipelining，收益 55%
- [Compute Optimization: mixed precision compute](exp_attn_bwd_q_001.md) — attention 场景下，Compute Optimization: mixed precision compute，收益 35%
- [Compute Optimization: blocking strategy](exp_attn_qk_cube_001.md) — attention 场景下，Compute Optimization: blocking strategy，收益 25%
- [Compute Optimization: online softmax algorithm](exp_attn_reduce_stage2_001.md) — attention 场景下，Compute Optimization: online softmax algorithm，收益 30%
- [Compute Optimization: attention sinks mechanism](exp_attn_sinks_prefill_001.md) — attention 场景下，Compute Optimization: attention sinks mechanism，收益 35%
- [Compute Optimization: vectorization alignment](exp_backward_sum_001.md) — reduction 场景下，Compute Optimization: vectorization alignment，收益 25%
- [Compute Optimization: block ptr tiling](exp_block_inverse_001.md) — misc 场景下，Compute Optimization: block ptr tiling，收益 35%
- [Compute Optimization: block ptr utilization](exp_block_matmul_inv_001.md) — matmul 场景下，Compute Optimization: block ptr utilization，收益 35%
- [Scheduling Optimization: grouped scheduling](exp_bmm_grouped_001.md) — matmul 场景下，Scheduling Optimization: grouped scheduling，收益 25%
- [Scheduling Optimization: two dimensional grid strategy](exp_causal_conv1d_prefill_003.md) — conv 场景下，Scheduling Optimization: two dimensional grid strategy，收益 2D Grid vs 1D Grid 有 25-39% 性能优势（搜索验证）；整体 Prefill kernel 1.60x 平均加速比
- [Compute Optimization: pointer base precompute](exp_causal_conv1d_prefill_004.md) — conv 场景下，Compute Optimization: pointer base precompute，收益 减少 10-20% 的标量地址计算指令（取决于循环体大小），在大 seq_len 场景收益更显著
- [Scheduling Optimization: pad slot early exit](exp_causal_conv1d_prefill_005.md) — conv 场景下，Scheduling Optimization: pad slot early exit，收益 在有 padding 序列的 batch 中节省无效计算（如 25% padding→~25% 资源节省）；最小开销仅 1 load + 1 cmp
- [Compute Optimization: conv bias silu triple fusion](exp_causal_conv1d_prefill_006.md) — conv 场景下，Compute Optimization: conv bias silu triple fusion，收益 消除 2 个额外 kernel launch（延迟降低约 30-40%）；Bias 寄存器驻留节省重复 load
- [Compute Optimization: dim is multiple mask elision](exp_causal_conv1d_prefill_007.md) — conv 场景下，Compute Optimization: dim is multiple mask elision，收益 减少 5-15% 标量指令（取决于 tile 数量），大 D 维度场景收益更显著
- [Compute Optimization: w loop unrolling with threshold](exp_causal_conv1d_update_002.md) — conv 场景下，Compute Optimization: w loop unrolling with threshold，收益 D≥2048: 2.3x-3.1x; D≤256: 0.41x-0.75x（需阈值控制）
- [Compute Optimization: dim is multiple mask elimination](exp_causal_conv1d_update_005.md) — conv 场景下，Compute Optimization: dim is multiple mask elimination，收益 减少 5-15% 的标量指令数（取决于 tile 数量），在 D 维度较大且整除 D_CHK_SIZE 时收益更显著
- [Cross-kernel Optimization Transfer: token loop vs w loop decision](exp_causal_conv1d_update_006.md) — conv 场景下，Cross-kernel Optimization Transfer: token loop vs w loop decision，收益 Bias 融合消除 1 次 kernel launch（约 10-20% 延迟）、指针预计算减少 5-10% 标量指令
- [Scheduling Optimization: bdt layout rationale](exp_causal_conv1d_update_007.md) — conv 场景下，Scheduling Optimization: bdt layout rationale，收益 BDT 布局 + 3D 任务分解 + 双缓冲状态更新组合贡献约 30-40% 整体性能提升（vs BTD + naive tiling）
- [Compute Optimization: numerical stability](exp_celoss_001.md) — loss 场景下，Compute Optimization: numerical stability，收益 35%
- [Compute Optimization: two pass algorithm](exp_celoss_bwd_001.md) — loss 场景下，Compute Optimization: two pass algorithm，收益 35%
- [Compute Optimization: numerical stability](exp_celoss_smooth_001.md) — loss 场景下，Compute Optimization: numerical stability，收益 30%
- [Compute Optimization: two pass algorithm](exp_celoss_smooth_bwd_001.md) — loss 场景下，Compute Optimization: two pass algorithm，收益 30%
- [Compute Optimization: chunked scan strategy](exp_chunk_cumsum_001.md) — ssm 场景下，Compute Optimization: chunked scan strategy，收益 25%
- [Compute Optimization: block ptr tiling](exp_chunk_fwd_001.md) — ssm 场景下，Compute Optimization: block ptr tiling，收益 35%
- [Compute Optimization: block ptr abstraction](exp_chunk_kkt_001.md) — ssm 场景下，Compute Optimization: block ptr abstraction，收益 35%
- [Compute Optimization: online softmax algorithm](exp_crossentropy_001.md) — loss 场景下，Compute Optimization: online softmax algorithm，收益 40%
- [Compute Optimization: grid stride scheduling](exp_crossentropy_backward_001.md) — loss 场景下，Compute Optimization: grid stride scheduling，收益 60%
- [Compute Optimization: tiling strategy](exp_cube1_qk_gemm_001.md) — matmul 场景下，Compute Optimization: tiling strategy，收益 35%
- [Compute Optimization: gemm based histogram](exp_digit_hist_001.md) — misc 场景下，Compute Optimization: gemm based histogram，收益 50%
- [Scheduling Optimization: grid stride loop](exp_element_mul_001.md) — misc 场景下，Scheduling Optimization: grid stride loop，收益 25%
- [Scheduling Optimization: 2D Block Tiling](exp_embedding_backward_001.md) — embedding 场景下，Scheduling Optimization: 2D Block Tiling，收益 35%
- [Scheduling Optimization: kernel fusion and recomputation](exp_entropy_backward_001.md) — loss 场景下，Scheduling Optimization: kernel fusion and recomputation，收益 25%
- [Scheduling Optimization: persistent kernel grid stride](exp_entropy_backward_d_weight_001.md) — loss 场景下，Scheduling Optimization: persistent kernel grid stride，收益 25%
- [Scheduling Optimization: grid stride scheduling](exp_entropy_backward_fused_001.md) — loss 场景下，Scheduling Optimization: grid stride scheduling，收益 35%
- [Compute Optimization: numerical stability reduction](exp_entropy_epilogue_001.md) — loss 场景下，Compute Optimization: numerical stability reduction，收益 25%
- [Scheduling Optimization: compilation specialization control](exp_expand_serial_001.md) — misc 场景下，Scheduling Optimization: compilation specialization control，收益 Baseline
- [Scheduling Optimization: grid stride loop](exp_flash_attn_gridstride_001.md) — attention 场景下，Scheduling Optimization: grid stride loop，收益 35%
- [Compute Optimization: mixed precision compute](exp_flashattn_bwd_kv_001.md) — attention 场景下，Compute Optimization: mixed precision compute，收益 35%
- [Compute Optimization: split k parallelism](exp_flashattn_splitk_gqa_001.md) — attention 场景下，Compute Optimization: split k parallelism，收益 35%
- [流程优化: 标准 score-归一化-加权三步流程](exp_fused_attention_001.md) — attention 场景下，流程优化: 标准 score-归一化-加权三步流程，收益 15%
- [Scheduling Optimization: operator fusion](exp_fused_attn_preprocess_001.md) — attention 场景下，Scheduling Optimization: operator fusion，收益 35%
- [Compute Optimization: Operator Fusion](exp_fused_delta_rule_001.md) — ssm 场景下，Compute Optimization: Operator Fusion，收益 35%
- [Compute Optimization: deep kernel fusion](exp_fused_delta_rule_npu_001.md) — ssm 场景下，Compute Optimization: deep kernel fusion，收益 35%
- [Compute Optimization: operator fusion](exp_fused_entropy_001.md) — loss 场景下，Compute Optimization: operator fusion，收益 60%
- [Compute Optimization: operator fusion](exp_fused_gating_001.md) — activation 场景下，Compute Optimization: operator fusion，收益 25%
- [Compute Optimization: operator fusion](exp_fused_gemm_scale_001.md) — matmul 场景下，Compute Optimization: operator fusion，收益 25%
- [Compute Optimization: operator fusion](exp_fused_index_001.md) — misc 场景下，Compute Optimization: operator fusion，收益 45%
- [Scheduling Optimization: async pipeline scheduling](exp_fused_matmul_norm_sigmoid_001.md) — matmul 场景下，Scheduling Optimization: async pipeline scheduling，收益 30%
- [Scheduling Optimization: operator fusion](exp_fused_matmul_rms_001.md) — matmul 场景下，Scheduling Optimization: operator fusion，收益 25%
- [Scheduling Optimization: operator fusion](exp_fused_matmul_rmsnorm_001.md) — matmul 场景下，Scheduling Optimization: operator fusion，收益 25%
- [Scheduling Optimization: operator fusion](exp_fused_matmul_rope_001.md) — matmul 场景下，Scheduling Optimization: operator fusion，收益 25%
- [Scheduling Optimization: hardware unit isolation](exp_fused_norm_rope_matmul_001.md) — matmul 场景下，Scheduling Optimization: hardware unit isolation，收益 35%
- [Compute Optimization: operator fusion](exp_fused_qkv_rmsnorm_rope_001.md) — normalization 场景下，Compute Optimization: operator fusion，收益 25%
- [Compute Optimization: operator fusion](exp_fused_rmsnorm_sigmoid_001.md) — normalization 场景下，Compute Optimization: operator fusion，收益 35%
- [Scheduling Optimization: deep operator fusion](exp_fused_transformer_block_001.md) — misc 场景下，Scheduling Optimization: deep operator fusion，收益 35%
- [Scheduling Optimization: grid stride loop](exp_geglu_flat_001.md) — activation 场景下，Scheduling Optimization: grid stride loop，收益 25%
- [Scheduling Optimization: autotune block m](exp_gelu_bwd_001.md) — activation 场景下，Scheduling Optimization: autotune block m，收益 20%
- [Scheduling Optimization: grid stride loop](exp_gelu_fwd_001.md) — activation 场景下，Scheduling Optimization: grid stride loop，收益 15%
- [Scheduling Optimization: dynamic work stealing](exp_grouped_gemm_001.md) — matmul 场景下，Scheduling Optimization: dynamic work stealing，收益 35%
- [Scheduling Optimization: dynamic load balancing](exp_grouped_gemm_bkmajor_001.md) — matmul 场景下，Scheduling Optimization: dynamic load balancing，收益 35%
- [Scheduling Optimization: dynamic load balancing](exp_grouped_gemm_bnmajor_001.md) — matmul 场景下，Scheduling Optimization: dynamic load balancing，收益 35%
- [Scheduling Optimization: dynamic load balancing](exp_grouped_matmul_001.md) — matmul 场景下，Scheduling Optimization: dynamic load balancing，收益 25%
- [Scheduling Optimization: dynamic load balancing](exp_grouped_matmul_bkmajor_001.md) — matmul 场景下，Scheduling Optimization: dynamic load balancing，收益 35%
- [Compute Optimization: numerical stability](exp_grpo_loss_001.md) — misc 场景下，Compute Optimization: numerical stability，收益 30%
- [Compute Optimization: register accumulation](exp_hist_count_001.md) — misc 场景下，Compute Optimization: register accumulation，收益 50%
- [Scheduling Optimization: activation recomputation](exp_hstu_attn_bwd_001.md) — attention 场景下，Scheduling Optimization: activation recomputation，收益 25%
- [Scheduling Optimization: static task scheduling](exp_hstu_attn_sched_001.md) — attention 场景下，Scheduling Optimization: static task scheduling，收益 15%
- [标量消除: block_ptr boundary_check 与 i32 比较定向消除](exp_kda_bwd_intra_scalar_opt_001.md) — ssm 场景下，标量消除: block_ptr boundary_check 与 i32 比较定向消除，收益 50%
- [计算优化: 对角块标量循环改 tl.dot](exp_kda_bwd_scalar_loop_to_dot_001.md) — ssm 场景下，计算优化: 对角块标量循环改 tl.dot，收益 40%
- [标量消除: i64/i32 标量降级定向消除](exp_kda_bwd_wy_scalar_opt_001.md) — ssm 场景下，标量消除: i64/i32 标量降级定向消除，收益 233%
- [计算优化: phase split 与 barrier 外提](exp_kda_bwd_wy_structure_opt_001.md) — ssm 场景下，计算优化: phase split 与 barrier 外提，收益 30%
- [Scheduling Optimization: diagonal traversal pattern](exp_launch_diagonal_001.md) — misc 场景下，Scheduling Optimization: diagonal traversal pattern，收益 20%
- [Compute Optimization: Operator Fusion](exp_layernorm_npu_001.md) — normalization 场景下，Compute Optimization: Operator Fusion，收益 30%
- [Compute Optimization: npu simd vectorization](exp_layernorm_npu_simd_001.md) — normalization 场景下，Compute Optimization: npu simd vectorization，收益 35%
- [Scheduling Optimization: heterogeneous async scheduling](exp_lightning_attn_001.md) — attention 场景下，Scheduling Optimization: heterogeneous async scheduling，收益 45%
- [Scheduling Optimization: persistent kernel grid stride loop](exp_linear_persistent_001.md) — misc 场景下，Scheduling Optimization: persistent kernel grid stride loop，收益 25%
- [Compute Optimization: two pass algorithm](exp_log_softmax_001.md) — softmax 场景下，Compute Optimization: two pass algorithm，收益 30%
- [Compute Optimization: two pass algorithm](exp_log_softmax_backward_001.md) — softmax 场景下，Compute Optimization: two pass algorithm，收益 25%
- [计算优化: fixpipe 随路量化 + K 维循环展开 + 预取](exp_matmul_002.md) — matmul 场景下，计算优化: fixpipe 随路量化 + K 维循环展开 + 预取，收益 35%
- [Scheduling Optimization: diagonal partitioning](exp_matmul_ascend_diag_001.md) — matmul 场景下，Scheduling Optimization: diagonal partitioning，收益 25%
- [Compute Optimization: block tiling strategy](exp_matmul_bias_fused_001.md) — matmul 场景下，Compute Optimization: block tiling strategy，收益 30%
- [Scheduling Optimization: diagonal tiling strategy](exp_matmul_diag_tiling_001.md) — matmul 场景下，Scheduling Optimization: diagonal tiling strategy，收益 25%
- [Scheduling Optimization: hardware specific instruction](exp_matmul_qn_001.md) — matmul 场景下，Scheduling Optimization: hardware specific instruction，收益 25%
- [Scheduling Optimization: micro tiling strategy](exp_matmul_small_tile_001.md) — matmul 场景下，Scheduling Optimization: micro tiling strategy，收益 15%
- [Scheduling Optimization: block swizzling](exp_matmul_swizzle_hardware_hint_001.md) — matmul 场景下，Scheduling Optimization: block swizzling，收益 25%
- [Compute Optimization: small tile strategy](exp_matmul_triton_base_001.md) — matmul 场景下，Compute Optimization: small tile strategy，收益 15%
- [Compute Optimization: vector parallelism strategy](exp_matmul_vec_parallel_001.md) — matmul 场景下，Compute Optimization: vector parallelism strategy，收益 35%
- [Compute Optimization: kernel fusion](exp_mhc_fused_gemm_sqrsum_001.md) — matmul 场景下，Compute Optimization: kernel fusion，收益 30%
- [Compute Optimization: Operator Fusion](exp_mhc_post_fusion_001.md) — misc 场景下，Compute Optimization: Operator Fusion，收益 35%
- [Scheduling Optimization: pid grouping strategy](exp_mm_grouped_reorder_001.md) — matmul 场景下，Scheduling Optimization: pid grouping strategy，收益 25%
- [Compute Optimization: memory alignment](exp_moe_wgrad_npu_001.md) — moe 场景下，Compute Optimization: memory alignment，收益 25%
- [Compute Optimization: vectorized binary search](exp_multinomial_001.md) — misc 场景下，Compute Optimization: vectorized binary search，收益 35%
- [Compute Optimization: grid stride scheduling](exp_partial_matmul_001.md) — matmul 场景下，Compute Optimization: grid stride scheduling，收益 25%
- [Scheduling Optimization: dynamic persistent scheduling](exp_partial_matmul_dynamic_001.md) — matmul 场景下，Scheduling Optimization: dynamic persistent scheduling，收益 30%
- [Compute Optimization: vectorized memory access](exp_partial_rope_001.md) — rope 场景下，Compute Optimization: vectorized memory access，收益 25%
- [性能避坑: 避免多维 stride 访问](exp_perf_007.md) — general_perf 场景下，性能避坑: 避免多维 stride 访问，收益 -40%（反例警示）
- [算子拆分优化: 按计算阶段拆分为简单 kernel](exp_perf_010.md) — general_perf 场景下，算子拆分优化: 按计算阶段拆分为简单 kernel，收益 25%
- [标量消除: i32 比较转 fp32 走向量单元](exp_perf_012.md) — general_perf 场景下，标量消除: i32 比较转 fp32 走向量单元，收益 35%
- [标量消除: min() 运行时分支改 tl.where](exp_perf_013.md) — general_perf 场景下，标量消除: min() 运行时分支改 tl.where，收益 15%
- [标量消除: tl.cdiv 编译期循环边界](exp_perf_014.md) — general_perf 场景下，标量消除: tl.cdiv 编译期循环边界，收益 10%
- [Compute Optimization: fused mrope logic](exp_qwen2vl_mrope_001.md) — rope 场景下，Compute Optimization: fused mrope logic，收益 25%
- [Compute Optimization: compile time dispatch](exp_radix_convert_001.md) — misc 场景下，Compute Optimization: compile time dispatch，收益 20%
- [Scheduling Optimization: lookback algorithm](exp_radix_sort_scatter_001.md) — gather_scatter 场景下，Scheduling Optimization: lookback algorithm，收益 25%
- [Compute Optimization: high precision accumulation](exp_recompute_w_u_001.md) — misc 场景下，Compute Optimization: high precision accumulation，收益 25%
- [Compute Optimization: vectorized memory access](exp_reduce_split_001.md) — reduction 场景下，Compute Optimization: vectorized memory access，收益 20%
- [Compute Optimization: vectorized load](exp_reject_sampler_001.md) — misc 场景下，Compute Optimization: vectorized load，收益 35%
- [Scheduling Optimization: vectorized global load](exp_rejection_sample_001.md) — misc 场景下，Scheduling Optimization: vectorized global load，收益 40%
- [Compute Optimization: two pass algorithm](exp_rmsnorm_backward_npu_001.md) — normalization 场景下，Compute Optimization: two pass algorithm，收益 25%
- [Compute Optimization: dynamic tiling strategy](exp_rmsnorm_bwd_001.md) — normalization 场景下，Compute Optimization: dynamic tiling strategy，收益 30%
- [Scheduling Optimization: grid stride scheduling](exp_rmsnorm_fwd_001.md) — normalization 场景下，Scheduling Optimization: grid stride scheduling，收益 25%
- [Scheduling Optimization: grid stride scheduling](exp_rmsnorm_kv_001.md) — normalization 场景下，Scheduling Optimization: grid stride scheduling，收益 25%
- [Compute Optimization: operator fusion](exp_rmsnorm_rope_001.md) — normalization 场景下，Compute Optimization: operator fusion，收益 25%
- [Compute Optimization: cos sin reuse](exp_rope_001.md) — rope 场景下，Compute Optimization: cos sin reuse，收益 15%
- [Scheduling Optimization: grid stride loop](exp_rope_backward_001.md) — rope 场景下，Scheduling Optimization: grid stride loop，收益 25%
- [Compute Optimization: fused rope computation](exp_rope_matmul_fused_001.md) — matmul 场景下，Compute Optimization: fused rope computation，收益 25%
- [Compute Optimization: kernel fusion](exp_rope_mqa_001.md) — attention 场景下，Compute Optimization: kernel fusion，收益 30%
- [Compute Optimization: vectorized non interleaved path](exp_rope_siso_001.md) — rope 场景下，Compute Optimization: vectorized non interleaved path，收益 25%
- [Compute Optimization: vectorized memory access](exp_rope_triton_001.md) — rope 场景下，Compute Optimization: vectorized memory access，收益 25%
- [Compute Optimization: batched reduction replacement](exp_scatter_wgrad_001.md) — gather_scatter 场景下，Compute Optimization: batched reduction replacement，收益 35%
- [Compute Optimization: grid stride loop](exp_sda_bwd_reduce_001.md) — reduction 场景下，Compute Optimization: grid stride loop，收益 25%
- [Scheduling Optimization: block tiling strategy](exp_sdpa_bwd_kv_001.md) — attention 场景下，Scheduling Optimization: block tiling strategy，收益 25%
- [Compute Optimization: autotuning block size](exp_sdpa_bwd_q_001.md) — attention 场景下，Compute Optimization: autotuning block size，收益 30%
- [Compute Optimization: online softmax](exp_sdpa_infer_ascend_001.md) — attention 场景下，Compute Optimization: online softmax，收益 50%
- [Scheduling Optimization: block local sorting](exp_shuffle_blockwise_001.md) — misc 场景下，Scheduling Optimization: block local sorting，收益 35%
- [Scheduling Optimization: grid stride scheduling](exp_sigmoid_grid_stride_001.md) — activation 场景下，Scheduling Optimization: grid stride scheduling，收益 15%
- [Scheduling Optimization: autotuning strategy](exp_silu_bwd_001.md) — activation 场景下，Scheduling Optimization: autotuning strategy，收益 15%
- [Scheduling Optimization: grid stride loop](exp_silu_fwd_001.md) — activation 场景下，Scheduling Optimization: grid stride loop，收益 25%
- [Compute Optimization: vectorized accumulation strategy](exp_softmax_inner_002.md) — softmax 场景下，Compute Optimization: vectorized accumulation strategy，收益 30%
- [Compute Optimization: online softmax algorithm](exp_softmax_non_inner_001.md) — softmax 场景下，Compute Optimization: online softmax algorithm，收益 30%
- [Compute Optimization: online softmax algorithm](exp_softmax_online_001.md) — softmax 场景下，Compute Optimization: online softmax algorithm，收益 35%
- [Compute Optimization: redundant mask elimination by matrix property](exp_solve_tril_001.md) — solve_tril 场景下，Compute Optimization: redundant mask elimination by matrix property，收益 1.33x-1.87x（BT=16 kernel）；~1.40x（BT=64 kernel）；BT=32 无此冗余故无变化
- [Compute Optimization: constexpr dual path full vs boundary](exp_solve_tril_003.md) — solve_tril 场景下，Compute Optimization: constexpr dual path full vs boundary，收益 此模式保证快速路径（99% chunk）达到最优展开性能，避免统一运行时循环导致的性能退化
- [Scheduling Optimization: three tier hierarchical dispatch](exp_solve_tril_004.md) — solve_tril 场景下，Scheduling Optimization: three tier hierarchical dispatch，收益 通过分治将 O(BT³) 求逆降为 O(BT²) 的递推 + O(BT) 个 16×16 矩阵乘法
- [Compute Optimization: padding strategy](exp_sort_bitonic_001.md) — misc 场景下，Compute Optimization: padding strategy，收益 25%
- [Compute Optimization: sparse iteration strategy](exp_sparse_attn_topk_001.md) — attention 场景下，Compute Optimization: sparse iteration strategy，收益 40%
- [Compute Optimization: sparse data access](exp_sparse_flash_attn_decode_001.md) — attention 场景下，Compute Optimization: sparse data access，收益 45%
- [Compute Optimization: scalar loop vectorization](exp_spec_decode_verify_scalar.md) — misc 场景下，Compute Optimization: scalar loop vectorization，收益 300%
- [Compute Optimization: kernel fusion](exp_ssm_gated_delta_001.md) — ssm 场景下，Compute Optimization: kernel fusion，收益 50%
- [Compute Optimization: activation recomputation](exp_swiglu_bwd_001.md) — activation 场景下，Compute Optimization: activation recomputation，收益 25%
- [Compute Optimization: software pipelining](exp_swiglu_bwd_flat_001.md) — activation 场景下，Compute Optimization: software pipelining，收益 35%
- [Scheduling Optimization: mixed precision computation](exp_swiglu_fwd_001.md) — activation 场景下，Scheduling Optimization: mixed precision computation，收益 15%
- [Compute Optimization: interleaved vectorization](exp_swiglu_oai_001.md) — activation 场景下，Compute Optimization: interleaved vectorization，收益 25%
- [Compute Optimization: operator fusion](exp_swiglu_quant_001.md) — activation 场景下，Compute Optimization: operator fusion，收益 35%
- [Compute Optimization: vectorized probability calculation](exp_top_p_sampling_001.md) — misc 场景下，Compute Optimization: vectorized probability calculation，收益 30%
- [Compute Optimization: kernel fusion](exp_topp_filter_001.md) — misc 场景下，Compute Optimization: kernel fusion，收益 35%
- [Compute Optimization: micro kernel tiling](exp_tril_solve_001.md) — misc 场景下，Compute Optimization: micro kernel tiling，收益 35%
- [Compute Optimization: vectorized chunked processing](exp_triton_rope_001.md) — rope 场景下，Compute Optimization: vectorized chunked processing，收益 25%

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

## 新增条目（GDN 三篇优化实践蒸馏 + staging 合入，2026-08-27）

- [chunk 递推算子的依赖二分法与 Pingpong 下发调度](chunk-recurrence-pingpong-dispatch.md) — 块内 CV/块间时序依赖分类治理，PingpongB 相对朴素 +24%~41%
- [依赖无关输入预加载与带宽仲裁意识](cv-preload-and-bandwidth-arbitration.md) — load 提前于等待边，AscendC Fwd_h +10%~22%；晚读出为 Triton 不可达区
- [负载均衡分核：grid 对齐物理核数，核内循环替代多维 grid](load-balance-grid-sizing.md) — GPU 三维 grid 迁移第一课，GDN 实测 +90%~99.6%
- [复合块多 kernel pipeline 分解模式](composite-kernel-decomposition.md) — ~30-op 复合块骨架：universal GEMM + LN + dual-path flash-attn

## 新增条目（ApplyRotaryPosEmb P4 后端契约沉淀，2026-09-09）

- [循环不变量按复用组外提](loop-invariant-group-hoisting.md) — 广播小表（cos/sin 等）按复用组外循环外提，实测单点 +100%；cache 驻留≠复用
- [Triton-Ascend 移位 load 降级契约](triton-ascend-shifted-load-degradation.md) — 大位移移位 load ~25x 标量化、2D tile ±1 不合并；配对/置换走 tl.split/tl.gather 寄存器路径
- [tl.trans 标量化灾难与寄存器重排替代](triton-ascend-trans-scalarization.md) — tl.trans 即使小转置也 ~15x 标量化；安全集 = reshape/split/join/interleave

## 新增条目（GELU 回流沉淀，2026-09-18）

- [逐元素小 shape 分裂 kernel 禁用固定 BLOCK=阈值上限写法](exp_elementwise_small_split_falsified_001.md) — numel=100 也算 8192 宽 tile，实测 geomean 0.5929 vs 0.7293 劣化；多档小 BLOCK 变体未证伪
- [逐元素算子 libdevice 超越函数 VECTOR 75%+ 且 top 全为 BAR(pipe=VECTOR) → 数学链 pipe_barrier 串行](exp_elementwise_erf_vector_bound_001.md) — erf 有理分式链 ~20 向量 op 逐 op barrier，计算 bound 非同步问题；解耦类优化点不适用
