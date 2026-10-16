---
kind: index
type: section_index
title: runbooks/memory-sram/（占位）
description: 分类1 内存与切块优化点库（OPT-*/AP-*/CT-*，跨算子单一共享、增量合并）。
---

# 分类1 内存与切块

覆盖二级维度：1.1 访存效率 · 1.2 Tiling/SRAM 预算 · 1.3 ND↔NZ · 1.4 32B/512B 对齐。

## 新增条目（P2 pilot + 第1批 + 剩余）

- [Cube-Vector 场景行宽必须 512B 对齐](row-width-512b-align.md) — CT-10，matmul 与 CV 向量化 load 的 512B 行宽硬约束
- [Tiling 预算与布局驱动的块大小选择](tiling-budget-and-layout-driven-selection.md) — OPT-12，UB/对齐/布局三重约束下的 BLOCK 选择
- [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](block-ptr-vs-scalar-load.md) — CT-11，保留硬件 block transfer 避免 2–3x 性能倒退
- [寄存器移位复用滑动窗口，消除重复 HBM 读](register-shift-sliding-window.md) — OPT-22，小窗口序列处理寄存器复用
- [preload + extract_slice 消除循环内同一数据重复 HBM 读取](preload-extract-slice-eliminate-reread.md) — OPT-23，寄存器子块提取替代重复 load
- [重叠 T-tiling 保证跨 tile 滑动窗口连续性](overlapping-t-tiling.md) — CT-14，T 维度切分保留 W-1 重叠区


## 新增条目（raw experiences 导入，待 review）

- [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](exp_ascend_large_shape_block_001.md) — 大 shape 场景下缩小 grid 尺寸、让每个核处理连续一段数据以提升缓存命中率，收益 12%
- [Memory Access Optimization: kv tile reuse](exp_attn_bwd_hstu_001.md) — attention 场景下，Memory Access Optimization: kv tile reuse，收益 35%
- [Memory Access Optimization: vectorized offset generation](exp_attn_causal_mask_001.md) — attention 场景下，Memory Access Optimization: vectorized offset generation，收益 15%
- [Memory Access Optimization: fp8 value casting](exp_attn_fp8_ub_slice_001.md) — attention 场景下，Memory Access Optimization: fp8 value casting，收益 30%
- [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — attention 场景下，Memory Access Optimization: block pointer abstraction，收益 25%
- [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — attention 场景下，Memory Access Optimization: online softmax，收益 40%
- [Memory Access Optimization: sparse gather optimization](exp_attn_sparse_bwd_001.md) — attention 场景下，Memory Access Optimization: sparse gather optimization，收益 40%
- [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — misc 场景下，Memory Access Optimization: vectorized 3d memory access，收益 30%
- [Memory Access Optimization: flattened grid scheduling](exp_causal_conv1d_001.md) — conv 场景下，Memory Access Optimization: flattened grid scheduling，收益 35%
- [Memory Access Optimization: block ptr utilization](exp_causal_conv1d_bwd_001.md) — conv 场景下，Memory Access Optimization: block ptr utilization，收益 35%
- [Memory Access Optimization: sliding window register reuse](exp_causal_conv1d_prefill_002.md) — conv 场景下，Memory Access Optimization: sliding window register reuse，收益 消除 75% 的 token load，实测贡献 Prefill kernel 整体加速比中的最大份
- [Memory Access Optimization: compile time branching](exp_causal_conv1d_states_001.md) — conv 场景下，Memory Access Optimization: compile time branching，收益 25%
- [Memory Access Optimization: fused state update](exp_causal_conv1d_update_001.md) — conv 场景下，Memory Access Optimization: fused state update，收益 35%
- [Memory Access Optimization: make block ptr hardware acceleration](exp_causal_conv1d_update_003.md) — conv 场景下，Memory Access Optimization: make block ptr hardware acceleration，收益 block_ptr vs scalar load: 在 D≥2048 场景差异可达 2
- [Memory Access Optimization: overlapping t tiling](exp_causal_conv1d_update_004.md) — conv 场景下，Memory Access Optimization: overlapping t tiling，收益 减少约 (num_T_tiles - 1) × (W-1) × D_CHK_SIZE 次全局内存 load，典型场景节
- [Memory Access Optimization: constexpr strides](exp_causal_conv1d_varlen_opt_001.md) — conv 场景下，Memory Access Optimization: constexpr strides，收益 22%
- [Memory Access Optimization: chunked recurrence strategy](exp_chunk_gated_delta_001.md) — ssm 场景下，Memory Access Optimization: chunked recurrence strategy，收益 35%
- [Add算子块大小动态调整优化经验](exp_elem_add_001_20260526.md) — elementwise 场景下，Add算子块大小动态调整优化经验，收益 从1.7890x加速比提升至2.1456x
- [带 alpha 缩放的逐元素 add 访存与块大小优化](exp_elemwise_add_001_20260526.md) — elementwise 场景下，带 alpha 缩放的逐元素 add 访存与块大小优化，收益 1.67x（平均时延 2.50ms 降至 1.50ms）
- [Memory Access Optimization: 2D Tiling Strategy](exp_embedding_001.md) — embedding 场景下，Memory Access Optimization: 2D Tiling Strategy，收益 25%
- [Memory Access Optimization: tiling strategy](exp_flash_attn_001.md) — attention 场景下，Memory Access Optimization: tiling strategy，收益 200%
- [Memory Access Optimization: online softmax algorithm](exp_flashattn_fwd_001.md) — attention 场景下，Memory Access Optimization: online softmax algorithm，收益 200%
- [Memory Access Optimization: simd index select](exp_flashattn_gqa_001.md) — attention 场景下，Memory Access Optimization: simd index select，收益 45%
- [Memory Access Optimization: operator fusion](exp_fuse_tlx_001.md) — misc 场景下，Memory Access Optimization: operator fusion，收益 35%
- [Memory Access Optimization: operator fusion](exp_fused_add_layernorm_001.md) — normalization 场景下，Memory Access Optimization: operator fusion，收益 30%
- [长序列注意力分块计算 + 在线 Softmax（内存 O(S²)→O(S)）](exp_fused_attention_002.md) — 长序列注意力场景下，分块计算与在线 Softmax 避免物化完整评分矩阵，收益 50%
- [Memory Access Optimization: operator fusion](exp_fused_norm_rope_001.md) — normalization 场景下，Memory Access Optimization: operator fusion，收益 25%
- [Memory Access Optimization: fused multi-tensor split and reshape](exp_fused_qkvzba_split_001.md) — misc 场景下，Memory Access Optimization: fused multi-tensor split and reshape，收益 25%
- [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](exp_fused_rms_norm_001.md) — normalization 场景下，Memory Access Optimization: fused residual-add RMSNorm with in-place writeback，收益 35%
- [Memory Access Optimization: fused split and dual-norm single-pass load](exp_fused_split_qk_norm_001.md) — normalization 场景下，Memory Access Optimization: fused split and dual-norm single-pass load，收益 35%
- [Memory Access Optimization: vectorized indexing](exp_gather_001.md) — gather_scatter 场景下，Memory Access Optimization: vectorized indexing，收益 30%
- [Memory Access Optimization: large block processing](exp_gather_flat_001.md) — gather_scatter 场景下，Memory Access Optimization: large block processing，收益 25%
- [Memory Access Optimization: grid stride loop](exp_geglu_bwd_001.md) — activation 场景下，Memory Access Optimization: grid stride loop，收益 25%
- [Memory Access Optimization: fused add-RMSNorm with weight hoisting](exp_gemma_rmsnorm_001.md) — matmul 场景下，Memory Access Optimization: fused add-RMSNorm with weight hoisting，收益 35%
- [Memory Access Optimization: Tiled Two-Pass Reduction](exp_gn_backward_001.md) — normalization 场景下，Memory Access Optimization: Tiled Two-Pass Reduction，收益 35%
- [Memory Access Optimization: sub blocking strategy](exp_gn_wb_backward_001.md) — misc 场景下，Memory Access Optimization: sub blocking strategy，收益 25%
- [Memory Access Optimization: indirect load scalarization](exp_grouped_attn_stage1_001.md) — attention 场景下，Memory Access Optimization: indirect load scalarization，收益 35%
- [Memory Access Optimization: dynamic vectorization tuning](exp_groupnorm_001.md) — normalization 场景下，Memory Access Optimization: dynamic vectorization tuning，收益 25%
- [Memory Access Optimization: fused lse reduction](exp_grpo_bwd_001.md) — misc 场景下，Memory Access Optimization: fused lse reduction，收益 25%
- [Memory Access Optimization: operator fusion](exp_hstu_attn_fwd_001.md) — attention 场景下，Memory Access Optimization: operator fusion，收益 35%
- [Memory Access Optimization: block pointer utilization](exp_idx_copy_001.md) — embedding 场景下，Memory Access Optimization: block pointer utilization，收益 25%
- [内存访问优化：转置 load 行优先化与跨阶段复用](exp_kda_bwd_intra_memory_opt_001.md) — ssm 场景下，内存访问优化：转置 load 行优先化与跨阶段复用，收益 35%
- [Memory Access Optimization: adaptive tiling strategy](exp_layernorm_bwd_001.md) — normalization 场景下，Memory Access Optimization: adaptive tiling strategy，收益 25%
- [Memory Access Optimization: dynamic workload tiling](exp_layernorm_bwd_fused_001.md) — normalization 场景下，Memory Access Optimization: dynamic workload tiling，收益 35%
- [Memory Access Optimization: persistent kernel strategy](exp_layernorm_fwd_001.md) — normalization 场景下，Memory Access Optimization: persistent kernel strategy，收益 20%
- [Memory Access Optimization: loop invariant code motion](exp_lightning_attn_decode_001.md) — attention 场景下，Memory Access Optimization: loop invariant code motion，收益 30%
- [Memory Access Optimization: kernel fusion](exp_logits_penalty_fusion_001.md) — misc 场景下，Memory Access Optimization: kernel fusion，收益 35%
- [Memory Access Optimization: redundant access elimination](exp_masked_fill_001.md) — misc 场景下，Memory Access Optimization: redundant access elimination，收益 25%
- [访存优化：大尺寸 matmul 的 256 分块与 L0C 复用](exp_matmul_001.md) — matmul 场景下，访存优化：大尺寸 matmul 的 256 分块与 L0C 复用，收益 40%
- [切分优化：A、B 均不转置 matmul 的 512B 行宽对齐分块](exp_matmul_003.md) — matmul 场景下，切分优化：A、B 均不转置 matmul 的 512B 行宽对齐分块，收益 10%
- [切分优化：A 不转置、B 转置 matmul 的 K0=256 统一分块](exp_matmul_004.md) — matmul 场景下，切分优化：A 不转置、B 转置 matmul 的 K0=256 统一分块，收益 10%
- [切分优化：双转置矩阵乘的 512B 对齐分块](exp_matmul_005.md) — matmul 场景下，切分优化：双转置矩阵乘的 512B 对齐分块，收益 32%
- [Memory Access Optimization: paged kv cache](exp_mla_paged_attn_001.md) — attention 场景下，Memory Access Optimization: paged kv cache，收益 45%
- [Memory Access Optimization: register buffering](exp_moe_binned_scatter_001.md) — gather_scatter 场景下，Memory Access Optimization: register buffering，收益 30%
- [Memory Access Optimization: vectorized metadata loading](exp_moe_gather_001.md) — gather_scatter 场景下，Memory Access Optimization: vectorized metadata loading，收益 20%
- [Memory Access Optimization: software pipelining](exp_moe_scatter_001.md) — gather_scatter 场景下，Memory Access Optimization: software pipelining，收益 35%
- [Memory Access Optimization: loop invariant code motion licm](exp_outer_product_001.md) — misc 场景下，Memory Access Optimization: loop invariant code motion licm，收益 45%
- [Memory Access Optimization: buffered writes](exp_padded_gather_001.md) — gather_scatter 场景下，Memory Access Optimization: buffered writes，收益 30%
- [Memory Access Optimization: block ptr utilization](exp_paged_attn_decode_001.md) — attention 场景下，Memory Access Optimization: block ptr utilization，收益 30%
- [Memory Access Optimization: Paged Memory Access Strategy](exp_paged_attn_flash_001.md) — attention 场景下，Memory Access Optimization: Paged Memory Access Strategy，收益 35%
- [Memory Access Optimization: paged kv cache with gqa head tiling](exp_paged_gqa_001.md) — attention 场景下，Memory Access Optimization: paged kv cache with gqa head tiling，收益 35%
- [Memory Access Optimization: head vectorization](exp_paged_kv_cache_001.md) — attention 场景下，Memory Access Optimization: head vectorization，收益 40%
- [Memory Access Optimization: paged kv indirection](exp_paged_prefill_001.md) — attention 场景下，Memory Access Optimization: paged kv indirection，收益 35%
- [内存访问优化：2D 张量块指针访存（make_block_ptr）](exp_perf_002.md) — general_perf 场景下，内存访问优化：2D 张量块指针访存（make_block_ptr），收益 25%
- [内存访问优化：非连续张量连续化 + 一维遍历](exp_perf_003.md) — general_perf 场景下，内存访问优化：非连续张量连续化 + 一维遍历，收益 30%
- [内存访问优化：stride 参数一律传 tensor.stride() 原生值](exp_perf_004.md) — general_perf 场景下，内存访问优化：stride 参数一律传 tensor.stride() 原生值，收益 20%
- [内存访问优化：contiguous() 一次性拷贝替代 stride 访问](exp_perf_006.md) — general_perf 场景下，内存访问优化：contiguous() 一次性拷贝替代 stride 访问，收益 35%
- [内存初始化优化：输出张量创建优选 torch.empty_like](exp_perf_008.md) — general_perf 场景下，内存初始化优化：输出张量创建优选 torch.empty_like，收益 8%
- [冗余操作优化：同形 element-wise 算子免 reshape](exp_perf_009.md) — general_perf 场景下，冗余操作优化：同形 element-wise 算子免 reshape，收益 10%
- [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](exp_perf_011.md) — general_perf 场景下，内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check，收益 50%
- [内存访问优化：行优先连续加载 + tl.trans 寄存器转置](exp_perf_015.md) — general_perf 场景下，内存访问优化：行优先连续加载 + tl.trans 寄存器转置，收益 30%
- [内存访问优化：跨阶段寄存器复用消除重复 load](exp_perf_016.md) — general_perf 场景下，内存访问优化：跨阶段寄存器复用消除重复 load，收益 15%
- [Memory Access Optimization: grid stride scheduling](exp_rmsnorm_001.md) — normalization 场景下，Memory Access Optimization: grid stride scheduling，收益 15%
- [Memory Access Optimization: register reuse intermediate](exp_rmsnorm_add_001.md) — normalization 场景下，Memory Access Optimization: register reuse intermediate，收益 30%
- [Memory Access Optimization: no-tiling register reuse with conflict-free dW accumulation](exp_rmsnorm_backward_001.md) — normalization 场景下，小列宽（n_cols<2048）反向的整行向量化加载与无冲突 dW 累加，收益 25%
- [Memory Access Optimization: backward column tiling with conflict-free dW accumulation](exp_rmsnorm_backward_tiled_001.md) — normalization 场景下，Memory Access Optimization: backward column tiling with conflict-free dW accumulation，收益 25%
- [Memory Access Optimization: dynamic block sizing with fused backward+add](exp_rmsnorm_bwd_fused_001.md) — normalization 场景下，Memory Access Optimization: dynamic block sizing with fused backward+add，收益 25%
- [Memory Access Optimization: two-pass reduction for large columns](exp_rmsnorm_bwd_large_cols_001.md) — normalization 场景下，Memory Access Optimization: two-pass reduction for large columns，收益 35%
- [Memory Access Optimization: 2D vector loading for small hidden dims](exp_rmsnorm_fused_001.md) — normalization 场景下，Memory Access Optimization: 2D vector loading for small hidden dims，收益 25%
- [Memory Access Optimization: fusing add/norm/bias/quant into one kernel](exp_rmsnorm_fused_quant_001.md) — normalization 场景下，Memory Access Optimization: fusing add/norm/bias/quant into one kernel，收益 50%
- [Memory Access Optimization: 2D block tiling with partial-sum reduction](exp_rmsnorm_grad_dw_001.md) — normalization 场景下，Memory Access Optimization: 2D block tiling with partial-sum reduction，收益 35%
- [Memory Access Optimization: single-pass tiling via UB residency](exp_rmsnorm_grad_dx_001.md) — normalization 场景下，Memory Access Optimization: single-pass tiling via UB residency，收益 40%
- [Memory Access Optimization: grid-stride scheduling with two-pass reduction](exp_rmsnorm_infer_001.md) — normalization 场景下，Memory Access Optimization: grid-stride scheduling with two-pass reduction，收益 25%
- [Memory Access Optimization: no-tiling single-load for small rows](exp_rmsnorm_notiling_001.md) — normalization 场景下，Memory Access Optimization: no-tiling single-load for small rows，收益 20%
- [Memory Access Optimization: column blocking for UB management](exp_rmsnorm_npu_001.md) — normalization 场景下，Memory Access Optimization: column blocking for UB management，收益 35%
- [Memory Access Optimization: column tiling for wide rows](exp_rmsnorm_tiled_001.md) — normalization 场景下，Memory Access Optimization: column tiling for wide rows，收益 25%
- [Memory Access Optimization: dynamic specialization with aligned vector load](exp_rope_forward_001.md) — rope 场景下，Memory Access Optimization: dynamic specialization with aligned vector load，收益 25%
- [Memory Access Optimization: cos sin reuse](exp_rope_npu_001.md) — 旋转位置编码场景下，Memory Access Optimization: cos sin reuse，收益 35%
- [Memory Access Optimization: vectorized memory access](exp_scatter_001.md) — gather_scatter 场景下，Memory Access Optimization: vectorized memory access，收益 30%
- [Memory Access Optimization: sorted bin grouping with padded alignment](exp_scatter_moe_padded_001.md) — gather_scatter 场景下，Memory Access Optimization: sorted bin grouping with padded alignment，收益 30%
- [Memory Access Optimization: binning strategy](exp_scatter_npu_001.md) — gather_scatter 场景下，Memory Access Optimization: binning strategy，收益 35%
- [Memory Access Optimization: padded binning for grouped scatter reduction](exp_scatter_wgrad_moe_001.md) — gather_scatter 场景下，Memory Access Optimization: padded binning for grouped scatter reduction，收益 30%
- [Memory Access Optimization: multibuffer software pipelining](exp_sdpa_bwd_reduce_001.md) — attention 场景下，Memory Access Optimization: multibuffer software pipelining，收益 25%
- [Memory Access Optimization: score-tile reuse with online softmax](exp_sdpa_flash_001.md) — attention 场景下，Memory Access Optimization: score-tile reuse with online softmax，收益 200%
- [Memory Access Optimization: block ptr abstraction](exp_sdpa_infer_001.md) — attention 场景下，Memory Access Optimization: block ptr abstraction，收益 150%
- [Memory Access Optimization: precision control](exp_skip_layernorm_001.md) — normalization 场景下，Memory Access Optimization: precision control，收益 30%
- [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_001.md) — softmax 场景下，Memory Access Optimization: two pass reduction strategy，收益 30%
- [Memory Access Optimization: two pass reduction strategy](exp_softmax_backward_non_inner_001.md) — softmax 场景下，Memory Access Optimization: two pass reduction strategy，收益 30%
- [Memory Access Optimization: preload extract slice eliminate hbm read](exp_solve_tril_002.md) — solve_tril 场景下，Memory Access Optimization: preload extract slice eliminate hbm read，收益 1.01x-1.37x（BT=16, V4 vs V1）；1.25
- [Memory Access Optimization: sparse iteration logic](exp_sparse_attn_backward_dq_001.md) — attention 场景下，Memory Access Optimization: sparse iteration logic，收益 40%
- [Memory Access Optimization: intermediate data residency](exp_spec_dec_reject_001.md) — misc 场景下，Memory Access Optimization: intermediate data residency，收益 60%
- [Memory Access Optimization: vectorized comparison](exp_spec_dec_verify_001.md) — misc 场景下，Memory Access Optimization: vectorized comparison，收益 50%
- [Memory Access Optimization: vectorized chunking](exp_speculative_recovery_001.md) — misc 场景下，Memory Access Optimization: vectorized chunking，收益 30%
- [Memory Access Optimization: block size tuning](exp_sum_scale_001.md) — reduction 场景下，Memory Access Optimization: block size tuning，收益 30%
- [Memory Access Optimization: grid stride loop](exp_swiglu_flat_001.md) — activation 场景下，Memory Access Optimization: grid stride loop，收益 25%
- [内存初始化优化：输出张量创建 API 选择](exp_tensor_create_001.md) — general_ascend 场景下，内存初始化优化：输出张量创建 API 选择，收益 8%
- [Memory Access Optimization: block ptr utilization](exp_topk_pad_001.md) — misc 场景下，Memory Access Optimization: block ptr utilization，收益 25%
- [Memory Access Optimization: software pipelining](exp_tv_distance_001.md) — misc 场景下，Memory Access Optimization: software pipelining，收益 25%

> 其余条目由摄入流程增量合并产生。

## trajectory 实例（P3/P5 提取）

- [activation Memory Access 实践聚合](traj-activation-memory_access.md) — 2 条 activation 访存优化 experience 聚合
- [embedding Memory Access 实践聚合](traj-embedding-memory_access.md) — 2 条 embedding 访存优化 experience 聚合
- [gather_scatter Memory Access 实践聚合](traj-gather_scatter-memory_access.md) — 10 条 gather_scatter 访存优化 experience 聚合
- [general_ascend Memory Access 实践聚合](traj-general_ascend-memory_access.md) — 1 条 general_ascend 访存优化 experience 聚合
- [general_ascend Memory Init 实践聚合](traj-general_ascend-memory_init.md) — 1 条 general_ascend 内存初始化 experience 聚合
- [general_perf Memory Access 实践聚合](traj-general_perf-memory_access.md) — 7 条 general_perf 访存优化 experience 聚合
- [general_perf Memory Init 实践聚合](traj-general_perf-memory_init.md) — 1 条 general_perf 内存初始化 experience 聚合
- [matmul Memory Access 实践聚合](traj-matmul-memory_access.md) — 2 条 matmul 访存优化 experience 聚合
- [misc Memory Access 实践聚合](traj-misc-memory_access.md) — 13 条 misc 访存优化 experience 聚合
- [reduction Memory Access 实践聚合](traj-reduction-memory_access.md) — 1 条 reduction 访存优化 experience 聚合
- [rope Memory Access 实践聚合](traj-rope-memory_access.md) — 2 条 rope 访存优化 experience 聚合
- [softmax Memory Access 实践聚合](traj-softmax-memory_access.md) — 2 条 softmax 访存优化 experience 聚合
- [solve_tril Memory Access 实践聚合](traj-solve_tril-memory_access.md) — 1 条 solve_tril 访存优化 experience 聚合
- [ssm Memory Access 实践聚合](traj-ssm-memory_access.md) — 2 条 ssm 访存优化 experience 聚合

## 新增条目（GDN 三篇优化实践蒸馏 + staging 合入，2026-08-27）

- [layout (B,T,H,D)→(B,H,T,D) 消除跨 Head 离散访存](layout-bthd-to-bhtd.md) — 序列分块迭代类算子 GM 侧预转置，GDN 实测平均 +38%~79%
- [UB 打满策略：调大分块吃满 192KB，微超则外提 mask/exp](ub-fill-and-compute-hoist.md) — BV/BK 64→128；UB 微超不退小块，计算外提调用侧
- [device-side gather 消除 strided tile load](device-side-gather-strided-load.md) — 访存结构重组，数据物化连续 workspace（本例 89x）

## 新增条目（ApplyRotaryPosEmb P4 后端契约沉淀，2026-09-09）

- [Triton-Ascend strided-2/interleave 访存契约](triton-ascend-strided2-interleave-contract.md) — stride-2 配对访存三个正确性缺陷（静默错值/并发 store 污染/丢 mask），正解 = 连续 load + tl.split/tl.interleave 寄存器重排

## 新增实例（SelfAttention P4 互补 merge，2026-09-30）

- [Tiling 预算与布局驱动的块大小选择](tiling-budget-and-layout-driven-selection.md) — append 已知实例：SelfAttention K3 tile 阶梯优先保 BSK（大 HD 免 UB fallback 骤减 BSK）+ 编译降档回退 + fp32 离线逐 shape 表，opt_iter_2→9 0.5577→0.7151（+28.2%）
