---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: portability
dimension: 4.3
title: 稀疏注意力 prefill：tle.index_select_simd 按需取 Top-K + cube1 加速 + 工作集 tile=64
description: 910b2/910b3 上稀疏注意力用 tle.index_select_simd 按预算索引只从 GM gather Top-K key（带宽大降），cube1 内建调 Cube 单元做 QK^T，在线 softmax LSE 分块更新，手工 workspace 按 CACHE_BLOCK=64 tile，tle.sub_vec_id 核内向量道并行，声称收益 60%
tags: [跨芯片适配, matrix-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 稀疏注意力 prefill：tle.index_select_simd 按需取 Top-K + cube1 加速 + 工作集 tile=64

【优化维度】：跨芯片适配
【适用条件】：平台 910b2/910b3 ｜ triton-ascend 版本 未标注（依赖 tle/cube1 专有内建）｜ dtype 混合精度（FP32 LSE 累加）｜ shape 长上下文 prefill，需预计算 Top-K 索引
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：tle.index_select_simd / cube1 / tle.sub_vec_id 为 Ascend 专有内建，GPU Triton 无对应物；依赖外部预计算的 Top-K 索引，索引缺失或质量差时稀疏假设不成立；稀疏度低（Top-K 接近全量）时 gather 开销抵消带宽收益；CACHE_BLOCK=64 按 UB 容量调优；910C/310P 未验证（unverified)）

### 原则
- **sparse indexing simd**：Utilizes tle.index_select_simd to gather only the Top-inputB keys from global memory based on the Indices_ptr. This significantly reduces memory bandwidth consumption compared to dense attention by avoiding the load of irrelevant key tokens.
- **cube unit acceleration**：Explicitly invokes the hardware Cube unit via the cube1 intrinsic for the inputA @ inputB^T matrix multiplication. This offloads the heavy compute workload from the vector unit to the dedicated matrix accelerator, maximizing throughput.
- **online softmax stability**：Implements the numerically stable online softmax algorithm using Log-Sum-Exp (LSE) logic. It processes the sequence in blocks (BLOCK_SBS), updating the LSE and accumulator incrementally, which prevents overflow/underflow and avoids materializing the full [S, S] attention matrix.
- **workspace tiling strategy**：Manually manages workspace buffers (Wksp_K_ptr, Wksp_qk_ptr, etc.) with a CACHE_BLOCK size of 64. This ensures that data loaded into the Unified Buffer (UB) is fully utilized before being discarded, optimizing data residency and minimizing HBM access latency.
- **intra core parallelism**：Leverages tle.sub_vec_id to distribute the processing of heads and blocks across the vector lanes or sub-cores within a single AI Core, ensuring high hardware utilization.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/199-sparse_flash_attention_prefill_kernel.py
```

**坏实践（反例）**：稀疏注意力仍按稠密方式加载全部 key 再掩码 → 无关 key token 的 GM 带宽照付，稀疏性完全浪费 → ✅ 正解即本条：`tle.index_select_simd` 按 Top-K 索引只 gather 需要的 key，QK^T 用 cube1 内建派给 Cube 单元，LSE 在线累加避免物化完整注意力矩阵

**已知实例**：attention §raw (`experiences/attention/exp_sparse_flash_attn_prefill_001.yaml`)
- 声称性能收益：60%
- 备注：This kernel is highly optimized for the Ascend 910B series architecture using proprietary intrinsics (tle, cube1). It requires pre-computed Top-inputB indices and is designed for the prefill stage of long-context LLMs.

<!-- okf:related:start -->

# 相关

- 实践案例: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](../../ops/fused-infer-attention-score.md) — 主题相关：共享 attention
- 实践案例: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../../ops/mla-prolog-v3.md) — 主题相关：共享 attention
- 实践案例: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](../../ops/moe-gating-top-k-softmax-v2.md) — 同属优化知识，主题相关
- 实践案例: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](../../ops/moe-gating-top-k-softmax.md) — 同属优化知识，主题相关
- 实践案例: [PromptFlashAttention inference/prefill on Ascend NPU](../../ops/prompt-flash-attention.md) — 主题相关：共享 attention
- 相关主题: [Triton API Quick Reference for Ascend Development](../../reference/triton-api-cheatsheet.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: dynamic autotuning](../compute-pipeline/exp_addmm_ascend_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: dual task pipelining](../compute-pipeline/exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: mixed precision compute](../compute-pipeline/exp_attn_bwd_q_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: blocking strategy](../compute-pipeline/exp_attn_qk_cube_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](exp_ascend_blockptr_boundary_scalar_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [核心数在 __init__ 获取一次，按算子类型选 VEC/CUBE 核数](exp_ascend_core_num_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [大 shape 用交错循环：grid 固定为核心数，kernel 内按步长遍历](exp_ascend_grid_large_shape_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [地址计算全程保持 int32，避免 int64 标量降级](exp_ascend_int64_address_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [整数比较标量降级：比较操作数转 fp32 或编译期消除](exp_ascend_int_compare_scalar_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [并行任务分配首选 for i in range(pid, total, core_num) 交错模式](exp_ascend_parallel_pattern_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [BLOCK_SIZE <65536 且子块内存 ≤L1 缓存，超限按维度多次嵌套切分](exp_ascend_split_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [Cube/Vector 流水线编排 + 序列分半（down kernel）适配片上内存](exp_attn_bwd_q_down_001.md) — 主题相关：共享 attention, matrix-operations, 跨芯片适配
- 相关主题: [Ascend 专用 tiling 配置：tile_mix cube/vector loop + multibuffer + HIVM CV 自动均衡](exp_attn_sda_001.md) — 主题相关：共享 attention, matrix-operations, 跨芯片适配
- 相关主题: [标签索引转 int32 比较 + GROUP_SIZE_M 分组调度 + 大 K tile](exp_entropy_backward_npu_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [反向 KV kernel：HIVM CV 均衡 + 双缓冲 + 非对称 tile（128×64）+ 序列分半](exp_flashattn_bwd_kv_ascend_001.md) — 主题相关：共享 attention, matrix-operations, 跨芯片适配
- 相关主题: [异构并行融合：MatMul 派 Cube、归一化+位置编码派 Vector，dl.async_task 异步掩盖](exp_fused_matmul_rms_rope_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [GQA 分页 KV：tl.parallel bind_sub_block 双向量核并行 + KV 按组复用 + block_table 遍历](exp_gqa_paged_npu_001.md) — 主题相关：共享 attention, 跨芯片适配
- 相关主题: [MatMul 调度：对角遍历 grouped_launch + dot_pad_only_k hint + 寄存器原位累加](exp_matmul_ascend_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [波前调度 MatMul：对角遍历 + dot_pad_only_k + autotune tile 长宽比](exp_matmul_ascend_wavefront_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [UB tiling（BLOCK_Q×blockK 分块）+ tl.math.fma 复数乘法 + tl.interleave 交错布局](exp_rope_llama4_npu_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [dot_pad_only_k hint + 对角 block 调度 + batch/head 循环内置融合](exp_rope_matmul_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [反向 KV 梯度 kernel：HIVM CV 均衡 + tile_mix=2 + 双缓冲 + 256×64 tile + bf16 存储 fp32 累加](exp_sda_bwd_kv_ascend_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [反向 Q kernel：CV 均衡 + 双缓冲 + 代数掩码（(mask-1)*1e6）+ 64×128 三角遍历](exp_sda_bwd_q_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [前向 SDA kernel：CV 均衡 + 双缓冲 + bf16 乘 fp32 累加 + 64×256 tile 分相处理](exp_sda_fwd_ascend_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配

<!-- okf:related:end -->
