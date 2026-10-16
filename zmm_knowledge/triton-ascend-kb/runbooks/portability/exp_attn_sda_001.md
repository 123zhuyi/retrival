---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: portability
dimension: 4.3
title: Ascend 专用 tiling 配置：tile_mix cube/vector loop + multibuffer + HIVM CV 自动均衡
description: 910b2/910b3/91093/91095 上用 tile_mix_cube_loop/tile_mix_vector_loop 显式映射计算循环到 Cube/Vector 单元，配 set_workspace_multibuffer 2 双缓冲与 enable_hivm_auto_cv_balance，在线 softmax 累加用 FP32、IO 保持 BF16，声称收益 35%
tags: [跨芯片适配, matrix-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Ascend 专用 tiling 配置：tile_mix cube/vector loop + multibuffer + HIVM CV 自动均衡

【优化维度】：跨芯片适配
【适用条件】：平台 910b2/910b3/91093/91095 ｜ triton-ascend 版本 未标注（依赖 CANN HIVM 扩展）｜ dtype BF16 输入输出 + FP32 累加 ｜ shape 长序列（split-decode，SEQ_LEN = S // 2）
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：tile_mix_*/set_workspace_multibuffer/enable_hivm_auto_cv_balance 均为 CANN 专用扩展，GPU Triton 与非 HIVM 后端的 triton-ascend 版本不可用；无 Cube/Vector 混合计算（纯 element-wise）的 kernel 上 CV 均衡无收益；split-decode 划分（S//2）只对自回归长序列场景有意义）

### 原则
- **hardware specific tiling**：Utilizes `tile_mix_cube_loop` and `tile_mix_vector_loop` configuration parameters to explicitly map the computation loops to the Ascend NPU's Cube (Matrix) and Vector units, ensuring the instruction mix matches the hardware's pipeline capabilities.
- **multibuffer pipeline**：Enables `multibuffer` with `set_workspace_multibuffer 2` to implement a software pipelining strategy. This overlaps global memory loads (inputB, inputC) with the compute operations of the current block, effectively hiding memory latency.
- **hivm auto balance**：Activates `enable_hivm_auto_cv_balance True` to leverage the compiler's internal heuristics for the Ascend architecture. This automatically balances the resource allocation and instruction dispatch between the Cube unit (for matrix multiplies in QK^T) and the Vector unit (for softmax and masking), preventing unit starvation.
- **mixed precision accumulation**：Performs the online softmax accumulation in FP32 (`HIGH_TYPE`) while keeping inputs and outputs in BF16 (`LOW_TYPE`). This maintains numerical stability for the Log-Sum-Exp calculations without increasing the bandwidth pressure for global memory access.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/260-kernel_sda_fwd_down.py
```

**坏实践（反例）**：在 Ascend 上沿用 GPU 式隐式调度（不配 tile_mix、不开 multibuffer）→ 计算循环未被映射到 Cube/Vector 单元、GM 加载延迟无法被计算掩盖，Cube 与 Vector 互相等待 → ✅ 正解即本条：显式 tile_mix_cube_loop/tile_mix_vector_loop + `set_workspace_multibuffer 2` + `enable_hivm_auto_cv_balance True`；注意 multibuffer 增大 workspace 占用，UB/工作区紧张时需权衡

**已知实例**：attention §raw (`experiences/attention/exp_attn_sda_001.yaml`)
- 声称性能收益：35%
- 备注：This kernel is specifically optimized for Ascend NPUs using CANN-specific Triton extensions (HIVM, Cube loops). It implements a split-decode attention pattern (SEQ_LEN = S // 2) suitable for autoregressive tasks.

<!-- okf:related:start -->

# 相关

- 相关主题: [Scheduling Optimization: dynamic autotuning](../compute-pipeline/exp_addmm_ascend_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: dual task pipelining](../compute-pipeline/exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: mixed precision compute](../compute-pipeline/exp_attn_bwd_q_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: blocking strategy](../compute-pipeline/exp_attn_qk_cube_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: attention sinks mechanism](../compute-pipeline/exp_attn_sinks_prefill_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: block ptr tiling](../compute-pipeline/exp_block_inverse_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: block ptr utilization](../compute-pipeline/exp_block_matmul_inv_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [multibuffer/unit_flag 等编译选项需实测验证，禁止默认开启](../compute-pipeline/multibuffer-empirical.md) — 同属优化知识，主题相关
- 相关主题: [算子融合模式与约束决策](../compute-pipeline/operator-fusion-patterns.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: multibuffer software pipelining](../memory-sram/exp_sdpa_bwd_reduce_001.md) — 主题相关：共享 attention
- 相关主题: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](exp_ascend_blockptr_boundary_scalar_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [核心数在 __init__ 获取一次，按算子类型选 VEC/CUBE 核数](exp_ascend_core_num_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [大 shape 用交错循环：grid 固定为核心数，kernel 内按步长遍历](exp_ascend_grid_large_shape_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [地址计算全程保持 int32，避免 int64 标量降级](exp_ascend_int64_address_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [整数比较标量降级：比较操作数转 fp32 或编译期消除](exp_ascend_int_compare_scalar_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [并行任务分配首选 for i in range(pid, total, core_num) 交错模式](exp_ascend_parallel_pattern_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [BLOCK_SIZE <65536 且子块内存 ≤L1 缓存，超限按维度多次嵌套切分](exp_ascend_split_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [Cube/Vector 流水线编排 + 序列分半（down kernel）适配片上内存](exp_attn_bwd_q_down_001.md) — 主题相关：共享 attention, matrix-operations, 跨芯片适配
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
- 相关主题: [稀疏注意力 prefill：tle.index_select_simd 按需取 Top-K + cube1 加速 + 工作集 tile=64](exp_sparse_flash_attn_prefill_001.md) — 主题相关：共享 attention, matrix-operations, 跨芯片适配

<!-- okf:related:end -->
