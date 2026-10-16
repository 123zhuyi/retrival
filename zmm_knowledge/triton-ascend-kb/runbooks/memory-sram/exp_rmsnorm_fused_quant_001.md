---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: fusing add/norm/bias/quant into one kernel"
description: "normalization 场景下，Memory Access Optimization: fusing add/norm/bias/quant into one kernel，收益 50%"
tags: [访存效率, reduction-operations, normalization]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: fusing add/norm/bias/quant into one kernel

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3，fp16/bf16 输入、int8 量化输出，hidden_size >= 128 的推理归一化+量化场景；失效边界：hidden_size 过大、单行归一化结果无法整体驻留时，量化循环必须按 COL_BLOCK_SIZE 切片：COL_BLOCK_SIZE × dtype 字节数 × 驻留份数 ≤ UB 预算；无量化需求时应走 SCALE 条件关闭的纯加+归一化路径；需要反向传播的训练场景不适用）

### 原则
- **operator fusion**：The kernel fuses four distinct operations (Element-wise Add, RMSNorm, Bias Addition, and Affine Quantization) into a single kernel launch. This drastically reduces global memory access by keeping intermediate results (e.g., the sum of input and residual) in high-speed registers/SRAM rather than writing them back to HBM.
- **weight caching**：The normalization weights (`norm_weight_values`) are loaded outside the main batch processing loop (`for i in tl.range...`). This leverages data reuse, as the same weights are applied to every row in the batch, significantly reducing memory traffic for the weight tensor.
- **block tiling quantization**：The quantization logic employs a tiling strategy using `COL_BLOCK_SIZE`. By iterating over the hidden dimension in blocks (`range(0, hidden_size, COL_BLOCK_SIZE)`), the kernel efficiently manages register pressure and allows for vectorized loading of quantization parameters (`quant_scale_ptr`, `quant_offset_ptr`).
- **slice extraction**：The use of `tl.extract_slice` inside the quantization loop allows the kernel to operate on sub-sections of the normalized data without reloading the entire row, optimizing memory bandwidth usage during the per-channel quantization step.
- **precision management**：The variance calculation is explicitly upcast to `tl.float32` before summation and division to ensure numerical stability during the RMSNorm calculation, while the final output is cast to `tl.int8` with saturation to handle quantization clamping correctly.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/189-add_rmsnorm_bias_kernel.py
```

**坏实践（反例）**：加、归一化、bias、量化拆成 4 个独立 kernel
```python
S = add(X, R)        # kernel 1，S 写 GM
N = rmsnorm(S)       # kernel 2，读 S 写 N
B = N + bias         # kernel 3，读 N 写 B
Q = quant(B)         # kernel 4，读 B 写 Q
```
后果：每个中间结果各付一次 GM 写 + 读，外加 3 次多余 kernel launch。✅ 正解：四算子融合为单 kernel，中间结果驻留寄存器/SRAM；归一化权重在 batch 循环外只载一次；量化段用 `tl.extract_slice` 取子块避免整行重载（收益 50%）。

**已知实例**：normalization §raw (`experiences/normalization/exp_rmsnorm_fused_quant_001.yaml`)
- 声称性能收益：50%
- 备注：This kernel is specifically optimized for inference scenarios requiring post-fusion normalization and quantization（网络末端归一化与量化串联的推理场景）。The `SCALE` conditional allows it to function as a standard Add+RMSNorm kernel when quantization is not required.

<!-- okf:related:start -->

# 相关

- 实践案例: [Add（broadcast 逐元素加 + alpha）Ascend C → Triton 语义转化卡](../../ops/add-broadcast-alpha.md) — 同属优化知识，主题相关
- 实践案例: [Elementwise add with a tail mask](../../ops/elementwise-add.md) — 同属优化知识，主题相关
- 实践案例: [FusedBiasLeakyRelu（偏置+LeakyReLU+缩放融合）Ascend C → Triton 语义转化卡](../../ops/fused-bias-leaky-relu.md) — 同属优化知识，主题相关
- 实践案例: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](../../ops/kv-quant-sparse-flash-attention.md) — 同属优化知识，主题相关
- 实践案例: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../../ops/mla-preprocess.md) — 同属优化知识，主题相关
- 实践案例: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](../../ops/mla-prolog.md) — 同属优化知识，主题相关
- 实践案例: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](../../ops/moe-init-routing-quant-v2.md) — 同属优化知识，主题相关
- 实践案例: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](../../ops/moe-init-routing-quant.md) — 同属优化知识，主题相关
- 实践案例: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](../../ops/moe-init-routing-v3.md) — 同属优化知识，主题相关
- 实践案例: [PromptFlashAttention inference/prefill on Ascend NPU](../../ops/prompt-flash-attention.md) — 同属优化知识，主题相关
- 实践案例: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](../../ops/quant-lightning-indexer.md) — 同属优化知识，主题相关
- 实践案例: [RMSNorm-RoPE fused kernel on Ascend 910B1](../../ops/rmsnorm-rope.md) — 主题相关：共享 normalization
- 开发指南: [Elementwise 算子优化](../../reference/op-class-elementwise.md) — 同属优化知识，主题相关
- 相关主题: [跨 tile 归约用 tl.atomic_add 作为可靠原语](../algorithm-numerics/atomic-add-cross-tile-reduction.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: conv bias silu triple fusion](../compute-pipeline/exp_causal_conv1d_prefill_006.md) — 同属优化知识，主题相关
- 相关主题: [量化融合算子 int8 输出精度不一致的逐层下钻诊断链](../meta-skills/quant-fusion-int8-precision-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../meta-skills/quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
- 相关主题: [Add算子块大小动态调整优化经验](exp_elem_add_001_20260526.md) — 主题相关：共享 访存效率
- 相关主题: [带 alpha 缩放的逐元素 add 访存与块大小优化](exp_elemwise_add_001_20260526.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](exp_fused_rms_norm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](exp_fused_split_qk_norm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](exp_gemma_rmsnorm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: sub blocking strategy](exp_gn_wb_backward_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused lse reduction](exp_grpo_bwd_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic workload tiling](exp_layernorm_bwd_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: persistent kernel strategy](exp_layernorm_fwd_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: grid stride scheduling](exp_rmsnorm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: register reuse intermediate](exp_rmsnorm_add_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling register reuse with conflict-free dW accumulation](exp_rmsnorm_backward_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: backward column tiling with conflict-free dW accumulation](exp_rmsnorm_backward_tiled_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic block sizing with fused backward+add](exp_rmsnorm_bwd_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: two-pass reduction for large columns](exp_rmsnorm_bwd_large_cols_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D vector loading for small hidden dims](exp_rmsnorm_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D block tiling with partial-sum reduction](exp_rmsnorm_grad_dw_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling single-load for small rows](exp_rmsnorm_notiling_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column blocking for UB management](exp_rmsnorm_npu_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column tiling for wide rows](exp_rmsnorm_tiled_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: multibuffer software pipelining](exp_sdpa_bwd_reduce_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: precision control](exp_skip_layernorm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized chunking](exp_speculative_recovery_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: block size tuning](exp_sum_scale_001.md) — 主题相关：共享 reduction-operations, 访存效率

<!-- okf:related:end -->
