---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: '算子拆分优化: 按计算阶段拆分为简单 kernel'
description: 'general_perf 场景下，算子拆分优化: 按计算阶段拆分为简单 kernel，收益 25%'
tags: [多核并行与负载均衡, mix, general_perf]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 算子拆分优化: 按计算阶段拆分为简单 kernel

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：与本库融合类优化方向相反——拆分适用于单 kernel 逻辑过于复杂（编译/调度开销大）的场景；简单算子链拆分反而增加 launch 与中间张量 HBM 开销，需按复杂度权衡；数据点：拆分后单 kernel 编译时间缩短约 40%，调度并行度提升 25% 以上（备注自述））

### 原则
- **拆分原则**：复杂算子拆分为多个简单kernel，避免单个kernel逻辑过于复杂导致编译/调度开销大
- **拆分逻辑**：按计算阶段拆分（如“特征提取→归一化→激活”拆分为3个kernel），每个kernel专注单一功能
- **性能优势**：拆分后kernel更易适配硬件资源，调度延迟降低，并行度提升

### 代码/骨架出处
- `Triton`:
```python
# 复杂算子拆分示例: 原单个复杂kernel拆分为3个简单kernel
# 拆分前: 单个kernel包含卷积+归一化+激活（复杂）
# 拆分后: 3个独立简单kernel

# 简单kernel1: 卷积计算
@triton.jit
def conv_kernel(input_ptr, weight_ptr, output_ptr, B, H, W, C, K, blockSize: tl.constexpr=64):
    # 仅实现卷积计算，逻辑简单
    pid = triton.program_id(0)
    offsets = pid * blockSize + tl.arange(0, blockSize)
    mask = offsets < B*H*W*C
    data = tl.load(input_ptr + offsets, mask=mask)
    weight = tl.load(weight_ptr + tl.arange(0, K*K*C)[:K*K*C])
    conv_out = tl.dot(data, weight)
    tl.store(output_ptr + offsets, conv_out, mask=mask)

# 简单kernel2: 归一化计算
@triton.jit
def norm_kernel(input_ptr, output_ptr, n_elements, mean, std, blockSize: tl.constexpr=64):
    # 仅实现归一化，逻辑简单
    pid = triton.program_id(0)
    offsets = pid * blockSize + tl.arange(0, blockSize)
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets, mask=mask)
    norm_out = (data - mean) / (std + 1e-5)
    tl.store(output_ptr + offsets, norm_out, mask=mask)

# 简单kernel3: 激活计算
@triton.jit
def act_kernel(input_ptr, output_ptr, n_elements, blockSize: tl.constexpr=64):
    # 仅实现激活，逻辑简单
    pid = triton.program_id(0)
    offsets = pid * blockSize + tl.arange(0, blockSize)
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets, mask=mask)
    act_out = tl.nn.gelu(data)
    tl.store(output_ptr + offsets, act_out, mask=mask)

# 调用逻辑: 依次执行3个简单kernel
def forward_complex_op(input_tensor, weight):
    # 步骤1: 卷积
    conv_out = torch.empty_like(input_tensor)
    grid_conv = (triton.cdiv(input_tensor.numel(), 64),)
    conv_kernel[grid_conv](input_tensor, weight, conv_out, *input_tensor.shape, K=3, blockSize=64)
    
    # 步骤2: 归一化
    norm_out = torch.empty_like(conv_out)
    mean, std = conv_out.mean(), conv_out.std()
    grid_norm = (triton.cdiv(conv_out.numel(), 64),)
    norm_kernel[grid_norm](conv_out, norm_out, conv_out.numel(), mean, std, blockSize=64)
    
    # 步骤3: 激活
    final_out = torch.empty_like(norm_out)
    grid_act = (triton.cdiv(norm_out.numel(), 64),)
    act_kernel[grid_act](norm_out, final_out, norm_out.numel(), blockSize=64)
    
    return final_out
```

**坏实践（反例）**：单个巨型 kernel 塞入卷积+归一化+激活全部逻辑 → 编译慢、寄存器/UB 压力大、调度粒度粗 → ✅ 正解即本条：按计算阶段拆成 3 个简单 kernel，各自专注单一功能。注意与融合优化互为反面，选择依据是单 kernel 复杂度是否已压垮编译/调度。

**已知实例**：general_perf §raw (`experiences/general_perf/exp_perf_010.yaml`)
- 声称性能收益：25%
- 备注：适用于包含多阶段计算的复杂算子（如卷积+BN+激活、Attention全流程等），拆分后单个kernel编译时间缩短40%，910b2/910b3平台的调度并行度提升25%以上

<!-- okf:related:start -->

# 相关

- 相关主题: [块大小选型：2 的幂次 128/256/512 起调，blockSize×dtypeBytes 对齐 32B 且不超 UB](../autotuning/exp_perf_001.md) — 主题相关：共享 general_perf
- 相关主题: [autotune 配置空间须覆盖 num_warps(2/4/8)×num_stages(2/3/4) 与 tile 组合](../autotuning/exp_perf_017.md) — 主题相关：共享 general_perf
- 相关主题: [长序列注意力分块计算 + 在线 Softmax（内存 O(S²)→O(S)）](../memory-sram/exp_fused_attention_002.md) — 主题相关：共享 mix
- 相关主题: [内存访问优化：2D 张量块指针访存（make_block_ptr）](../memory-sram/exp_perf_002.md) — 主题相关：共享 general_perf
- 相关主题: [内存访问优化：非连续张量连续化 + 一维遍历](../memory-sram/exp_perf_003.md) — 主题相关：共享 general_perf
- 相关主题: [内存访问优化：stride 参数一律传 tensor.stride() 原生值](../memory-sram/exp_perf_004.md) — 主题相关：共享 general_perf
- 相关主题: [内存访问优化：contiguous() 一次性拷贝替代 stride 访问](../memory-sram/exp_perf_006.md) — 主题相关：共享 general_perf
- 相关主题: [内存初始化优化：输出张量创建优选 torch.empty_like](../memory-sram/exp_perf_008.md) — 主题相关：共享 general_perf
- 相关主题: [冗余操作优化：同形 element-wise 算子免 reshape](../memory-sram/exp_perf_009.md) — 主题相关：共享 general_perf
- 相关主题: [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](../memory-sram/exp_perf_011.md) — 主题相关：共享 general_perf
- 相关主题: [内存访问优化：行优先连续加载 + tl.trans 寄存器转置](../memory-sram/exp_perf_015.md) — 主题相关：共享 general_perf
- 相关主题: [内存访问优化：跨阶段寄存器复用消除重复 load](../memory-sram/exp_perf_016.md) — 主题相关：共享 general_perf
- 相关主题: [Scheduling Optimization: dynamic autotuning](exp_addmm_ascend_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dual task pipelining](exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grouped scheduling](exp_bmm_grouped_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: two dimensional grid strategy](exp_causal_conv1d_prefill_003.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: pad slot early exit](exp_causal_conv1d_prefill_005.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: bdt layout rationale](exp_causal_conv1d_update_007.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_element_mul_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: 2D Block Tiling](exp_embedding_backward_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: kernel fusion and recomputation](exp_entropy_backward_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: persistent kernel grid stride](exp_entropy_backward_d_weight_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_entropy_backward_fused_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: compilation specialization control](exp_expand_serial_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_flash_attn_gridstride_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [流程优化: 标准 score-归一化-加权三步流程](exp_fused_attention_001.md) — 主题相关：共享 mix, 多核并行与负载均衡
- 相关主题: [性能避坑: 避免多维 stride 访问](exp_perf_007.md) — 主题相关：共享 general_perf
- 相关主题: [标量消除: i32 比较转 fp32 走向量单元](exp_perf_012.md) — 主题相关：共享 general_perf
- 相关主题: [标量消除: min() 运行时分支改 tl.where](exp_perf_013.md) — 主题相关：共享 general_perf
- 相关主题: [标量消除: tl.cdiv 编译期循环边界](exp_perf_014.md) — 主题相关：共享 general_perf

<!-- okf:related:end -->
