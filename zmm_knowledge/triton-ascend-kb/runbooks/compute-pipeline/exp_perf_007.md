---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: '性能避坑: 避免多维 stride 访问'
description: 'general_perf 场景下，性能避坑: 避免多维 stride 访问，收益 -40%（反例警示）'
tags: [AICore 单元利用/Scalar 削减, general, general_perf]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 性能避坑: 避免多维 stride 访问

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：非规则张量布局必须 stride 访问时无法规避；连续化（contiguous）本身有一次拷贝开销，小张量或一次性访问场景可能不划算；实测数据点来自 1024x1024 2D 张量，stride 访问耗时为连续一维访问的 1.4 倍以上）

### 原则
- **开销分析**：stride访问每次load/store都需计算多维偏移，累积开销随张量尺寸线性增长
- **访存缺陷**：stride访问易导致内存访问不连续，缓存命中率低（比连续访问低50%）
- **避坑策略**：除非必须（如非规则张量布局），否则完全避免直接stride访问，优先连续化后一维遍历

### 代码/骨架出处
- `Triton`:
```python
# 反例: 不推荐的stride访问（累积开销大）
@triton.jit
def stride_access_kernel(input_ptr, output_ptr, M, N, stride_m, stride_n, blockM: tl.constexpr=128, blockN: tl.constexpr=128):
    pid = triton.program_id(0)
    rm = pid // (N // blockN)
    rn = pid % (N // blockN)
    # 每次访问需计算stride偏移，额外开销大
    offsets_m = rm * blockM + tl.arange(0, blockM)[:, None]
    offsets_n = rn * blockN + tl.arange(0, blockN)[None, :]
    data_offsets = offsets_m * stride_m + offsets_n * stride_n
    mask = (offsets_m < M) & (offsets_n < N)
    data = tl.load(input_ptr + data_offsets, mask=mask)
    result = data * 0.5
    tl.store(output_ptr + data_offsets, result, mask=mask)

# 正例: 推荐的连续化+一维访问（替代stride）
def replace_stride_access(input_tensor):
    if not input_tensor.is_contiguous():
        input_tensor = input_tensor.contiguous()
    output = torch.empty_like(input_tensor)
    n_elements = input_tensor.numel()
    grid = (triton.cdiv(n_elements, 128),)
    elementwise_opt_kernel[grid](input_tensor, output, n_elements, blockSize=128)
    return output
```

**坏实践（反例）**：即本条主题——kernel 内对 2D 张量做 stride 访问，每次 load/store 都计算多维偏移且访存不连续 → 缓存命中率比连续访问低约 50%，实测 1024x1024 场景慢 1.4 倍以上（收益记为 -40%）→ ✅ 正解：先 `contiguous()` 连续化，再按一维遍历访问。

**已知实例**：general_perf §raw (`experiences/general_perf/exp_perf_007.yaml`)
- 声称性能收益：-40%
- 备注：负性能提升表示stride访问相比最优方案的性能损耗；910b2/910b3平台对stride访问的硬件惩罚显著，2D张量（1024x1024）的stride访问耗时是连续一维访问的1.4倍以上

<!-- okf:related:start -->

# 相关

- 相关主题: [块大小选型：2 的幂次 128/256/512 起调，blockSize×dtypeBytes 对齐 32B 且不超 UB](../autotuning/exp_perf_001.md) — 主题相关：共享 general, general_perf
- 相关主题: [autotune 配置空间须覆盖 num_warps(2/4/8)×num_stages(2/3/4) 与 tile 组合](../autotuning/exp_perf_017.md) — 主题相关：共享 general, general_perf
- 相关主题: [Memory Access Optimization: constexpr strides](../memory-sram/exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 general
- 相关主题: [Add算子块大小动态调整优化经验](../memory-sram/exp_elem_add_001_20260526.md) — 主题相关：共享 general
- 相关主题: [带 alpha 缩放的逐元素 add 访存与块大小优化](../memory-sram/exp_elemwise_add_001_20260526.md) — 主题相关：共享 general
- 相关主题: [内存访问优化：2D 张量块指针访存（make_block_ptr）](../memory-sram/exp_perf_002.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存访问优化：非连续张量连续化 + 一维遍历](../memory-sram/exp_perf_003.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存访问优化：stride 参数一律传 tensor.stride() 原生值](../memory-sram/exp_perf_004.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存访问优化：contiguous() 一次性拷贝替代 stride 访问](../memory-sram/exp_perf_006.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存初始化优化：输出张量创建优选 torch.empty_like](../memory-sram/exp_perf_008.md) — 主题相关：共享 general, general_perf
- 相关主题: [冗余操作优化：同形 element-wise 算子免 reshape](../memory-sram/exp_perf_009.md) — 主题相关：共享 general_perf
- 相关主题: [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](../memory-sram/exp_perf_011.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存访问优化：行优先连续加载 + tl.trans 寄存器转置](../memory-sram/exp_perf_015.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存访问优化：跨阶段寄存器复用消除重复 load](../memory-sram/exp_perf_016.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存初始化优化：输出张量创建 API 选择](../memory-sram/exp_tensor_create_001.md) — 主题相关：共享 general
- 相关主题: [if/else 分支 dtype 不一致：改 masked load，bool 张量 host 侧转 int32](../meta-skills/exp_debug_triton_branch_type_mismatch_001.md) — 主题相关：共享 general
- 相关主题: [Ascend 后端禁用语法及替代写法（控制流/lambda/链式布尔/直接索引/负偏移）](../portability/exp_api_001.md) — 主题相关：共享 general
- 相关主题: [复杂嵌套条件下 tl.where 改 if-else 分支](../portability/exp_api_002.md) — 主题相关：共享 general
- 相关主题: [静态上限循环：while 改 for range（上限须为 tl.constexpr）](../portability/exp_api_003.md) — 主题相关：共享 general
- 相关主题: [动态上限循环：for + if + MAX_ITERS 编译期上界替代 while](../portability/exp_api_004.md) — 主题相关：共享 general
- 相关主题: [禁止用 tl.where 动态计算内存偏移，改 if-else 静态分支](../portability/exp_ascend_where_001.md) — 主题相关：共享 general
- 相关主题: [tl.constexpr 仅限内核参数声明，Host 侧禁用](../portability/exp_constexpr_001.md) — 主题相关：共享 general
- 相关主题: [禁用 Python 切片语法，改用 get_element/extract_slice/insert_slice](../portability/exp_slice_001.md) — 主题相关：共享 general
- 相关主题: [禁止对 tl.arange 索引表达式调用 get_element](../portability/exp_slice_002.md) — 主题相关：共享 general
- 相关主题: [Compute Optimization: mixed precision compute](exp_attn_bwd_q_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: blocking strategy](exp_attn_qk_cube_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [算子拆分优化: 按计算阶段拆分为简单 kernel](exp_perf_010.md) — 主题相关：共享 general_perf
- 相关主题: [标量消除: i32 比较转 fp32 走向量单元](exp_perf_012.md) — 主题相关：共享 aicore 单元利用/scalar 削减, general, general_perf
- 相关主题: [标量消除: min() 运行时分支改 tl.where](exp_perf_013.md) — 主题相关：共享 aicore 单元利用/scalar 削减, general, general_perf
- 相关主题: [标量消除: tl.cdiv 编译期循环边界](exp_perf_014.md) — 主题相关：共享 aicore 单元利用/scalar 削减, general, general_perf

<!-- okf:related:end -->
