---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: 内存访问优化：2D 张量块指针访存（make_block_ptr）
description: general_perf 场景下，内存访问优化：2D 张量块指针访存（make_block_ptr），收益 25%
tags: [访存效率, general, general_perf]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 内存访问优化：2D 张量块指针访存（make_block_ptr）

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3，fp16/bf16/fp32，2D 张量且 M/N ≥128；失效边界：M 或 N 小于 block_shape（如 <128）时边界 tile 占比升高，boundary_check 的逐元素标量判断开销可能吞掉收益（此时参见手动指针+向量化 mask 的写法）；地址发散的 gather/scatter 等非规则访问模式不适用）

### 原则
- **API优选**：2D数据优先使用tl.make_block_ptr替代手动指针计算，自动优化内存合并访问
- **边界处理**：配合boundary_check参数，自动处理张量边界，避免越界访问同时保证访存连续性
- **粒度适配**：块指针的block_shape参数适配910b2/910b3的L1缓存粒度，提升缓存命中率

### 代码/骨架出处
- `Triton`:
```python
@triton.jit def two_d_data_opt_kernel(input_ptr, output_ptr, M, N, stride_m, stride_n, blockM: tl.constexpr=128, blockN: tl.constexpr=128):
    pid = tl.program_id(0)
    rm = pid // (N // blockN)
    rn = pid % (N // blockN)
    
    # 使用tl.make_block_ptr优化2D数据访问（核心优化点）
    block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(M, N),
        strides=(stride_m, stride_n),
        offsets=(rm * blockM, rn * blockN),
        block_shape=(blockM, blockN),
        order=(0, 1)
    )
    # 自动边界检查+内存合并加载
    data = tl.load(block_ptr, boundary_check=(0, 1))
    
    # 示例计算: 2D数据元素平方
    result = data * data
    
    # 构造输出块指针并存储
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(M, N),
        strides=(stride_m, stride_n),
        offsets=(rm * blockM, rn * blockN),
        block_shape=(blockM, blockN),
        order=(0, 1)
    )
    tl.store(output_block_ptr, result, boundary_check=(0, 1))
```

**坏实践（反例）**：2D 数据手动算指针、不做边界检查
```python
offs_m = rm * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
offs_n = rn * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
data = tl.load(x_ptr + offs_m * stride_m + offs_n * stride_n)  # 越界风险+无连续性提示
```
后果：边界 mask 全靠手写，偏移算错即越界/数据错位；编译器缺少连续访存提示，访存开销约多 20%。✅ 正解：`tl.make_block_ptr` + `boundary_check=(0, 1)`，由 API 自动处理边界并保持 GM→UB 连续访存，block_shape 对齐 L1 缓存粒度。

**已知实例**：general_perf §raw (`experiences/general_perf/exp_perf_002.yaml`)
- 声称性能收益：25%
- 备注：适用于所有2D张量（如矩阵、特征图）的访存场景，tl.make_block_ptr可自动优化内存合并，相比手动计算指针减少约20%的访存开销，910b2/910b3平台效果尤为显著

<!-- okf:related:start -->

# 相关

- 相关主题: [块大小选型：2 的幂次 128/256/512 起调，blockSize×dtypeBytes 对齐 32B 且不超 UB](../autotuning/exp_perf_001.md) — 主题相关：共享 general, general_perf
- 相关主题: [autotune 配置空间须覆盖 num_warps(2/4/8)×num_stages(2/3/4) 与 tile 组合](../autotuning/exp_perf_017.md) — 主题相关：共享 general, general_perf
- 相关主题: [性能避坑: 避免多维 stride 访问](../compute-pipeline/exp_perf_007.md) — 主题相关：共享 general, general_perf
- 相关主题: [算子拆分优化: 按计算阶段拆分为简单 kernel](../compute-pipeline/exp_perf_010.md) — 主题相关：共享 general_perf
- 相关主题: [标量消除: i32 比较转 fp32 走向量单元](../compute-pipeline/exp_perf_012.md) — 主题相关：共享 general, general_perf
- 相关主题: [标量消除: min() 运行时分支改 tl.where](../compute-pipeline/exp_perf_013.md) — 主题相关：共享 general, general_perf
- 相关主题: [标量消除: tl.cdiv 编译期循环边界](../compute-pipeline/exp_perf_014.md) — 主题相关：共享 general, general_perf
- 相关主题: [if/else 分支 dtype 不一致：改 masked load，bool 张量 host 侧转 int32](../meta-skills/exp_debug_triton_branch_type_mismatch_001.md) — 主题相关：共享 general
- 相关主题: [Ascend 后端禁用语法及替代写法（控制流/lambda/链式布尔/直接索引/负偏移）](../portability/exp_api_001.md) — 主题相关：共享 general
- 相关主题: [复杂嵌套条件下 tl.where 改 if-else 分支](../portability/exp_api_002.md) — 主题相关：共享 general
- 相关主题: [静态上限循环：while 改 for range（上限须为 tl.constexpr）](../portability/exp_api_003.md) — 主题相关：共享 general
- 相关主题: [动态上限循环：for + if + MAX_ITERS 编译期上界替代 while](../portability/exp_api_004.md) — 主题相关：共享 general
- 相关主题: [禁止用 tl.where 动态计算内存偏移，改 if-else 静态分支](../portability/exp_ascend_where_001.md) — 主题相关：共享 general
- 相关主题: [tl.constexpr 仅限内核参数声明，Host 侧禁用](../portability/exp_constexpr_001.md) — 主题相关：共享 general
- 相关主题: [禁用 Python 切片语法，改用 get_element/extract_slice/insert_slice](../portability/exp_slice_001.md) — 主题相关：共享 general
- 相关主题: [禁止对 tl.arange 索引表达式调用 get_element](../portability/exp_slice_002.md) — 主题相关：共享 general
- 相关主题: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](block-ptr-vs-scalar-load.md) — 同属优化知识，主题相关
- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](exp_ascend_large_shape_block_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: constexpr strides](exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 general, 访存效率
- 相关主题: [Add算子块大小动态调整优化经验](exp_elem_add_001_20260526.md) — 主题相关：共享 general, 访存效率
- 相关主题: [带 alpha 缩放的逐元素 add 访存与块大小优化](exp_elemwise_add_001_20260526.md) — 主题相关：共享 general, 访存效率
- 相关主题: [内存访问优化：非连续张量连续化 + 一维遍历](exp_perf_003.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存访问优化：stride 参数一律传 tensor.stride() 原生值](exp_perf_004.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存访问优化：contiguous() 一次性拷贝替代 stride 访问](exp_perf_006.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存初始化优化：输出张量创建优选 torch.empty_like](exp_perf_008.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [冗余操作优化：同形 element-wise 算子免 reshape](exp_perf_009.md) — 主题相关：共享 general_perf
- 相关主题: [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](exp_perf_011.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存访问优化：行优先连续加载 + tl.trans 寄存器转置](exp_perf_015.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存访问优化：跨阶段寄存器复用消除重复 load](exp_perf_016.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存初始化优化：输出张量创建 API 选择](exp_tensor_create_001.md) — 主题相关：共享 general, 访存效率

<!-- okf:related:end -->
