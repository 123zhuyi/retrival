---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "内存初始化优化：输出张量创建 API 选择"
description: "general_ascend 场景下，内存初始化优化：输出张量创建 API 选择，收益 8%"
tags: [访存效率, general, general_ascend]
confidence: provisional
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 内存初始化优化：输出张量创建 API 选择

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3，全 dtype，输出张量将被 kernel 全量覆盖写入；失效边界：小张量（元素数 <1e6）时初始化开销占比低，收益不明显；输出张量存在 kernel 未覆盖区域时不能用 empty（残留未初始化脏数据），必须用 zeros/ones 兜底）

### 原则
- **优选API**：Host侧创建输出张量优先使用`torch.empty`/`torch.empty_like`，无初始化开销且默认连续内存布局
- **禁用API**：禁止使用`torch.zeros`/`torch.ones`创建输出张量，此类API会产生额外的内存初始化开销（约5%-10%）
- **逻辑适配**：torch.empty 创建的未初始化张量，可通过Triton kernel的store操作覆盖，不影响计算正确性

### 代码/骨架出处
- `Triton`:
```python
# ========== 输出张量创建规范示例 ==========
def tensor_create_correct(input_tensor):
    # ✅ 正确: 使用torch.empty_like（无初始化开销，默认连续）
    output = torch.empty_like(input_tensor)
    
    # 错误: 使用torch.zeros_like（额外初始化开销）
    # output = torch.zeros_like(input_tensor)
    
    # 错误: 使用torch.ones_like（额外初始化开销）
    # output = torch.ones_like(input_tensor)
    
    n_elements = input_tensor.numel()
    blockSize = 128
    grid = (triton.cdiv(n_elements, blockSize),)
    constexpr_correct_kernel[grid](input_tensor, output, n_elements, blockSize)
    return output

# 补充: 自定义形状用torch.empty
def tensor_empty_custom(shape, dtype=torch.float16):
    output = torch.empty(shape, dtype=dtype, device='ascend')  # 适配Ascend后端
    return output
```

**坏实践（反例）**：用 `torch.zeros`/`torch.ones` 创建即将被 kernel 全量覆盖的输出张量
```python
output = torch.zeros_like(input_tensor)    # 先整段清零写一遍
kernel[grid](input_tensor, output, ...)    # kernel store 又全量覆盖一遍
```
后果：zeros/ones 的初始化写入随即被 kernel 的 store 完全覆盖，白付一趟约 5%-10% 的内存初始化开销，大张量（元素数 ≥1e6）上浪费更显著。✅ 正解：改用 `torch.empty`/`torch.empty_like` 创建输出张量（无初始化开销、默认连续布局），由 Triton kernel 的 store 完成唯一一次写入，正确性不受影响。

**已知实例**：general_ascend §raw (`experiences/general_ascend/exp_tensor_create_001.yaml`)
- 声称性能收益：8%
- 备注：910b2/910b3平台下，`torch.empty`系列API相比`zeros/ones`可节省约8%的初始化时间，尤其对大张量（元素数≥1e6）效果更显著

<!-- okf:related:start -->

# 相关

- 相关主题: [块大小选型：2 的幂次 128/256/512 起调，blockSize×dtypeBytes 对齐 32B 且不超 UB](../autotuning/exp_perf_001.md) — 主题相关：共享 general
- 相关主题: [autotune 配置空间须覆盖 num_warps(2/4/8)×num_stages(2/3/4) 与 tile 组合](../autotuning/exp_perf_017.md) — 主题相关：共享 general
- 相关主题: [性能避坑: 避免多维 stride 访问](../compute-pipeline/exp_perf_007.md) — 主题相关：共享 general
- 相关主题: [标量消除: i32 比较转 fp32 走向量单元](../compute-pipeline/exp_perf_012.md) — 主题相关：共享 general
- 相关主题: [标量消除: min() 运行时分支改 tl.where](../compute-pipeline/exp_perf_013.md) — 主题相关：共享 general
- 相关主题: [标量消除: tl.cdiv 编译期循环边界](../compute-pipeline/exp_perf_014.md) — 主题相关：共享 general
- 相关主题: [Ascend 后端禁用语法及替代写法（控制流/lambda/链式布尔/直接索引/负偏移）](../portability/exp_api_001.md) — 主题相关：共享 general
- 相关主题: [动态上限循环：for + if + MAX_ITERS 编译期上界替代 while](../portability/exp_api_004.md) — 主题相关：共享 general
- 相关主题: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](../portability/exp_ascend_blockptr_boundary_scalar_001.md) — 主题相关：共享 general_ascend
- 相关主题: [移除确定性访存上的冗余 compile_hint（mayDiscretememaccess）](../portability/exp_ascend_compile_hint_001.md) — 主题相关：共享 general_ascend
- 相关主题: [核心数在 __init__ 获取一次，按算子类型选 VEC/CUBE 核数](../portability/exp_ascend_core_num_001.md) — 主题相关：共享 general_ascend
- 相关主题: [Grid 必须是 ≤3 维 tuple 且各维乘积 ≤65535](../portability/exp_ascend_grid_001.md) — 主题相关：共享 general_ascend
- 相关主题: [大 shape 用交错循环：grid 固定为核心数，kernel 内按步长遍历](../portability/exp_ascend_grid_large_shape_001.md) — 主题相关：共享 general_ascend
- 相关主题: [地址计算全程保持 int32，避免 int64 标量降级](../portability/exp_ascend_int64_address_001.md) — 主题相关：共享 general_ascend
- 相关主题: [整数比较标量降级：比较操作数转 fp32 或编译期消除](../portability/exp_ascend_int_compare_scalar_001.md) — 主题相关：共享 general_ascend
- 相关主题: [并行任务分配首选 for i in range(pid, total, core_num) 交错模式](../portability/exp_ascend_parallel_pattern_001.md) — 主题相关：共享 general_ascend
- 相关主题: [BLOCK_SIZE <65536 且子块内存 ≤L1 缓存，超限按维度多次嵌套切分](../portability/exp_ascend_split_001.md) — 主题相关：共享 general_ascend
- 相关主题: [禁止用 tl.where 动态计算内存偏移，改 if-else 静态分支](../portability/exp_ascend_where_001.md) — 主题相关：共享 general, general_ascend
- 相关主题: [tl.constexpr 仅限内核参数声明，Host 侧禁用](../portability/exp_constexpr_001.md) — 主题相关：共享 general, general_ascend
- 相关主题: [标量类型转换仅用 .to(type)，禁用类型构造函数](../portability/exp_scalar_cast_001.md) — 主题相关：共享 general_ascend
- 相关主题: [禁用 Python 切片语法，改用 get_element/extract_slice/insert_slice](../portability/exp_slice_001.md) — 主题相关：共享 general, general_ascend
- 相关主题: [禁止对 tl.arange 索引表达式调用 get_element](../portability/exp_slice_002.md) — 主题相关：共享 general, general_ascend
- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](exp_ascend_large_shape_block_001.md) — 主题相关：共享 general_ascend, 访存效率
- 相关主题: [Memory Access Optimization: constexpr strides](exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 general, 访存效率
- 相关主题: [Add算子块大小动态调整优化经验](exp_elem_add_001_20260526.md) — 主题相关：共享 general, 访存效率
- 相关主题: [带 alpha 缩放的逐元素 add 访存与块大小优化](exp_elemwise_add_001_20260526.md) — 主题相关：共享 general, 访存效率
- 相关主题: [内存访问优化：2D 张量块指针访存（make_block_ptr）](exp_perf_002.md) — 主题相关：共享 general, 访存效率
- 相关主题: [内存访问优化：非连续张量连续化 + 一维遍历](exp_perf_003.md) — 主题相关：共享 general, 访存效率
- 相关主题: [内存访问优化：stride 参数一律传 tensor.stride() 原生值](exp_perf_004.md) — 主题相关：共享 general, 访存效率
- 相关主题: [内存访问优化：contiguous() 一次性拷贝替代 stride 访问](exp_perf_006.md) — 主题相关：共享 general, 访存效率
- 相关主题: [内存初始化优化：输出张量创建优选 torch.empty_like](exp_perf_008.md) — 主题相关：共享 general, 访存效率
- 相关主题: [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](exp_perf_011.md) — 主题相关：共享 general, 访存效率
- 相关主题: [内存访问优化：行优先连续加载 + tl.trans 寄存器转置](exp_perf_015.md) — 主题相关：共享 general, 访存效率
- 相关主题: [内存访问优化：跨阶段寄存器复用消除重复 load](exp_perf_016.md) — 主题相关：共享 general, 访存效率

<!-- okf:related:end -->
