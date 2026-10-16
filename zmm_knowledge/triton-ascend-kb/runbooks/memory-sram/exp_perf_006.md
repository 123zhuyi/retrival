---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: 内存访问优化：contiguous() 一次性拷贝替代 stride 访问
description: general_perf 场景下，内存访问优化：contiguous() 一次性拷贝替代 stride 访问，收益 35%
tags: [访存效率, general, general_perf]
confidence: provisional
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 内存访问优化：contiguous() 一次性拷贝替代 stride 访问

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3，任意 dtype，n_elements ≥128 的非连续张量 element-wise 场景；失效边界：张量已连续时转换无意义；小张量上一次性拷贝占比升高，收益需实测；stride 访问为算法必需（滑窗、转置读取等）的场景不适用）

### 原则
- **成本权衡**：.contiguous()仅产生一次性内存拷贝开销，远低于多次stride访问的累积计算/访存开销
- **访问效率**：连续内存的一维访问缓存友好，命中率比stride访问提升40%以上
- **核心策略**：非连续张量优先完成`contiguous()`转换，再通过一维指针遍历，整体性能最优

### 代码/骨架出处
- `Triton`:
```python
class ContiguousOptModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor, blockSize=1024):
        # 核心优化: 非连续张量先转换（一次性开销）
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()  # 转为连续内存布局
        
        # 一维访问: 无stride计算开销，缓存友好
        output_tensor = torch.empty_like(input_tensor)
        n_elements = input_tensor.numel()
        grid = (triton.cdiv(n_elements, blockSize),)
        elementwise_opt_kernel[grid](input_tensor, output_tensor, n_elements, blockSize)
        return output_tensor

@triton.jit
def elementwise_opt_kernel(input_ptr, output_ptr, n_elements, blockSize: tl.constexpr):
    pid = triton.program_id(0)
    offsets = pid * blockSize + tl.arange(0, blockSize)
    mask = offsets < n_elements
    # 一维访问连续内存，无额外偏移计算
    data = tl.load(input_ptr + offsets, mask=mask)
    result = tl.nn.silu(data)  # 示例元素级计算
    tl.store(output_ptr + offsets, result, mask=mask)
```

**坏实践（反例）**：不做连续化，kernel 内直接 stride 访问非连续张量
```python
# 输入为非连续视图，逐元素 stride 偏移
offs = offs_m * stride_m + offs_n * stride_n
data = tl.load(x_ptr + offs, mask=mask)
out = tl.nn.silu(data)
```
后果：每次 load 附带 stride 计算，缓存命中率比连续访问低 40% 以上。✅ 正解：host 端一次 `.contiguous()` 转连续布局，kernel 改一维遍历；即使计入拷贝开销，整体仍比 stride 访问高 30% 以上。

**已知实例**：general_perf §raw (`experiences/general_perf/exp_perf_006.yaml`)
- 声称性能收益：35%
- 备注：适用于所有非连续张量的元素级运算场景，910b2/910b3平台下，即使包含`contiguous()`的一次性拷贝，整体性能仍比直接stride访问高30%以上

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
- 相关主题: [Memory Access Optimization: constexpr strides](exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 general, 访存效率
- 相关主题: [Add算子块大小动态调整优化经验](exp_elem_add_001_20260526.md) — 主题相关：共享 general, 访存效率
- 相关主题: [带 alpha 缩放的逐元素 add 访存与块大小优化](exp_elemwise_add_001_20260526.md) — 主题相关：共享 general, 访存效率
- 相关主题: [内存访问优化：2D 张量块指针访存（make_block_ptr）](exp_perf_002.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存访问优化：非连续张量连续化 + 一维遍历](exp_perf_003.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存访问优化：stride 参数一律传 tensor.stride() 原生值](exp_perf_004.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存初始化优化：输出张量创建优选 torch.empty_like](exp_perf_008.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [冗余操作优化：同形 element-wise 算子免 reshape](exp_perf_009.md) — 主题相关：共享 general_perf
- 相关主题: [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](exp_perf_011.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存访问优化：行优先连续加载 + tl.trans 寄存器转置](exp_perf_015.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存访问优化：跨阶段寄存器复用消除重复 load](exp_perf_016.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存初始化优化：输出张量创建 API 选择](exp_tensor_create_001.md) — 主题相关：共享 general, 访存效率
- 相关主题: [输入进入 kernel 前必须 contiguous](input-contiguous.md) — 同属优化知识，主题相关
- 相关主题: [目标维度必须归一化到末维（permute->contiguous->处理->permute 回）](normalize-dim-to-last.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
