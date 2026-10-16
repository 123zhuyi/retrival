---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: 内存初始化优化：输出张量创建优选 torch.empty_like
description: general_perf 场景下，内存初始化优化：输出张量创建优选 torch.empty_like，收益 8%
tags: [访存效率, general, general_perf]
confidence: provisional
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 内存初始化优化：输出张量创建优选 torch.empty_like

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3，任意 dtype，输出将被 kernel 全量覆写的场景，小张量（<1e5 元素）收益更明显；失效边界：输出确实需要零初始化语义（累加缓冲、kernel 只部分写入）时必须用 zeros_like，换 empty_like 会引入脏数据；超大张量时初始化时间占比小，约 8% 的收益被稀释）

### 原则
- **创建规范**：优先使用`torch.empty_like()`创建输出张量，默认生成连续内存布局，无初始化开销
- **避坑要点**：避免`torch.zeros()`/`torch.ones()`，此类API会产生额外的内存初始化开销（约5%-10%）
- **兼容性**：torch.empty_like()的连续特性可无缝衔接一维访问优化，无需二次转换

### 代码/骨架出处
- `Triton`:
```python
class OutputTensorCreateModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        # 优化1: 使用torch.empty_like创建输出（默认连续，无初始化开销）
        output_tensor = torch.empty_like(input_tensor)
        # 反例: 不推荐 - torch.zeros_like有初始化开销
        # output_tensor = torch.zeros_like(input_tensor)
        
        # 确保输入连续后一维访问
        input_contig = input_tensor.contiguous() if not input_tensor.is_contiguous() else input_tensor
        n_elements = input_contig.numel()
        grid = (triton.cdiv(n_elements, 512),)
        elementwise_opt_kernel[grid](input_contig, output_tensor, n_elements, blockSize=512)
        return output_tensor
```

**坏实践（反例）**：输出将被 kernel 全量覆写，仍用 zeros_like 创建
```python
output = torch.zeros_like(x)   # 先全量清零
kernel[grid](x, output, ...)   # 随后又被 kernel 逐元素覆写
```
后果：zeros 的全量初始化写纯属冗余，产生约 5%-10% 额外开销（小张量占比更高）。✅ 正解：改用 `torch.empty_like()`，无初始化开销、默认连续布局，可无缝衔接一维访问优化。

**已知实例**：general_perf §raw (`experiences/general_perf/exp_perf_008.yaml`)
- 声称性能收益：8%
- 备注：适用于所有Triton算子的输出张量创建场景，`torch.empty_like()`相比`torch.zeros_like`可节省约8%的初始化时间，且默认连续的特性适配一维访问优化，910b2/910b3平台小张量（<1e5元素）的初始化开销占比更高

<!-- okf:related:start -->

# 相关

- 相关主题: [块大小选型：2 的幂次 128/256/512 起调，blockSize×dtypeBytes 对齐 32B 且不超 UB](../autotuning/exp_perf_001.md) — 主题相关：共享 general, general_perf
- 相关主题: [autotune 配置空间须覆盖 num_warps(2/4/8)×num_stages(2/3/4) 与 tile 组合](../autotuning/exp_perf_017.md) — 主题相关：共享 general, general_perf
- 相关主题: [性能避坑: 避免多维 stride 访问](../compute-pipeline/exp_perf_007.md) — 主题相关：共享 general, general_perf
- 相关主题: [算子拆分优化: 按计算阶段拆分为简单 kernel](../compute-pipeline/exp_perf_010.md) — 主题相关：共享 general_perf
- 相关主题: [标量消除: i32 比较转 fp32 走向量单元](../compute-pipeline/exp_perf_012.md) — 主题相关：共享 general, general_perf
- 相关主题: [标量消除: min() 运行时分支改 tl.where](../compute-pipeline/exp_perf_013.md) — 主题相关：共享 general, general_perf
- 相关主题: [标量消除: tl.cdiv 编译期循环边界](../compute-pipeline/exp_perf_014.md) — 主题相关：共享 general, general_perf
- 相关主题: [tl.dot vs torch.matmul 精度差异诊断](../meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [if/else 分支 dtype 不一致：改 masked load，bool 张量 host 侧转 int32](../meta-skills/exp_debug_triton_branch_type_mismatch_001.md) — 主题相关：共享 general
- 相关主题: [Ascend 后端禁用语法及替代写法（控制流/lambda/链式布尔/直接索引/负偏移）](../portability/exp_api_001.md) — 主题相关：共享 general
- 相关主题: [复杂嵌套条件下 tl.where 改 if-else 分支](../portability/exp_api_002.md) — 主题相关：共享 general
- 相关主题: [静态上限循环：while 改 for range（上限须为 tl.constexpr）](../portability/exp_api_003.md) — 主题相关：共享 general
- 相关主题: [动态上限循环：for + if + MAX_ITERS 编译期上界替代 while](../portability/exp_api_004.md) — 主题相关：共享 general
- 相关主题: [禁止用 tl.where 动态计算内存偏移，改 if-else 静态分支](../portability/exp_ascend_where_001.md) — 主题相关：共享 general
- 相关主题: [tl.constexpr 仅限内核参数声明，Host 侧禁用](../portability/exp_constexpr_001.md) — 主题相关：共享 general
- 相关主题: [禁用 Python 切片语法，改用 get_element/extract_slice/insert_slice](../portability/exp_slice_001.md) — 主题相关：共享 general
- 相关主题: [禁止对 tl.arange 索引表达式调用 get_element](../portability/exp_slice_002.md) — 主题相关：共享 general
- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](exp_ascend_large_shape_block_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: constexpr strides](exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 general, 访存效率
- 相关主题: [Add算子块大小动态调整优化经验](exp_elem_add_001_20260526.md) — 主题相关：共享 general, 访存效率
- 相关主题: [带 alpha 缩放的逐元素 add 访存与块大小优化](exp_elemwise_add_001_20260526.md) — 主题相关：共享 general, 访存效率
- 相关主题: [内存访问优化：2D 张量块指针访存（make_block_ptr）](exp_perf_002.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存访问优化：非连续张量连续化 + 一维遍历](exp_perf_003.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存访问优化：stride 参数一律传 tensor.stride() 原生值](exp_perf_004.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存访问优化：contiguous() 一次性拷贝替代 stride 访问](exp_perf_006.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [冗余操作优化：同形 element-wise 算子免 reshape](exp_perf_009.md) — 主题相关：共享 general_perf
- 相关主题: [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](exp_perf_011.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存访问优化：行优先连续加载 + tl.trans 寄存器转置](exp_perf_015.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存访问优化：跨阶段寄存器复用消除重复 load](exp_perf_016.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存初始化优化：输出张量创建 API 选择](exp_tensor_create_001.md) — 主题相关：共享 general, 访存效率

<!-- okf:related:end -->
