---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: 内存访问优化：stride 参数一律传 tensor.stride() 原生值
description: general_perf 场景下，内存访问优化：stride 参数一律传 tensor.stride() 原生值，收益 20%
tags: [访存效率, general, general_perf]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 内存访问优化：stride 参数一律传 tensor.stride() 原生值

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3，任意 dtype/维度，多维手动指针访存的 kernel；失效边界：一维连续张量或已使用 make_block_ptr 的 kernel 不涉及手动 stride；输入布局本身非连续时，应先评估走「连续化+一维遍历」路径而非修正 stride）

### 原则
- **精准设计**：stride参数需严格匹配张量的内存布局（通过tensor.stride()验证），错误设置会导致非连续访存/数据错位
- **避坑要点**：避免随意设置stride值，优先使用张量原生stride；多维stride需按“高维到低维”顺序传递
- **性能影响**：错误的stride设置会导致访存效率下降50%以上，甚至出现计算结果错误

### 代码/骨架出处
- `Triton`:
```python
# 正确的stride使用示例
@triton.jit
def stride_opt_kernel(input_ptr, output_ptr, M, N, stride_m, stride_n, blockM: tl.constexpr=128, blockN: tl.constexpr=128):
    pid = triton.program_id(0)
    rm = pid // (N // blockN)
    rn = pid % (N // blockN)
    
    # 正确的stride计算: 基于原生stride，按维度顺序计算偏移
    offsets_m = rm * blockM + tl.arange(0, blockM)[:, None]
    offsets_n = rn * blockN + tl.arange(0, blockN)[None, :]
    # 验证: stride_m对应M维度步幅，stride_n对应N维度步幅
    input_offsets = offsets_m * stride_m + offsets_n * stride_n
    mask = (offsets_m < M) & (offsets_n < N)
    
    data = tl.load(input_ptr + input_offsets, mask=mask)
    result = data + 1.0
    tl.store(output_ptr + input_offsets, result, mask=mask)

# 调用时传递原生stride（核心正确操作）
def forward(input_tensor):
    M, N = input_tensor.shape
    output = torch.empty_like(input_tensor)
    grid = (triton.cdiv(M, 128) * triton.cdiv(N, 128),)
    # 传递tensor.stride()原生步幅，避免手动计算错误
    stride_opt_kernel[grid](input_tensor, output, M, N, input_tensor.stride(0), input_tensor.stride(1), blockM=128, blockN=128)
    return output
```

**坏实践（反例）**：按逻辑形状想当然硬编码 stride
```python
stride_m, stride_n = N, 1  # 假设行优先连续，未看实际布局
data = tl.load(x_ptr + offs_m * stride_m + offs_n * stride_n)
```
后果：张量实为切片/转置视图时数据错位，或造成非连续访存，访存效率下降 50% 以上，甚至计算结果错误。✅ 正解：调用侧传 `input_tensor.stride(0)/(1)` 原生值，多维 stride 按「高维到低维」顺序传递。

**已知实例**：general_perf §raw (`experiences/general_perf/exp_perf_004.yaml`)
- 声称性能收益：20%
- 备注：所有带多维内存访问的Triton kernel都需严格验证stride参数，优先传递tensor.stride()的原生值，避免手动计算；910b2/910b3平台对stride不连续的访存惩罚尤为严重

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
- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](exp_ascend_large_shape_block_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: constexpr strides](exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 general, 访存效率
- 相关主题: [Add算子块大小动态调整优化经验](exp_elem_add_001_20260526.md) — 主题相关：共享 general, 访存效率
- 相关主题: [带 alpha 缩放的逐元素 add 访存与块大小优化](exp_elemwise_add_001_20260526.md) — 主题相关：共享 general, 访存效率
- 相关主题: [Memory Access Optimization: fused multi-tensor split and reshape](exp_fused_qkvzba_split_001.md) — 主题相关：共享 访存效率
- 相关主题: [内存访问优化：2D 张量块指针访存（make_block_ptr）](exp_perf_002.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存访问优化：非连续张量连续化 + 一维遍历](exp_perf_003.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存访问优化：contiguous() 一次性拷贝替代 stride 访问](exp_perf_006.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存初始化优化：输出张量创建优选 torch.empty_like](exp_perf_008.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [冗余操作优化：同形 element-wise 算子免 reshape](exp_perf_009.md) — 主题相关：共享 general_perf
- 相关主题: [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](exp_perf_011.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存访问优化：行优先连续加载 + tl.trans 寄存器转置](exp_perf_015.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存访问优化：跨阶段寄存器复用消除重复 load](exp_perf_016.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存初始化优化：输出张量创建 API 选择](exp_tensor_create_001.md) — 主题相关：共享 general, 访存效率

<!-- okf:related:end -->
