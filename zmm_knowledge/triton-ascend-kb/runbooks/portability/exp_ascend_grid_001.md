---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: portability
dimension: 4.2
title: Grid 必须是 ≤3 维 tuple 且各维乘积 ≤65535
description: 910b2/910b3 上 Triton kernel 的 grid 必须为 tuple、最多 3 维、各维乘积 ≤65535，违反直接编译/启动失败；优先用 1 维 grid (core_num,)
tags: [编译器约束/hints, 通用优化, general_ascend]
confidence: provisional
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Grid 必须是 ≤3 维 tuple 且各维乘积 ≤65535

【优化维度】：编译器约束/hints
【适用条件】：平台 910b2/910b3 ｜ triton-ascend 版本 未标注（按验证版本）｜ dtype 全部 ｜ shape 任意（nd）
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：并行度需求超过 65535 时不能靠扩大 grid 解决，需在 kernel 内加循环（grid-stride/持久化）摊还任务；该上限是 910b2/910b3 实测约束，GPU Triton 无此限制，迁移来的多维度大 grid 写法必须改造；910C/310P 未验证（unverified)）

### 原则
- **维度限制**：Grid必须是tuple类型，且最多支持3维（如`(x,)`、`(x, y)`、`(x, y, z)`），超出维度会触发编译错误
- **大小限制**：Grid各维度数值的乘积必须≤65535（即`x * y * z ≤ 65535`），超出会导致kernel启动失败
- **配置原则**：优先使用1维Grid（`(core_num,)`），简化逻辑且避免乘积超限

### 代码/骨架出处
- `Triton`:
```python
# ========== Grid设置规范示例 ==========
def grid_config_correct(input_tensor):
    M, N = input_tensor.shape
    output = torch.empty_like(input_tensor)
    VEC_CORE_NUM = 48  # Ascend 910B2默认VEC核心数
    
    # 正确：1维Grid（乘积=48 ≤65535）
    grid = (VEC_CORE_NUM,)
    
    # 正确：2维Grid（乘积=24*2=48 ≤65535）
    # grid = (24, 2)
    
    # 错误：Grid不是tuple类型
    # grid = VEC_CORE_NUM
    
    # 错误：4维Grid（超出3维限制）
    # grid = (12, 2, 2, 1)
    
    # 错误：乘积超限（256*256=65536 >65535）
    # grid = (256, 256)
    
    multi_split_kernel[grid](
        input_tensor, output, M, N,
        input_tensor.stride(0), input_tensor.stride(1),
        blockM=128, blockN=128, MAX_SPLITS=100
    )
    return output
```

**坏实践（反例）**：Grid设置违反规范会直接导致kernel编译/启动失败，910b2/910b3平台无容错机制，需严格遵循维度和大小限制

**已知实例**：general_ascend §raw (`experiences/general_ascend/exp_ascend_grid_001.yaml`)
- 声称性能收益：0%

<!-- okf:related:start -->

# 相关

- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](../memory-sram/exp_ascend_large_shape_block_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [内存初始化优化：输出张量创建 API 选择](../memory-sram/exp_tensor_create_001.md) — 主题相关：共享 general_ascend
- 相关主题: [内存访问三查：mask 防越界、stride 匹配布局、索引上限判断](../meta-skills/exp_debug_001.md) — 主题相关：共享 通用优化
- 相关主题: [控制流避坑：禁 return/break/continue，复杂条件分步 mask 组合](../meta-skills/exp_debug_002.md) — 主题相关：共享 通用优化
- 相关主题: [Grid/Block 配置校验：乘积 ≤65535、≤3 维 tuple、大 shape 交错循环](../meta-skills/exp_debug_003.md) — 主题相关：共享 通用优化
- 相关主题: [并发写入必须用原子操作且类型与目标内存严格匹配](../meta-skills/exp_debug_004.md) — 主题相关：共享 通用优化
- 相关主题: [切片与索引语法校验：禁 Python 切片、arange 结果禁 get_element](../meta-skills/exp_debug_005.md) — 主题相关：共享 通用优化
- 相关主题: [Ascend 后端禁用语法及替代写法（控制流/lambda/链式布尔/直接索引/负偏移）](exp_api_001.md) — 主题相关：共享 编译器约束/hints
- 相关主题: [复杂嵌套条件下 tl.where 改 if-else 分支](exp_api_002.md) — 主题相关：共享 编译器约束/hints
- 相关主题: [静态上限循环：while 改 for range（上限须为 tl.constexpr）](exp_api_003.md) — 主题相关：共享 编译器约束/hints
- 相关主题: [动态上限循环：for + if + MAX_ITERS 编译期上界替代 while](exp_api_004.md) — 主题相关：共享 编译器约束/hints
- 相关主题: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](exp_ascend_blockptr_boundary_scalar_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [移除确定性访存上的冗余 compile_hint（mayDiscretememaccess）](exp_ascend_compile_hint_001.md) — 主题相关：共享 general_ascend, 编译器约束/hints, 通用优化
- 相关主题: [核心数在 __init__ 获取一次，按算子类型选 VEC/CUBE 核数](exp_ascend_core_num_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [大 shape 用交错循环：grid 固定为核心数，kernel 内按步长遍历](exp_ascend_grid_large_shape_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [地址计算全程保持 int32，避免 int64 标量降级](exp_ascend_int64_address_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [整数比较标量降级：比较操作数转 fp32 或编译期消除](exp_ascend_int_compare_scalar_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [并行任务分配首选 for i in range(pid, total, core_num) 交错模式](exp_ascend_parallel_pattern_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [BLOCK_SIZE <65536 且子块内存 ≤L1 缓存，超限按维度多次嵌套切分](exp_ascend_split_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [禁止用 tl.where 动态计算内存偏移，改 if-else 静态分支](exp_ascend_where_001.md) — 主题相关：共享 general_ascend, 编译器约束/hints
- 相关主题: [tl.constexpr 仅限内核参数声明，Host 侧禁用](exp_constexpr_001.md) — 主题相关：共享 general_ascend, 编译器约束/hints
- 相关主题: [标量类型转换仅用 .to(type)，禁用类型构造函数](exp_scalar_cast_001.md) — 主题相关：共享 general_ascend, 编译器约束/hints, 通用优化
- 相关主题: [禁用 Python 切片语法，改用 get_element/extract_slice/insert_slice](exp_slice_001.md) — 主题相关：共享 general_ascend, 编译器约束/hints
- 相关主题: [禁止对 tl.arange 索引表达式调用 get_element](exp_slice_002.md) — 主题相关：共享 general_ascend, 编译器约束/hints
- 相关主题: [迁移 kernel 须按 CANN 版本审计 launch 参数（8.5.0 删 no-limit/sync_solver/num_stages）](exp_solve_tril_005.md) — 主题相关：共享 编译器约束/hints

<!-- okf:related:end -->
