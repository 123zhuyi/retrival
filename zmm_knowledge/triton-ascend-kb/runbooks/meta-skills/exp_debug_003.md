---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: meta-skills
dimension: 6.2
title: Grid/Block 配置校验：乘积 ≤65535、≤3 维 tuple、大 shape 交错循环
description: 910b2/910b3/910b4 上 kernel 启动失败先查 grid：各维乘积须 ≤65535、必须为 ≤3 维 tuple；大 shape 算子用 for i in range(pid, total, core_num) 交错循环规避超限
tags: [编译/精度修复链, 通用优化, general_debug]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Grid/Block 配置校验：乘积 ≤65535、≤3 维 tuple、大 shape 交错循环

【优化维度】：编译/精度修复链
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 910b4；失效边界：65535 上限与 ≤3 维限制是 910b 系列实测约束，GPU Triton 无此限制；与 portability 的 exp_ascend_grid_001（约束规则）、exp_ascend_grid_large_shape_001（交错循环方案）同源，本卡是启动失败时的诊断入口；触发条件：kernel launch 失败或报 grid 相关错误时执行本卡检查）

### 原则
- **Grid大小限制**：Grid各维度乘积必须≤65535，超出会导致kernel启动失败
- **大shape处理**：大shape算子必须采用交错循环`for i in range(pid, total, core_num)`处理，规避Grid超限
- **Grid维度规范**：Grid必须是tuple类型且维度数≤3（如`(x,)`、`(x,y)`、`(x,y,z)`）

### 代码/骨架出处
- `Triton`:
```python
# ========== Grid/Block配置正确/错误示例 ==========
def grid_block_correct(input_tensor):
    M, N = input_tensor.shape
    output = torch.empty_like(input_tensor)
    CORE_NUM = 48
    blockSize = 256
    
    # 正确: Grid总大小≤65535
    grid = (CORE_NUM,)
    # 正确: 大shape用交错循环处理
    @triton.jit
def large_shape_kernel(input_ptr, output_ptr, M, N, CORE_NUM: tl.constexpr):
    pid = tl.program_id(0)
    for row_idx in range(pid, M, CORE_NUM):  # 交错循环
        # 处理逻辑...
    
    # 错误: Grid总大小超限（256*256=65536>65535）
    # grid = (256, 256)
    # 错误: 大shape未用交错循环
    # grid = (triton.cdiv(M*N, blockSize),)
    # 错误: Grid非tuple类型或维度超3
    # grid = CORE_NUM
    # grid = (10, 10, 10, 10)
```

**坏实践（反例）**：Grid配置错误是kernel启动失败的高频原因，910b系列平台对Grid维度和大小限制无容错机制

**已知实例**：general_debug §raw (`experiences/general_debug/exp_debug_003.yaml`)
- 声称性能收益：0%
- 备注：Grid配置错误是kernel启动失败的高频原因，910b系列平台对Grid维度和大小限制无容错机制；大shape算子未用交错循环会直接触发Grid超限报错

<!-- okf:related:start -->

# 相关

- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](../memory-sram/exp_ascend_large_shape_block_001.md) — 主题相关：共享 通用优化
- 相关主题: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](../portability/exp_ascend_blockptr_boundary_scalar_001.md) — 主题相关：共享 通用优化
- 相关主题: [移除确定性访存上的冗余 compile_hint（mayDiscretememaccess）](../portability/exp_ascend_compile_hint_001.md) — 主题相关：共享 通用优化
- 相关主题: [核心数在 __init__ 获取一次，按算子类型选 VEC/CUBE 核数](../portability/exp_ascend_core_num_001.md) — 主题相关：共享 通用优化
- 相关主题: [Grid 必须是 ≤3 维 tuple 且各维乘积 ≤65535](../portability/exp_ascend_grid_001.md) — 主题相关：共享 通用优化
- 相关主题: [大 shape 用交错循环：grid 固定为核心数，kernel 内按步长遍历](../portability/exp_ascend_grid_large_shape_001.md) — 主题相关：共享 通用优化
- 相关主题: [地址计算全程保持 int32，避免 int64 标量降级](../portability/exp_ascend_int64_address_001.md) — 主题相关：共享 通用优化
- 相关主题: [整数比较标量降级：比较操作数转 fp32 或编译期消除](../portability/exp_ascend_int_compare_scalar_001.md) — 主题相关：共享 通用优化
- 相关主题: [并行任务分配首选 for i in range(pid, total, core_num) 交错模式](../portability/exp_ascend_parallel_pattern_001.md) — 主题相关：共享 通用优化
- 相关主题: [BLOCK_SIZE <65536 且子块内存 ≤L1 缓存，超限按维度多次嵌套切分](../portability/exp_ascend_split_001.md) — 主题相关：共享 通用优化
- 相关主题: [标量类型转换仅用 .to(type)，禁用类型构造函数](../portability/exp_scalar_cast_001.md) — 主题相关：共享 通用优化
- 相关主题: [动态长度标量更新 kernel：do_not_specialize 防多版本编译，编译时间降 >80%](exp_bonus_renew_001.md) — 主题相关：共享 编译/精度修复链
- 相关主题: [内存访问三查：mask 防越界、stride 匹配布局、索引上限判断](exp_debug_001.md) — 主题相关：共享 general_debug, 编译/精度修复链, 通用优化
- 相关主题: [控制流避坑：禁 return/break/continue，复杂条件分步 mask 组合](exp_debug_002.md) — 主题相关：共享 general_debug, 编译/精度修复链, 通用优化
- 相关主题: [并发写入必须用原子操作且类型与目标内存严格匹配](exp_debug_004.md) — 主题相关：共享 general_debug, 编译/精度修复链, 通用优化
- 相关主题: [切片与索引语法校验：禁 Python 切片、arange 结果禁 get_element](exp_debug_005.md) — 主题相关：共享 general_debug, 编译/精度修复链, 通用优化
- 相关主题: [if/else 分支 dtype 不一致：改 masked load，bool 张量 host 侧转 int32](exp_debug_triton_branch_type_mismatch_001.md) — 主题相关：共享 general_debug, 编译/精度修复链
- 相关主题: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
- 相关主题: [形状驱动算子决策树](shape-driven-operator-decision-tree.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
