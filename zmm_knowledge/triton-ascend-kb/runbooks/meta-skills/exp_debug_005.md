---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: meta-skills
dimension: 6.2
title: 切片与索引语法校验：禁 Python 切片、arange 结果禁 get_element
description: 910b2/910b3/910b4 上切片相关编译失败先查：b[0]/b[i:j] 等 Python 切片须改 tl.get_element/tl.extract_slice/tl.insert_slice；tl.arange 编译期索引表达式禁调 get_element，直接数值计算
tags: [编译/精度修复链, 通用优化, general_debug]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 切片与索引语法校验：禁 Python 切片、arange 结果禁 get_element

【优化维度】：编译/精度修复链
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 910b4；失效边界：tl.load 得到的实际数据张量可正常用 get_element/extract_slice，仅 tl.arange 编译期表达式受限；与 portability 的 exp_slice_001/exp_slice_002 同源，本卡是切片类编译报错的诊断入口；触发条件：编译报错指向切片/索引/get_element 时执行本卡检查）

### 原则
- **切片语法规范**：禁止使用Python风格直接切片（如`b[0]`、`b[i:j]`），需用Triton专用切片API
- **arange索引避坑**：禁止对`tl.arange`生成的编译期索引张量使用`tl.get_element`，需直接数值计算
- **切片API使用**：单元素提取用`tl.get_element`，切片提取/插入用`tl.extract_slice`/`tl.insert_slice`

### 代码/骨架出处
- `Triton`:
```python
# ========== 切片与索引正确/错误示例 ==========
@triton.jit
def slice_index_correct_kernel(input_ptr, output_ptr, n_elements, blockSize: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * blockSize + tl.arange(0, blockSize)
    mask = offsets < n_elements
    tensor = tl.load(input_ptr + offsets, mask=mask)
    
    # 正确: 使用专用切片API
    element = tl.get_element(tensor, (0,))  # 单元素提取
    sub_tensor = tl.extract_slice(tensor, [0], [32], [1])  # 切片提取
    
    # 错误: Python风格直接切片
    # element = tensor[0]
    # sub_tensor = tensor[0:32]
    # 错误: 对tl.arange张量用tl.get_element
    # idx = tl.get_element(offsets, [5])
    # 正确: 直接数值计算替代
    idx = pid * blockSize + 5
```

**坏实践（反例）**：Python风格切片语法在Triton中完全不支持，会直接编译失败

**已知实例**：general_debug §raw (`experiences/general_debug/exp_debug_005.yaml`)
- 声称性能收益：0%
- 备注：Python风格切片语法在Triton中完全不支持，会直接编译失败；`tl.arange`是编译期表达式而非实际张量，`tl.get_element`调用会触发语法错误

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
- 相关主题: [禁用 Python 切片语法，改用 get_element/extract_slice/insert_slice](../portability/exp_slice_001.md) — 同属优化知识，主题相关
- 相关主题: [禁止对 tl.arange 索引表达式调用 get_element](../portability/exp_slice_002.md) — 同属优化知识，主题相关
- 相关主题: [动态长度标量更新 kernel：do_not_specialize 防多版本编译，编译时间降 >80%](exp_bonus_renew_001.md) — 主题相关：共享 编译/精度修复链
- 相关主题: [内存访问三查：mask 防越界、stride 匹配布局、索引上限判断](exp_debug_001.md) — 主题相关：共享 general_debug, 编译/精度修复链, 通用优化
- 相关主题: [控制流避坑：禁 return/break/continue，复杂条件分步 mask 组合](exp_debug_002.md) — 主题相关：共享 general_debug, 编译/精度修复链, 通用优化
- 相关主题: [Grid/Block 配置校验：乘积 ≤65535、≤3 维 tuple、大 shape 交错循环](exp_debug_003.md) — 主题相关：共享 general_debug, 编译/精度修复链, 通用优化
- 相关主题: [并发写入必须用原子操作且类型与目标内存严格匹配](exp_debug_004.md) — 主题相关：共享 general_debug, 编译/精度修复链, 通用优化
- 相关主题: [if/else 分支 dtype 不一致：改 masked load，bool 张量 host 侧转 int32](exp_debug_triton_branch_type_mismatch_001.md) — 主题相关：共享 general_debug, 编译/精度修复链

<!-- okf:related:end -->
