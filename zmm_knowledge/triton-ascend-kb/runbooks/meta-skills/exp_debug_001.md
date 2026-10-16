---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: meta-skills
dimension: 6.2
title: 内存访问三查：mask 防越界、stride 匹配布局、索引上限判断
description: 910b2/910b3/910b4 上 kernel 崩溃/数据错乱先查内存访问：所有 tl.load/tl.store 带 mask 或 boundary_check、stride 参数与张量实际布局一致、所有索引进 offset < n_elements 判断
tags: [编译/精度修复链, 通用优化, general_debug]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 内存访问三查：mask 防越界、stride 匹配布局、索引上限判断

【优化维度】：编译/精度修复链
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 910b4；失效边界：完整 tile 且对齐的场景可用 constexpr ALIGNED 分支省 mask（保留运行时开销与编译期确定性的权衡）；stride 检查只覆盖仿射连续布局，间接寻址（index 张量）需改查 index 本身的值域；触发条件：kernel 崩溃、设备异常或输出数据错乱时优先执行本卡三查）

### 原则
- **mask检查**：所有`tl.load/tl.store`操作必须配备mask或boundary_check，防止内存越界访问
- **stride检查**：确认stride参数与张量实际内存布局匹配，避免因stride错误导致的偏移计算偏差
- **索引检查**：所有数组索引需通过条件判断（如`offset < n_elements`）确保不越界

### 代码/骨架出处
- `Triton`:
```python
# ========== 内存访问正确/错误示例 ==========
@triton.jit
def memory_access_correct_kernel(input_ptr, output_ptr, n_elements, stride, blockSize: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * blockSize + tl.arange(0, blockSize)
    
    # 正确: 所有load/store带mask防越界
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets * stride, mask=mask)  # stride参数正确匹配
    tl.store(output_ptr + offsets * stride, data * 2, mask=mask)
    
    # 错误: 无mask的load/store（易越界）
    # data = tl.load(input_ptr + offsets * stride)
    # 错误: stride参数错误（如误用stride=2而非实际stride）
    # data = tl.load(input_ptr + offsets * 2, mask=mask)
    # 错误: 索引未检查（如offsets无上限判断）
    # offsets = pid * blockSize + tl.arange(0, blockSize + 100)
```

**坏实践（反例）**：stride参数错误会导致数据读取错乱，需与张量实际stride严格一致

**已知实例**：general_debug §raw (`experiences/general_debug/exp_debug_001.yaml`)
- 声称性能收益：0%
- 备注：内存访问问题是Triton kernel最常见的崩溃原因，910b系列平台下越界访问会直接触发设备异常；stride参数错误会导致数据读取错乱，需与张量实际stride严格一致

<!-- okf:related:start -->

# 相关

- 实践案例: [BlockSparseAttention (block-sparse mask) on Ascend NPU](../../ops/block-sparse-attention.md) — 同属优化知识，主题相关
- 实践案例: [Elementwise add with a tail mask](../../ops/elementwise-add.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: dim is multiple mask elision](../compute-pipeline/exp_causal_conv1d_prefill_007.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: dim is multiple mask elimination](../compute-pipeline/exp_causal_conv1d_update_005.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: redundant mask elimination by matrix property](../compute-pipeline/exp_solve_tril_001.md) — 同属优化知识，主题相关
- 相关主题: [整数比较/加法必须转 float32，避免 Vector 标量降级](../compute-pipeline/int-compare-cast-float32.md) — 同属优化知识，主题相关
- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](../memory-sram/exp_ascend_large_shape_block_001.md) — 主题相关：共享 通用优化
- 相关主题: [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](../memory-sram/exp_perf_011.md) — 同属优化知识，主题相关
- 相关主题: [UB 打满策略：调大分块吃满 192KB，微超则把 mask/exp 计算外提到调用侧](../memory-sram/ub-fill-and-compute-hoist.md) — 同属优化知识，主题相关
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
- 相关主题: [反向 Q kernel：CV 均衡 + 双缓冲 + 代数掩码（(mask-1)*1e6）+ 64×128 三角遍历](../portability/exp_sda_bwd_q_001.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend mask 纪律与编译期陷阱（fp32 mask 污染 / stride-0 memref / 禁 % 运算符 / 2D mask store 丢格）](../portability/triton-ascend-mask-codegen-traps.md) — mask 检查项实例
- 相关主题: [Triton-Ascend kernel 调试排查清单](debugging-checklist.md) — 内存访问三查即本卡 Part1 的实例化
- 相关主题: [动态长度标量更新 kernel：do_not_specialize 防多版本编译，编译时间降 >80%](exp_bonus_renew_001.md) — 主题相关：共享 编译/精度修复链
- 相关主题: [控制流避坑：禁 return/break/continue，复杂条件分步 mask 组合](exp_debug_002.md) — 主题相关：共享 general_debug, 编译/精度修复链, 通用优化
- 相关主题: [Grid/Block 配置校验：乘积 ≤65535、≤3 维 tuple、大 shape 交错循环](exp_debug_003.md) — 主题相关：共享 general_debug, 编译/精度修复链, 通用优化
- 相关主题: [并发写入必须用原子操作且类型与目标内存严格匹配](exp_debug_004.md) — 主题相关：共享 general_debug, 编译/精度修复链, 通用优化
- 相关主题: [切片与索引语法校验：禁 Python 切片、arange 结果禁 get_element](exp_debug_005.md) — 主题相关：共享 general_debug, 编译/精度修复链, 通用优化
- 相关主题: [if/else 分支 dtype 不一致：改 masked load，bool 张量 host 侧转 int32](exp_debug_triton_branch_type_mismatch_001.md) — 主题相关：共享 general_debug, 编译/精度修复链

<!-- okf:related:end -->
