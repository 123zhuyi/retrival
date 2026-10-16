---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: meta-skills
dimension: "6.2"
title: Triton-Ascend kernel 调试排查清单
description: 编译错误、精度异常、运行崩溃三类问题的系统性排查步骤：mask/stride/索引、fp32 参考实现对比、编译错误映射表。
tags: [debugging, mask, stride, alignment, numerical_error, compile_error]
confidence: verified
status: verified
created_at: "2026-08-10T00:00:00Z"
updated_at: "2026-08-10T00:00:00Z"
---

## DG-10 Triton-Ascend kernel 调试排查清单

【优化维度】：Agent 元技能｜修复链
【泛化层级】：Triton通用（迁移条件：所有 Triton-Ascend kernel 调试；无失效边界）

<原则：Ascend 后端调试信息比 GPU 稀疏，必须按“内存访问 → 数值精度 → 编译错误”三步系统化排查。80% 的问题由 mask 缺失、stride 不匹配或索引 int64 降级引起。>

**Part 1：内存访问三项检查**

| 检查项 | 必做动作 | 常见错误 |
|---|---|---|
| mask | 所有 `tl.load/tl.store` 必须带 `mask` 或 `boundary_check` | 无 mask，越界读写出 NaN/崩溃 |
| stride | stride 必须与张量实际内存布局匹配 | 写死 `stride=2` 或 transpose 后未更新 |
| 索引类型 | `tl.program_id/tl.arange` 结果显式 `.to(tl.int32)` | int64 地址计算降级为标量循环 |

```python
# 正确
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
data = tl.load(ptr + offsets * stride, mask=mask)

# 错误
data = tl.load(ptr + offsets * stride)            # 无 mask
data = tl.load(ptr + offsets * 2, ...)            # stride 写死
```

**Part 2：精度对比调试**

```python
# 1. fp32 参考实现
ref = reference_impl(x.float())

# 2. Triton kernel
out = triton_kernel(x)

# 3. 相对误差
max_err = (out - ref).abs().max() / ref.abs().max()
# fp16 应 < 1e-3；fp32 应 < 1e-5

# 4. 误差大时，中间计算升 fp32
acc = tl.zeros([BLOCK], dtype=tl.float32)
data = tl.load(ptr + offsets, mask=mask).to(tl.float32)
# ... compute ...
tl.store(out_ptr + offsets, acc.to(tl.float16), mask=mask)
```

**Part 3：编译错误快速映射表**

| 错误/症状 | 最常见原因 | 修复方向 |
|---|---|---|
| `tl.constexpr` 相关错误 | host 侧误用 tl.constexpr | 改为普通 Python 变量 |
| Grid 乘积超限 | x*y*z > 65535 | 减小 grid 维度或增大 BLOCK |
| UB 超限 | tile × dtype > 192KB | 减小 BLOCK_M/N/K |
| 维度对齐错误 | 尾轴 stride 非 512B 倍数 | pad 或改 BLOCK |
| NaN/Inf 输出 | exp 溢出、除零、mask 错误 | 见 [exp 稳定化](../algorithm-numerics/exp-stable-softmax-rmsnorm.md) |
| `aiv_scalar_ratio` 高 | int64/int32 比较或加法降级 | `.to(tl.float32)` 后比较/加法 |
| `aiv_mte2_time` 高 | 非连续访问或 bank conflict | 检查 stride、block_ptr、对角 mapping |

**排查流程图**：

```
1. 编译失败？
   YES → 查 constexpr/grid/UB/对齐
   NO  → 2
2. 运行崩溃/NaN/Inf？
   YES → 查 mask/stride/索引 + exp/除零
   NO  → 3
3. 精度不对？
   YES → fp32 参考实现对比，逐步升精度定位
   NO  → 性能问题，转 profiling 流程
```

## 已知实例（P3 挂载）

### Experience YAML 反链
- `exp_api_001` — 语法避坑 （收益 0%）
  - 该规则为Triton通用语法避坑项，违反会导致编译失败/运行错误，无直接性能提升但可保证代码正确性；910b2/910b3平台对语法违规的报错更严格
- `exp_api_002` — 语法适配 （收益 0%）
  - Ascend后端特有适配规则，`tl.where`在复杂嵌套条件下可能触发执行错误，需切换为if-else；简单条件保留`tl.where`可减少代码复杂度
- `exp_api_003` — 循环适配 （收益 0%）
  - Ascend后端不支持while循环，静态上限场景`for range`是最优替代，无额外开销；循环上限必须为`tl.constexpr`，否则归为动态上限场景
- `exp_api_004` — 循环适配 （收益 -5%）
  - 动态上限场景存在约5%的空循环开销（因遍历至MAX_ITERS）；910b2/910b3平台需保证`MAX_ITERS`覆盖所有可能的`n_iters`值，避免逻辑截断
- `exp_slice_001` — 语法规范 （收益 0%）
  - 该规则为Triton切片操作核心语法规范，违反会直接导致编译失败；910b2/910b3平台对Python风格切片语法无兼容容错，必须严格使用专用API
- `exp_slice_002` — 语法避坑 （收益 0%）
  - tl.arange生成的是编译时常量索引表达式，而非实际占用内存的张量，调用`get_element()`会触发编译错误；910b2/910b3平台无容错机制，必须严格遵循该规则
- `exp_tensor_create_001` — 内存初始化优化 （收益 8%）
  - 910b2/910b3平台下，`torch.empty`系列API相比`zeros/ones`可节省约8%的初始化时间，尤其对大张量（元素数≥1e6）效果更显著
- `exp_ascend_split_001` — 硬件适配优化 （收益 20%）
  - 910b2/910b3平台下，BLOCK_SIZE≥65536会直接编译失败，线程块内存超缓存会导致访存效率下降50%以上；多次切分可在符合限制的前提下充分利用硬件算力
- `exp_ascend_core_num_001` — 硬件适配优化 （收益 0%）
  - 该规则无直接性能提升，但可规避forward中设备同步导致的10%-15%性能损耗；910B2默认VEC=48、CUBE=24，910B3默认VEC=64、CUBE=32，可按需调整兜底值
- `exp_ascend_parallel_pattern_001` — 负载均衡优化 （收益 10%）
  - 该模式相比手动分块的并行方案，负载均衡性提升10%，且代码量减少30%；适配所有总任务数远大于核心数的大shape场景
- `exp_ascend_large_shape_block_001` — 内存访问优化 （收益 12%）
  - 该方案通过连续内存访问提升缓存命中率，相比交错循环方案，内存密集型算子性能提升12%；优先用于大向量element-wise、连续分块matmul等对内存连续性敏感的场景
- `exp_debug_001` — 调试排查 （收益 0%）
  - 内存访问问题是Triton kernel最常见的崩溃原因，910b系列平台下越界访问会直接触发设备异常；stride参数错误会导致数据读取错乱，需与张量实际stride严格一致
- `exp_debug_002` — 调试排查 （收益 0%）
  - 控制流关键字（return/break/continue）误用会导致编译失败，复杂条件用mask组合可避免控制流分支带来的性能损耗，同时保证逻辑正确性
- `exp_debug_003` — 调试排查 （收益 0%）
  - Grid配置错误是kernel启动失败的高频原因，910b系列平台对Grid维度和大小限制无容错机制；大shape算子未用交错循环会直接触发Grid超限报错
- `exp_debug_004` — 调试排查 （收益 0%）
  - 并发写入未用原子操作会导致数据竞争（结果随机错误），原子操作类型不匹配会触发编译失败；910b系列平台仅支持基础原子操作（add/max/min），需按需选择
- `exp_debug_005` — 调试排查 （收益 0%）
  - Python风格切片语法在Triton中完全不支持，会直接编译失败；`tl.arange`是编译期表达式而非实际张量，`tl.get_element`调用会触发语法错误
- `exp_attn_fp8_ub_slice_001` — Memory Access Optimization （收益 30%）
  - The slicing logic is specifically required for HEAD_DIM >= 256 to avoid hardware resource limits. For smaller heads, the standard…
  - 对应 kernel: `raw_experiences/extracted_kernels/011-_attn_fwd_inner.py`
- `exp_ascend_compile_hint_001` — 调优经验 （收益 25%）
  - 此优化针对的是过度使用compile_hint的情况。如果原始kernel没有加hint，则无需此优化。移除hint后应验证性能和精度无回退。
- `exp_ascend_int64_address_001` — 硬件约束 （收益 50%）
  - 在确认i32范围足够的前提下使用。对超大规模张量（如T>65535且B较大），可能需要分块或其他策略避免i64。
- `exp_debug_triton_branch_type_mismatch_001` — Debugging
  - 三个关联 bug 的修复链：(1) if/else 分支类型→ masked load； (2) bool→i8→ int32 转换；(3) tl.where 条件窗口更新。

<!-- okf:related:start -->

# 相关

- 实践案例: [BlockSparseAttention (block-sparse mask) on Ascend NPU](../../ops/block-sparse-attention.md) — 同属优化知识，主题相关
- 实践案例: [Elementwise add with a tail mask](../../ops/elementwise-add.md) — 同属优化知识，主题相关
- 实践案例: [RoPE paired-channel pointer addressing](../../ops/rope-pointer-addressing.md) — 主题相关：共享 compile_error
- 开发指南: [Legacy 知识单元类型与格式规范（triton-knowledge-retrieval）](../../reference/SPEC-Knowledge-Unit.md) — 同属优化知识，主题相关
- 开发指南: [Legacy 缓存读写协议（triton-knowledge-retrieval）](../../reference/SPEC-Retrieve-legacy.md) — 同属优化知识，主题相关
- 相关主题: [Triton API Quick Reference for Ascend Development](../../reference/triton-api-cheatsheet.md) — 同属优化知识，主题相关
- 相关主题: [grid_size 必须与 num_cores 入参严格相等](../compute-pipeline/grid-strict-match-num-cores.md) — 主题相关：共享 stride
- 相关主题: [整数比较/加法必须转 float32，避免 Vector 标量降级](../compute-pipeline/int-compare-cast-float32.md) — 主题相关：共享 mask
- 相关主题: [persistent kernel 模式（grid-stride 摊还启动开销）](../compute-pipeline/persistent-kernel-pattern.md) — 同属优化知识，主题相关
- 相关主题: [Triton Block Size 通用调优原则](../compute-pipeline/traj-general_perf-block_size.md) — 同属优化知识，主题相关
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](../compute-pipeline/traj-rope-scheduling.md) — 同属优化知识，主题相关
- 相关主题: [输入进入 kernel 前必须 contiguous](../memory-sram/input-contiguous.md) — 主题相关：共享 stride
- 相关主题: [pertensor scale 必须物化为连续 [1,N]，禁止 stride-0 广播 load](../memory-sram/pertensor-scale-materialize.md) — 同属优化知识，主题相关
- 相关主题: [Cube-Vector 场景行宽必须 512B 对齐](../memory-sram/row-width-512b-align.md) — 主题相关：共享 alignment
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](../memory-sram/traj-general_perf-memory_access.md) — 主题相关：共享 stride
- 相关主题: [kernel 内用 tl.cdiv，host 侧用 triton.cdiv（禁止混用）](../portability/cdiv-kernel-vs-host.md) — 主题相关：共享 compile_error
- 相关主题: [GPU-source 迁移到 Ascend NPU 的 memory/index 改造检查清单](gpu-to-ascend-migration-checklist.md) — 同属优化知识，主题相关
- 相关主题: [Profiling 驱动的瓶颈诊断与优化路由](profiling-bottleneck-checklist.md) — 同属优化知识，主题相关
- 相关主题: [Triton Kernel 通用语法避坑清单](traj-general_api-syntax_pitfall.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend 切片语法避坑：tl.arange 不是张量](traj-general_ascend-syntax_pitfall.md) — 同属优化知识，主题相关
- 相关主题: [Ascend Triton 语法规范避坑聚合](traj-general_ascend-syntax_spec.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend 调试排查典型问题聚合](traj-general_debug-debugging.md) — 主题相关：共享 compile_error, debugging
- 相关主题: [避免 Stride 访问：优先连续化后一维遍历](traj-general_perf-perf_pitfall.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
