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
updated_at: "2026-09-14T00:00:00Z"
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
| 大面积伪失败（数值错位/NaN/整组报错） | 环境双后端注册冲突（未设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 报 Two accelerators cannot be used） | 运行环境显式 `export TORCH_DEVICE_BACKEND_AUTOLOAD=0`，配套对齐 LD_LIBRARY_PATH；先查环境再怀疑代码 |

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
- `exp_ascend_split_001` — 硬件适配优化 （收益 20%）
  - 910b2/910b3平台下，BLOCK_SIZE≥65536会直接编译失败，线程块内存超缓存会导致访存效率下降50%以上；多次切分可在符合限制的前提下充分利用硬件算力
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
- `exp_ascend_int64_address_001` — 硬件约束 （收益 50%）
  - 在确认i32范围足够的前提下使用。对超大规模张量（如T>65535且B较大），可能需要分块或其他策略避免i64。
- `exp_debug_triton_branch_type_mismatch_001` — Debugging
  - 三个关联 bug 的修复链：(1) if/else 分支类型→ masked load； (2) bool→i8→ int32 转换；(3) tl.where 条件窗口更新。

<!-- okf:related:start -->

# 相关

- 实践案例: [BlockSparseAttention (block-sparse mask) on Ascend NPU](../../ops/block-sparse-attention.md) — 同属优化知识，主题相关
- 实践案例: [Elementwise add with a tail mask](../../ops/elementwise-add.md) — 同属优化知识，主题相关
- 实践案例: [RoPE paired-channel pointer addressing](../../ops/rope-pointer-addressing.md) — compile_error 修复实例（指针寻址边界谓词）
- 相关主题: [Compute Optimization: vectorization alignment](../compute-pipeline/exp_backward_sum_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: dim is multiple mask elision](../compute-pipeline/exp_causal_conv1d_prefill_007.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: dim is multiple mask elimination](../compute-pipeline/exp_causal_conv1d_update_005.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: memory alignment](../compute-pipeline/exp_moe_wgrad_npu_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: redundant mask elimination by matrix property](../compute-pipeline/exp_solve_tril_001.md) — 同属优化知识，主题相关
- 相关主题: [grid_size 必须与 num_cores 入参严格相等](../compute-pipeline/grid-strict-match-num-cores.md) — stride/grid 配置错误即 Part1/Part3 排查项
- 相关主题: [整数比较/加法必须转 float32，避免 Vector 标量降级](../compute-pipeline/int-compare-cast-float32.md) — aiv_scalar_ratio 高排查项映射到本卡
- 相关主题: [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](../memory-sram/exp_perf_011.md) — mask/block_ptr 访存修复与 Part1 三查一致
- 相关主题: [Memory Access Optimization: sorted bin grouping with padded alignment](../memory-sram/exp_scatter_moe_padded_001.md) — 同属优化知识，主题相关
- 相关主题: [输入进入 kernel 前必须 contiguous](../memory-sram/input-contiguous.md) — stride 匹配布局检查项的 host 侧前置
- 相关主题: [Cube-Vector 场景行宽必须 512B 对齐](../memory-sram/row-width-512b-align.md) — alignment 检查项的具体契约
- 相关主题: [UB 打满策略：调大分块吃满 192KB，微超则把 mask/exp 计算外提到调用侧](../memory-sram/ub-fill-and-compute-hoist.md) — Part3 UB 超限映射的预算策略卡
- 相关主题: [kernel 内用 tl.cdiv，host 侧用 triton.cdiv（禁止混用）](../portability/cdiv-kernel-vs-host.md) — compile_error 映射表条目的具体契约
- 相关主题: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](../portability/exp_ascend_blockptr_boundary_scalar_001.md) — block_ptr boundary_check 标量降级修复实例
- 相关主题: [反向 Q kernel：CV 均衡 + 双缓冲 + 代数掩码（(mask-1)*1e6）+ 64×128 三角遍历](../portability/exp_sda_bwd_q_001.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend mask 纪律与编译期陷阱（fp32 mask 污染 / stride-0 memref / 禁 % 运算符 / 2D mask store 丢格）](../portability/triton-ascend-mask-codegen-traps.md) — Part1 mask 检查与陷阱契约互补
- 相关主题: [内存访问三查：mask 防越界、stride 匹配布局、索引上限判断](exp_debug_001.md) — 内存访问三查即本卡 Part1 的实例化
- 相关主题: [控制流避坑：禁 return/break/continue，复杂条件分步 mask 组合](exp_debug_002.md) — 控制流避坑与本卡 Part3 映射同源
- 相关主题: [GPU-source 迁移到 Ascend NPU 的 memory/index 改造检查清单](gpu-to-ascend-migration-checklist.md) — 迁移前置检查与调试排查清单互补
- 相关主题: [Profiling 驱动的瓶颈诊断与优化路由](profiling-bottleneck-checklist.md) — 排查流程第 4 步（性能问题）路由到 profiling 清单

<!-- okf:related:end -->
