---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: meta-skills
dimension: 6.2
title: if/else 分支 dtype 不一致：改 masked load，bool 张量 host 侧转 int32
description: 910b2/910b3 + bf16/fp16 下 "Mismatched type for variable between then block and else block" 修复链：if/else 分支改 masked load + other=0.0；bool 张量 Ascend load 后为 i8 不能作 mask，host 侧转 int32 再比较；滑窗循环越界时 tl.where 窗口值被零覆盖
tags: [编译/精度修复链, general, general_debug]
confidence: unverified
status: draft
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## if/else 分支 dtype 不一致：改 masked load，bool 张量 host 侧转 int32

【优化维度】：编译/精度修复链
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3；失效边界：SSA 分支 dtype 一致性是 Triton 前端通用约束，非 Ascend 独有；bool→i8 是 Ascend 后端的 load 行为，GPU 上 bool 张量可直接作 mask；masked load + other=0.0 仅当零值是合法中性元素时等价于 else 分支，否则需显式统一两个分支的初始化 dtype；仅 910b2/910b3 + bf16/fp16 验证）

### 原则
- **症状 → 根因**：报 `Mismatched type for variable between then block and else block` —— Triton SSA 要求同一变量在 if/else 两个分支产生相同 dtype；then 分支从 tensor load 得 bf16、else 分支 `tl.zeros` 默认 fp32 时触发
- **修复 1（分支统一）**：用 masked load（`mask=condition, other=0.0`）替代 if/else 分支，或保证两分支初始化 dtype 完全一致
- **修复 2（bool mask）**：bool 张量在 Ascend 上 load 后为 i8，不能直接作 mask（报 `tt.load op operand #1 must be 1-bit signless integer (mask is i8 instead of i1)`）；host 侧转 int32，kernel 内用 `int32 > 0` 比较生成 bool mask
- **修复 3（滑窗覆盖）**：循环超出有效 token 范围时，tl.where 条件窗口的值会被零覆盖（`Sliding window values overwritten by zeros when loop exceeds valid token range`）；需用有效范围 mask 保护窗口更新

### 代码/骨架出处
- `Triton`:
```python
# 反模式：if/else 分支 dtype 不一致
# if cond:
#     x = tl.load(ptr + offs)          # bf16
# else:
#     x = tl.zeros([BLOCK], dtype=tl.float32)  # fp32 → SSA 类型冲突

# 正模式 1：masked load 替代分支
x = tl.load(ptr + offs, mask=cond, other=0.0)

# 正模式 2：bool 张量作 mask（host 侧先转 int32）
cond_i32 = tl.load(cond_ptr + offs)      # host 已转 int32
mask = cond_i32 > 0                       # 比较生成 i1 bool mask
data = tl.load(data_ptr + offs, mask=mask, other=0.0)
```

**坏实践（反例）**：把 host 传入的 bool 张量在 kernel 里 load 后直接当 mask 用 → Ascend 上 bool load 为 i8，编译报 `mask is i8 instead of i1` → ✅ 正解即本条：host 侧转 int32 + kernel 内 `> 0` 比较；if/else 分支各写各的 dtype 则触发 SSA 类型冲突

**已知实例**：general_debug §raw (`experiences/general_debug/exp_debug_triton_branch_type_mismatch_001.yaml`)
- 备注：三个关联 bug 的修复链：(1) if/else 分支类型→ masked load； (2) bool→i8→ int32 转换；(3) tl.where 条件窗口更新。

<!-- okf:related:start -->

# 相关

- 实践案例: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](../../ops/moe-compute-expert-tokens.md) — 同属优化知识，主题相关
- 相关主题: [反量化域必须按 x.dtype 分流（fp16 域 vs fp32 域）](../algorithm-numerics/dequant-domain-dtype-split.md) — 同属优化知识，主题相关
- 相关主题: [GEMM 操作数保持 low_dtype，fp32 累加](../algorithm-numerics/gemm-low-dtype-fp32-accum.md) — 同属优化知识，主题相关
- 相关主题: [编译器融合消除中间 dtype 舍入时，拆分独立 kernel 强制 GM round-trip](../algorithm-numerics/split-kernel-gm-roundtrip.md) — 同属优化知识，主题相关
- 相关主题: [块大小选型：2 的幂次 128/256/512 起调，blockSize×dtypeBytes 对齐 32B 且不超 UB](../autotuning/exp_perf_001.md) — 主题相关：共享 general
- 相关主题: [autotune 配置空间须覆盖 num_warps(2/4/8)×num_stages(2/3/4) 与 tile 组合](../autotuning/exp_perf_017.md) — 主题相关：共享 general
- 相关主题: [性能避坑: 避免多维 stride 访问](../compute-pipeline/exp_perf_007.md) — 主题相关：共享 general
- 相关主题: [标量消除: i32 比较转 fp32 走向量单元](../compute-pipeline/exp_perf_012.md) — 主题相关：共享 general
- 相关主题: [标量消除: min() 运行时分支改 tl.where](../compute-pipeline/exp_perf_013.md) — 主题相关：共享 general
- 相关主题: [标量消除: tl.cdiv 编译期循环边界](../compute-pipeline/exp_perf_014.md) — 主题相关：共享 general
- 相关主题: [整数比较/加法必须转 float32，避免 Vector 标量降级](../compute-pipeline/int-compare-cast-float32.md) — 同属优化知识，主题相关
- 相关主题: [kernel 内索引必须转 int32，避免 int64 降级](../compute-pipeline/int32-index-cast.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: constexpr strides](../memory-sram/exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 general
- 相关主题: [Add算子块大小动态调整优化经验](../memory-sram/exp_elem_add_001_20260526.md) — 主题相关：共享 general
- 相关主题: [带 alpha 缩放的逐元素 add 访存与块大小优化](../memory-sram/exp_elemwise_add_001_20260526.md) — 主题相关：共享 general
- 相关主题: [内存访问优化：2D 张量块指针访存（make_block_ptr）](../memory-sram/exp_perf_002.md) — 主题相关：共享 general
- 相关主题: [内存访问优化：非连续张量连续化 + 一维遍历](../memory-sram/exp_perf_003.md) — 主题相关：共享 general
- 相关主题: [内存访问优化：stride 参数一律传 tensor.stride() 原生值](../memory-sram/exp_perf_004.md) — 主题相关：共享 general
- 相关主题: [内存访问优化：contiguous() 一次性拷贝替代 stride 访问](../memory-sram/exp_perf_006.md) — 主题相关：共享 general
- 相关主题: [内存初始化优化：输出张量创建优选 torch.empty_like](../memory-sram/exp_perf_008.md) — 主题相关：共享 general
- 相关主题: [内存访问优化：行优先连续加载 + tl.trans 寄存器转置](../memory-sram/exp_perf_015.md) — 主题相关：共享 general
- 相关主题: [内存访问优化：跨阶段寄存器复用消除重复 load](../memory-sram/exp_perf_016.md) — 主题相关：共享 general
- 相关主题: [多维输入必须在 host 侧扁平化为 2D](../memory-sram/host-flatten-to-2d.md) — 同属优化知识，主题相关
- 相关主题: [kernel 内用 tl.cdiv，host 侧用 triton.cdiv（禁止混用）](../portability/cdiv-kernel-vs-host.md) — 同属优化知识，主题相关
- 相关主题: [Ascend 后端禁用语法及替代写法（控制流/lambda/链式布尔/直接索引/负偏移）](../portability/exp_api_001.md) — 主题相关：共享 general
- 相关主题: [复杂嵌套条件下 tl.where 改 if-else 分支](../portability/exp_api_002.md) — 主题相关：共享 general
- 相关主题: [动态上限循环：for + if + MAX_ITERS 编译期上界替代 while](../portability/exp_api_004.md) — 主题相关：共享 general
- 相关主题: [地址计算全程保持 int32，避免 int64 标量降级](../portability/exp_ascend_int64_address_001.md) — 同属优化知识，主题相关
- 相关主题: [禁止用 tl.where 动态计算内存偏移，改 if-else 静态分支](../portability/exp_ascend_where_001.md) — 主题相关：共享 general
- 相关主题: [tl.constexpr 仅限内核参数声明，Host 侧禁用](../portability/exp_constexpr_001.md) — 主题相关：共享 general
- 相关主题: [标签索引转 int32 比较 + GROUP_SIZE_M 分组调度 + 大 K tile](../portability/exp_entropy_backward_npu_001.md) — 同属优化知识，主题相关
- 相关主题: [动态长度标量更新 kernel：do_not_specialize 防多版本编译，编译时间降 >80%](exp_bonus_renew_001.md) — 主题相关：共享 编译/精度修复链
- 相关主题: [内存访问三查：mask 防越界、stride 匹配布局、索引上限判断](exp_debug_001.md) — 主题相关：共享 general_debug, 编译/精度修复链
- 相关主题: [控制流避坑：禁 return/break/continue，复杂条件分步 mask 组合](exp_debug_002.md) — 主题相关：共享 general_debug, 编译/精度修复链
- 相关主题: [Grid/Block 配置校验：乘积 ≤65535、≤3 维 tuple、大 shape 交错循环](exp_debug_003.md) — 主题相关：共享 general_debug, 编译/精度修复链
- 相关主题: [并发写入必须用原子操作且类型与目标内存严格匹配](exp_debug_004.md) — 主题相关：共享 general_debug, 编译/精度修复链
- 相关主题: [切片与索引语法校验：禁 Python 切片、arange 结果禁 get_element](exp_debug_005.md) — 主题相关：共享 general_debug, 编译/精度修复链

<!-- okf:related:end -->
