---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: constexpr strides"
description: "conv 场景下，Memory Access Optimization: constexpr strides，收益 22%"
tags: [访存效率, general, conv]
confidence: provisional
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: constexpr strides

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3，bf16/fp16，D≥1024、W=4 的变长滑窗场景；失效边界：D<1024 时 BLOCK_D=256 的降 grid 收益稀释；W≠4 需重算滑窗寄存器个数与重叠区长度；care_padding=False 要求 load 无越界 padding 需求，存在边界越界 mask 的访问模式不适用；bool→int32 与 float32 比较为 Ascend 编译器绕坑写法，其他后端无需迁移）

### 原则
- **constexpr strides**：将 conv_states 的 stride_line 和 stride_state 声明为 tl.constexpr， 使编译器在编译期完成地址偏移常量折叠，消除运行时乘法指令。贡献度 40%。
- **block d 256**：BLOCK_D 从 128 增大到 256，将 dim=4096 时的 grid 程序数从 32 降为 16， 每程序工作量翻倍，减少调度开销。UB 占用仍远低于 192KB。贡献度 35%。
- **care padding false**：所有 tl.load 使用 care_padding=False，消除 MTE2→Vector 同步屏障， 提升 load/compute 流水线并行度。贡献度 25%。
- **conditional window shift**：使用 tl.where 条件更新滑动窗口（prev0/prev1/prev2）， 避免循环超出有效 token 范围后用零值覆盖历史状态。这是功能正确性修复但也影响性能。
- **bool to int32**：将 bool tensor (has_initial_state) 在 host 侧转为 int32， kernel 中用 int32 > 0 比较替代直接 bool mask，避免 i8 mask 编译错误。
- **float32 compare**：标量整数比较 (cache_idx >= 0) 转换为 float32 比较， 避免 int32 CMP scalar fallback。

### 代码/骨架出处
- `description`:
```python
Causal Conv1D forward + state update, 2D grid (batch, cdiv(dim,BLOCK_D)), sliding window register reuse
```

**坏实践（反例）**：stride 作为运行时标量传入，寻址偏移在 kernel 内临时计算
```python
def kernel(..., stride_line, stride_state):        # 运行时标量
    offs = di * stride_line + si * stride_state    # 每次 load 都附带乘法
    st = tl.load(conv_states_ptr + offs, care_padding=True, ...)  # 还多一道 MTE2→Vector 屏障
```
后果：地址偏移无法常量折叠，每次访存附带运行时乘法指令（YAML 实测贡献度 40%）；care_padding 默认开启又引入同步屏障（贡献度 25%）。✅ 正解：stride_line/stride_state 声明为 tl.constexpr 做编译期常量折叠，所有 tl.load 置 care_padding=False，并把 BLOCK_D 提到 256 使 grid 程序数减半。

**已知实例**：conv §raw (`experiences/conv/exp_causal_conv1d_varlen_opt_001.yaml`)
- 声称性能收益：22%
- 备注：此算子为 Causal Conv1D with variable-length sequences, conv_states update, optional bias and SiLU activation。关键挑战：(1) if/else 分支类型一致性约束； (2) bool tensor 在 Ascend 上的 i8 类型问题；(3) 循环超出有效范围时窗口移位逻辑保护。 最终实现 8.47x 加速比（vs PyTorch baseline），优化后 1.22x vs Phase 3 基线。

<!-- okf:related:start -->

# 相关

- 开发指南: [Ascend NPU 特有语法与模式](../../reference/ascend-patterns.md) — 同属优化知识，主题相关
- 相关主题: [块大小选型：2 的幂次 128/256/512 起调，blockSize×dtypeBytes 对齐 32B 且不超 UB](../autotuning/exp_perf_001.md) — 主题相关：共享 general
- 相关主题: [autotune 配置空间须覆盖 num_warps(2/4/8)×num_stages(2/3/4) 与 tile 组合](../autotuning/exp_perf_017.md) — 主题相关：共享 general
- 相关主题: [用 tl.constexpr 编译期消除运行时分支](../compute-pipeline/constexpr-branch-elimination.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: two dimensional grid strategy](../compute-pipeline/exp_causal_conv1d_prefill_003.md) — 主题相关：共享 conv
- 相关主题: [Compute Optimization: pointer base precompute](../compute-pipeline/exp_causal_conv1d_prefill_004.md) — 主题相关：共享 conv
- 相关主题: [Scheduling Optimization: pad slot early exit](../compute-pipeline/exp_causal_conv1d_prefill_005.md) — 主题相关：共享 conv
- 相关主题: [Compute Optimization: conv bias silu triple fusion](../compute-pipeline/exp_causal_conv1d_prefill_006.md) — 主题相关：共享 conv
- 相关主题: [Compute Optimization: dim is multiple mask elision](../compute-pipeline/exp_causal_conv1d_prefill_007.md) — 主题相关：共享 conv
- 相关主题: [Compute Optimization: w loop unrolling with threshold](../compute-pipeline/exp_causal_conv1d_update_002.md) — 主题相关：共享 conv
- 相关主题: [Cross-kernel Optimization Transfer: token loop vs w loop decision](../compute-pipeline/exp_causal_conv1d_update_006.md) — 主题相关：共享 conv
- 相关主题: [Scheduling Optimization: bdt layout rationale](../compute-pipeline/exp_causal_conv1d_update_007.md) — 主题相关：共享 conv
- 相关主题: [性能避坑: 避免多维 stride 访问](../compute-pipeline/exp_perf_007.md) — 主题相关：共享 general
- 相关主题: [标量消除: i32 比较转 fp32 走向量单元](../compute-pipeline/exp_perf_012.md) — 主题相关：共享 general
- 相关主题: [标量消除: min() 运行时分支改 tl.where](../compute-pipeline/exp_perf_013.md) — 主题相关：共享 general
- 相关主题: [标量消除: tl.cdiv 编译期循环边界](../compute-pipeline/exp_perf_014.md) — 主题相关：共享 general
- 相关主题: [Compute Optimization: constexpr dual path full vs boundary](../compute-pipeline/exp_solve_tril_003.md) — 同属优化知识，主题相关
- 相关主题: [if/else 分支 dtype 不一致：改 masked load，bool 张量 host 侧转 int32](../meta-skills/exp_debug_triton_branch_type_mismatch_001.md) — 主题相关：共享 general
- 相关主题: [负优化排除清单：微小矩阵禁 tl.dot、constexpr 展开优先、preload 按 grid 规模决策](../meta-skills/exp_solve_tril_006.md) — 同属优化知识，主题相关
- 相关主题: [关键长度参数必须 tl.constexpr 静态化触发启动级特化](../portability/constexpr-launch-specialization.md) — 同属优化知识，主题相关
- 相关主题: [Ascend 后端禁用语法及替代写法（控制流/lambda/链式布尔/直接索引/负偏移）](../portability/exp_api_001.md) — 主题相关：共享 general
- 相关主题: [复杂嵌套条件下 tl.where 改 if-else 分支](../portability/exp_api_002.md) — 主题相关：共享 general
- 相关主题: [静态上限循环：while 改 for range（上限须为 tl.constexpr）](../portability/exp_api_003.md) — 主题相关：共享 general
- 相关主题: [动态上限循环：for + if + MAX_ITERS 编译期上界替代 while](../portability/exp_api_004.md) — 主题相关：共享 general
- 相关主题: [禁止用 tl.where 动态计算内存偏移，改 if-else 静态分支](../portability/exp_ascend_where_001.md) — 主题相关：共享 general
- 相关主题: [tl.constexpr 仅限内核参数声明，Host 侧禁用](../portability/exp_constexpr_001.md) — 主题相关：共享 general
- 相关主题: [禁用 Python 切片语法，改用 get_element/extract_slice/insert_slice](../portability/exp_slice_001.md) — 主题相关：共享 general
- 相关主题: [禁止对 tl.arange 索引表达式调用 get_element](../portability/exp_slice_002.md) — 主题相关：共享 general
- 相关主题: [Memory Access Optimization: kv tile reuse](exp_attn_bwd_hstu_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fp8 value casting](exp_attn_fp8_ub_slice_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: flattened grid scheduling](exp_causal_conv1d_001.md) — 主题相关：共享 conv, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_causal_conv1d_bwd_001.md) — 主题相关：共享 conv, 访存效率
- 相关主题: [Memory Access Optimization: sliding window register reuse](exp_causal_conv1d_prefill_002.md) — 主题相关：共享 conv, 访存效率
- 相关主题: [Memory Access Optimization: compile time branching](exp_causal_conv1d_states_001.md) — 主题相关：共享 conv, 访存效率
- 相关主题: [Memory Access Optimization: fused state update](exp_causal_conv1d_update_001.md) — 主题相关：共享 conv, 访存效率
- 相关主题: [Memory Access Optimization: make block ptr hardware acceleration](exp_causal_conv1d_update_003.md) — 主题相关：共享 conv, 访存效率
- 相关主题: [Memory Access Optimization: overlapping t tiling](exp_causal_conv1d_update_004.md) — 主题相关：共享 conv, 访存效率
- 相关主题: [Add算子块大小动态调整优化经验](exp_elem_add_001_20260526.md) — 主题相关：共享 general, 访存效率
- 相关主题: [带 alpha 缩放的逐元素 add 访存与块大小优化](exp_elemwise_add_001_20260526.md) — 主题相关：共享 general, 访存效率
- 相关主题: [Memory Access Optimization: simd index select](exp_flashattn_gqa_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: dynamic vectorization tuning](exp_groupnorm_001.md) — 主题相关：共享 访存效率
- 相关主题: [内存访问优化：转置 load 行优先化与跨阶段复用](exp_kda_bwd_intra_memory_opt_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: adaptive tiling strategy](exp_layernorm_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion](exp_lightning_attn_decode_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache](exp_mla_paged_attn_001.md) — 主题相关：共享 访存效率
- 相关主题: [内存访问优化：2D 张量块指针访存（make_block_ptr）](exp_perf_002.md) — 主题相关：共享 general, 访存效率
- 相关主题: [内存访问优化：非连续张量连续化 + 一维遍历](exp_perf_003.md) — 主题相关：共享 general, 访存效率
- 相关主题: [内存访问优化：stride 参数一律传 tensor.stride() 原生值](exp_perf_004.md) — 主题相关：共享 general, 访存效率
- 相关主题: [内存访问优化：contiguous() 一次性拷贝替代 stride 访问](exp_perf_006.md) — 主题相关：共享 general, 访存效率
- 相关主题: [内存初始化优化：输出张量创建优选 torch.empty_like](exp_perf_008.md) — 主题相关：共享 general, 访存效率
- 相关主题: [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](exp_perf_011.md) — 主题相关：共享 general, 访存效率
- 相关主题: [内存访问优化：行优先连续加载 + tl.trans 寄存器转置](exp_perf_015.md) — 主题相关：共享 general, 访存效率
- 相关主题: [内存访问优化：跨阶段寄存器复用消除重复 load](exp_perf_016.md) — 主题相关：共享 general, 访存效率
- 相关主题: [Memory Access Optimization: preload extract slice eliminate hbm read](exp_solve_tril_002.md) — 主题相关：共享 访存效率
- 相关主题: [内存初始化优化：输出张量创建 API 选择](exp_tensor_create_001.md) — 主题相关：共享 general, 访存效率
- 相关主题: [BLOCK 向上取 2 的幂并声明为 constexpr](pow2-block-constexpr.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
