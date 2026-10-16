---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: '标量消除: i32 比较转 fp32 走向量单元'
description: 'general_perf 场景下，标量消除: i32 比较转 fp32 走向量单元，收益 35%'
tags: [AICore 单元利用/Scalar 削减, general, general_perf]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 标量消除: i32 比较转 fp32 走向量单元

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：仅 i32 的 LT/GT/LE/GE 比较标量降级，i32 的 EQ/NE 与 f16/f32 比较本就走向量单元；程序级单次分支（如 if i_ti < T_local）影响极小无需转换；索引范围需在 fp32 精确表示内（mask 语义无精度损失））

### 原则
- **降级规则**：在Ascend 910B上，i32的LT/GT/LE/GE比较全部标量降级；只有f16/f32比较和i32的EQ/NE走向量单元
- **症状**：profiling中aiv_scalar_ratio高，且代码中存在大量mask生成（如o_k < K, o_i >= j, o_i[:,None] >= o_i[None,:]等）
- **核心方案**：将整数比较操作数转换为fp32后再比较，使比较走Vector CMP单元而非Scalar
- **注意**：程序级单次分支（如if i_ti < T_local）只执行一次，影响极小，无需转换；重点优化循环内或向量化mask比较
- **典型收益**：每个mask从逐元素标量比较变为向量比较，在BC=16场景下每个mask节省16~256次标量操作；在KDA intra kernel上10+处mask可显著降低AIV scalar

### 代码/骨架出处
- `Triton`:
```python
# 反模式：i32 LT/GT/LE/GE → 标量降级
m_k = o_k < K
m_i_diag_qk = (o_i[:, None] >= o_i[None, :]) & ((i_ti + o_i[:, None]) < T_local)

# 正模式：转fp32后比较 → 向量单元
o_i_f = o_i.to(tl.float32)
o_k_f = o_k.to(tl.float32)
T_local_f = T_local.to(tl.float32)
K_f = K.to(tl.float32)
m_k = o_k_f < K_f
m_i_diag_qk = (o_i_f[:, None] >= o_i_f[None, :]) & ((i_ti + o_i_f[:, None]) < T_local_f)
```

**坏实践（反例）**：循环内向量化 mask 用 i32 的 LT/GT/LE/GE 比较（如 `o_k < K`）→ 全部标量降级到 Scalar 单元，aiv_scalar_ratio 飙高（BC=16 场景每个 mask 浪费 16~256 次标量操作）→ ✅ 正解即本条：比较操作数转 fp32 再比较，走 Vector CMP 单元（对 mask 逻辑无精度损失）。

**已知实例**：general_perf §raw (`experiences/general_perf/exp_perf_012.yaml`)
- 声称性能收益：35%
- 备注：此优化对由tl.arange生成的索引比较尤其有效。当参与比较的变量本身就是tl.full构造的fp32标量时，可直接使用而无需额外转换。注意fp32比较对mask逻辑无精度损失。

<!-- okf:related:start -->

# 相关

- 相关主题: [块大小选型：2 的幂次 128/256/512 起调，blockSize×dtypeBytes 对齐 32B 且不超 UB](../autotuning/exp_perf_001.md) — 主题相关：共享 general, general_perf
- 相关主题: [autotune 配置空间须覆盖 num_warps(2/4/8)×num_stages(2/3/4) 与 tile 组合](../autotuning/exp_perf_017.md) — 主题相关：共享 general, general_perf
- 相关主题: [Memory Access Optimization: constexpr strides](../memory-sram/exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 general
- 相关主题: [Add算子块大小动态调整优化经验](../memory-sram/exp_elem_add_001_20260526.md) — 主题相关：共享 general
- 相关主题: [带 alpha 缩放的逐元素 add 访存与块大小优化](../memory-sram/exp_elemwise_add_001_20260526.md) — 主题相关：共享 general
- 相关主题: [内存访问优化：2D 张量块指针访存（make_block_ptr）](../memory-sram/exp_perf_002.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存访问优化：非连续张量连续化 + 一维遍历](../memory-sram/exp_perf_003.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存访问优化：stride 参数一律传 tensor.stride() 原生值](../memory-sram/exp_perf_004.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存访问优化：contiguous() 一次性拷贝替代 stride 访问](../memory-sram/exp_perf_006.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存初始化优化：输出张量创建优选 torch.empty_like](../memory-sram/exp_perf_008.md) — 主题相关：共享 general, general_perf
- 相关主题: [冗余操作优化：同形 element-wise 算子免 reshape](../memory-sram/exp_perf_009.md) — 主题相关：共享 general_perf
- 相关主题: [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](../memory-sram/exp_perf_011.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存访问优化：行优先连续加载 + tl.trans 寄存器转置](../memory-sram/exp_perf_015.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存访问优化：跨阶段寄存器复用消除重复 load](../memory-sram/exp_perf_016.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存初始化优化：输出张量创建 API 选择](../memory-sram/exp_tensor_create_001.md) — 主题相关：共享 general
- 相关主题: [if/else 分支 dtype 不一致：改 masked load，bool 张量 host 侧转 int32](../meta-skills/exp_debug_triton_branch_type_mismatch_001.md) — 主题相关：共享 general
- 相关主题: [Ascend 后端禁用语法及替代写法（控制流/lambda/链式布尔/直接索引/负偏移）](../portability/exp_api_001.md) — 主题相关：共享 general
- 相关主题: [复杂嵌套条件下 tl.where 改 if-else 分支](../portability/exp_api_002.md) — 主题相关：共享 general
- 相关主题: [静态上限循环：while 改 for range（上限须为 tl.constexpr）](../portability/exp_api_003.md) — 主题相关：共享 general
- 相关主题: [动态上限循环：for + if + MAX_ITERS 编译期上界替代 while](../portability/exp_api_004.md) — 主题相关：共享 general
- 相关主题: [禁止用 tl.where 动态计算内存偏移，改 if-else 静态分支](../portability/exp_ascend_where_001.md) — 主题相关：共享 general
- 相关主题: [tl.constexpr 仅限内核参数声明，Host 侧禁用](../portability/exp_constexpr_001.md) — 主题相关：共享 general
- 相关主题: [禁用 Python 切片语法，改用 get_element/extract_slice/insert_slice](../portability/exp_slice_001.md) — 主题相关：共享 general
- 相关主题: [禁止对 tl.arange 索引表达式调用 get_element](../portability/exp_slice_002.md) — 主题相关：共享 general
- 相关主题: [标量消除: block_ptr boundary_check 与 i32 比较定向消除](exp_kda_bwd_intra_scalar_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [标量消除: i64/i32 标量降级定向消除](exp_kda_bwd_wy_scalar_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [性能避坑: 避免多维 stride 访问](exp_perf_007.md) — 主题相关：共享 aicore 单元利用/scalar 削减, general, general_perf
- 相关主题: [算子拆分优化: 按计算阶段拆分为简单 kernel](exp_perf_010.md) — 主题相关：共享 general_perf
- 相关主题: [标量消除: min() 运行时分支改 tl.where](exp_perf_013.md) — 主题相关：共享 aicore 单元利用/scalar 削减, general, general_perf
- 相关主题: [标量消除: tl.cdiv 编译期循环边界](exp_perf_014.md) — 主题相关：共享 aicore 单元利用/scalar 削减, general, general_perf

<!-- okf:related:end -->
