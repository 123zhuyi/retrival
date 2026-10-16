---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: 内存访问优化：行优先连续加载 + tl.trans 寄存器转置
description: general_perf 场景下，内存访问优化：行优先连续加载 + tl.trans 寄存器转置，收益 30%
tags: [访存效率, general, general_perf]
confidence: provisional
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 内存访问优化：行优先连续加载 + tl.trans 寄存器转置

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3，fp16/bf16/fp32，原始数据行优先存储、需按转置视角加载子块的场景；失效边界：原始布局本身为列优先时不适用，需在 host 端预转置或改用其他数据编排；H*BT 较小（stride 接近连续）时收益稀释；BC 较大时寄存器转置推高寄存器压力，需评估 spill 风险）

### 原则
- **问题根因**：转置视角加载矩阵子块时（stride=(1, H*BT)），列方向每次跳跃H*BT个元素，当H*BT很大时（如H=32, BT=64 → stride=2048）访存极不连续，MTE2带宽利用率低
- **核心方案**：改为按原始行优先布局连续加载BC×BC子块，再用tl.trans在寄存器内转置。tl.trans是寄存器操作，开销远低于非合并访存
- **适用场景**：需要加载转置子块但原始数据以行优先存储的情况，即各类行优先存储矩阵需按转置视角读取子块的场景
- **边界处理**：仍需用mask保证子块不越界，other=0.0填充
- **典型收益**：将stride=H*BT非合并访存变为stride=1合并访存+寄存器转置，可显著降低aiv_mte2_time

### 代码/骨架出处
- `Triton`:
```python
# 反模式：转置 block_ptr，非合并访存
p_dAqk = tl.make_block_ptr(dAqk_local, (BT, T_local), (1, H*BT),
                         (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))

# 正模式：行优先加载 + 寄存器转置
row_idx = i_t * BT + i_j * BC + o_i[:, None]
col_idx = i_i * BC + o_i[None, :]
dAqk_mask = (row_idx < T_local)[:, None] & (col_idx < BT)[None, :]
p_dAqk_row = dAqk_local + row_idx * H * BT + col_idx
b_dAqk_row = tl.load(p_dAqk_row, mask=dAqk_mask, other=0.0)
b_dAqk = tl.trans(b_dAqk_row)
```

**坏实践（反例）**：用转置视角的 block_ptr 直接读子块
```python
p = tl.make_block_ptr(x, (BT, T), (1, H*BT), (i*BC, j*BC), (BC, BC), (0, 1))
b = tl.load(p, boundary_check=(0, 1))  # 列方向 stride=H*BT 跳跃读取
```
后果：列方向每次跳 H*BT 个元素（H=32、BT=64 时 stride=2048），GM→UB 访存极不连续，MTE2 带宽利用率低。✅ 正解：按行优先布局连续加载 BC×BC 子块（stride=1 合并访存），再用 `tl.trans` 在寄存器内转置，边界用 mask+other=0.0 兜底。

**已知实例**：general_perf §raw (`experiences/general_perf/exp_perf_015.yaml`)
- 声称性能收益：30%
- 备注：此方法要求原始张量布局允许按期望方向连续读取。如果原始数据布局本身就是列优先，可能需要预先在host端转置或在kernel内使用其他数据编排策略。

<!-- okf:related:start -->

# 相关

- 相关主题: [块大小选型：2 的幂次 128/256/512 起调，blockSize×dtypeBytes 对齐 32B 且不超 UB](../autotuning/exp_perf_001.md) — 主题相关：共享 general, general_perf
- 相关主题: [autotune 配置空间须覆盖 num_warps(2/4/8)×num_stages(2/3/4) 与 tile 组合](../autotuning/exp_perf_017.md) — 主题相关：共享 general, general_perf
- 相关主题: [性能避坑: 避免多维 stride 访问](../compute-pipeline/exp_perf_007.md) — 主题相关：共享 general, general_perf
- 相关主题: [算子拆分优化: 按计算阶段拆分为简单 kernel](../compute-pipeline/exp_perf_010.md) — 主题相关：共享 general_perf
- 相关主题: [标量消除: i32 比较转 fp32 走向量单元](../compute-pipeline/exp_perf_012.md) — 主题相关：共享 general, general_perf
- 相关主题: [标量消除: min() 运行时分支改 tl.where](../compute-pipeline/exp_perf_013.md) — 主题相关：共享 general, general_perf
- 相关主题: [标量消除: tl.cdiv 编译期循环边界](../compute-pipeline/exp_perf_014.md) — 主题相关：共享 general, general_perf
- 相关主题: [tl.trans 标量化灾难与寄存器重排替代（split/interleave/reshape 安全集）](../compute-pipeline/triton-ascend-trans-scalarization.md) — 同属优化知识，主题相关
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
- 相关主题: [内存访问优化：2D 张量块指针访存（make_block_ptr）](exp_perf_002.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存访问优化：非连续张量连续化 + 一维遍历](exp_perf_003.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存访问优化：stride 参数一律传 tensor.stride() 原生值](exp_perf_004.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存访问优化：contiguous() 一次性拷贝替代 stride 访问](exp_perf_006.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存初始化优化：输出张量创建优选 torch.empty_like](exp_perf_008.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [冗余操作优化：同形 element-wise 算子免 reshape](exp_perf_009.md) — 主题相关：共享 general_perf
- 相关主题: [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](exp_perf_011.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存访问优化：跨阶段寄存器复用消除重复 load](exp_perf_016.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存初始化优化：输出张量创建 API 选择](exp_tensor_create_001.md) — 主题相关：共享 general, 访存效率

<!-- okf:related:end -->
