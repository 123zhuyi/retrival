---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: portability
dimension: 4.3
title: 地址计算全程保持 int32，避免 int64 标量降级
description: 910b2/910b3 上 int64 通用算术（VAddOp/VMulOp 等）全部降级为标量循环，地址偏移中显式/隐式 int64 会显著拉高 scalar_ratio；确认偏移乘积 <2^31 后移除所有 .to(tl.int64)，声称收益 50%
tags: [跨芯片适配, 通用优化, general_ascend]
confidence: provisional
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 地址计算全程保持 int32，避免 int64 标量降级

【优化维度】：跨芯片适配
【适用条件】：平台 910b2/910b3 ｜ triton-ascend 版本 未标注（按验证版本）｜ dtype 全部（针对地址/索引计算）｜ shape 任意（nd）
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：地址偏移乘积可能 ≥2^31 时（如 B×T×T、超大张量扁平偏移）严禁改 int32——会溢出产生错误结果，此时应分块缩小单 kernel 偏移范围而非硬用 i64；仅 910b2/910b3 验证，910C/310P 未验证（unverified)）

### 原则
- **降级规则**：Ascend 910B上int64的通用算术操作（VAddOp, VMulOp等）全部降级为标量循环
- **常见陷阱**：地址计算中显式或隐式使用int64（如(i_b * NT + i_t).to(tl.int64), bos/eos用int64），导致每次地址偏移都走Scalar单元
- **解决方案**：保持地址计算全程int32。典型场景B=1~8, T=1024~8192时，B*T*T等乘积不超过i32范围（2^31≈21亿）
- **验证方法**：检查所有.to(tl.int64)调用，确认是否可以移除或替换为int32
- **典型收益**：消除所有i64地址计算的标量降级，可显著降低aic_scalar_ratio和aiv_scalar_ratio

### 代码/骨架出处
- `Triton`:
```python
# 反模式：i64 地址计算
i_tg = (i_b * NT + i_t).to(tl.int64)
bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)
h += (i_tg * H + i_h) * K * V

# 正模式：保持 i32
i_tg = i_b * NT + i_t
bos = i_b * T
eos = i_b * T + T
h += (i_tg * H + i_h) * K * V
```

**坏实践（反例）**：对超大规模张量（如T>65535且B较大），可能需要分块或其他策略避免i64

**已知实例**：general_ascend §raw (`experiences/general_ascend/exp_ascend_int64_address_001.yaml`)
- 声称性能收益：50%
- 备注：在确认i32范围足够的前提下使用。对超大规模张量（如T>65535且B较大），可能需要分块或其他策略避免i64。

<!-- okf:related:start -->

# 相关

- 实践案例: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](../../ops/moe-compute-expert-tokens.md) — 同属优化知识，主题相关
- 相关主题: [整数比较/加法必须转 float32，避免 Vector 标量降级](../compute-pipeline/int-compare-cast-float32.md) — 同属优化知识，主题相关
- 相关主题: [kernel 内索引必须转 int32，避免 int64 降级](../compute-pipeline/int32-index-cast.md) — 同属优化知识，主题相关
- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](../memory-sram/exp_ascend_large_shape_block_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [内存初始化优化：输出张量创建 API 选择](../memory-sram/exp_tensor_create_001.md) — 主题相关：共享 general_ascend
- 相关主题: [内存访问三查：mask 防越界、stride 匹配布局、索引上限判断](../meta-skills/exp_debug_001.md) — 主题相关：共享 通用优化
- 相关主题: [控制流避坑：禁 return/break/continue，复杂条件分步 mask 组合](../meta-skills/exp_debug_002.md) — 主题相关：共享 通用优化
- 相关主题: [Grid/Block 配置校验：乘积 ≤65535、≤3 维 tuple、大 shape 交错循环](../meta-skills/exp_debug_003.md) — 主题相关：共享 通用优化
- 相关主题: [并发写入必须用原子操作且类型与目标内存严格匹配](../meta-skills/exp_debug_004.md) — 主题相关：共享 通用优化
- 相关主题: [切片与索引语法校验：禁 Python 切片、arange 结果禁 get_element](../meta-skills/exp_debug_005.md) — 主题相关：共享 通用优化
- 相关主题: [if/else 分支 dtype 不一致：改 masked load，bool 张量 host 侧转 int32](../meta-skills/exp_debug_triton_branch_type_mismatch_001.md) — 同属优化知识，主题相关
- 相关主题: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](exp_ascend_blockptr_boundary_scalar_001.md) — 主题相关：共享 general_ascend, 跨芯片适配, 通用优化
- 相关主题: [移除确定性访存上的冗余 compile_hint（mayDiscretememaccess）](exp_ascend_compile_hint_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [核心数在 __init__ 获取一次，按算子类型选 VEC/CUBE 核数](exp_ascend_core_num_001.md) — 主题相关：共享 general_ascend, 跨芯片适配, 通用优化
- 相关主题: [Grid 必须是 ≤3 维 tuple 且各维乘积 ≤65535](exp_ascend_grid_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [大 shape 用交错循环：grid 固定为核心数，kernel 内按步长遍历](exp_ascend_grid_large_shape_001.md) — 主题相关：共享 general_ascend, 跨芯片适配, 通用优化
- 相关主题: [整数比较标量降级：比较操作数转 fp32 或编译期消除](exp_ascend_int_compare_scalar_001.md) — 主题相关：共享 general_ascend, 跨芯片适配, 通用优化
- 相关主题: [并行任务分配首选 for i in range(pid, total, core_num) 交错模式](exp_ascend_parallel_pattern_001.md) — 主题相关：共享 general_ascend, 跨芯片适配, 通用优化
- 相关主题: [BLOCK_SIZE <65536 且子块内存 ≤L1 缓存，超限按维度多次嵌套切分](exp_ascend_split_001.md) — 主题相关：共享 general_ascend, 跨芯片适配, 通用优化
- 相关主题: [禁止用 tl.where 动态计算内存偏移，改 if-else 静态分支](exp_ascend_where_001.md) — 主题相关：共享 general_ascend
- 相关主题: [Cube/Vector 流水线编排 + 序列分半（down kernel）适配片上内存](exp_attn_bwd_q_down_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [Ascend 专用 tiling 配置：tile_mix cube/vector loop + multibuffer + HIVM CV 自动均衡](exp_attn_sda_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [tl.constexpr 仅限内核参数声明，Host 侧禁用](exp_constexpr_001.md) — 主题相关：共享 general_ascend
- 相关主题: [标签索引转 int32 比较 + GROUP_SIZE_M 分组调度 + 大 K tile](exp_entropy_backward_npu_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [反向 KV kernel：HIVM CV 均衡 + 双缓冲 + 非对称 tile（128×64）+ 序列分半](exp_flashattn_bwd_kv_ascend_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [异构并行融合：MatMul 派 Cube、归一化+位置编码派 Vector，dl.async_task 异步掩盖](exp_fused_matmul_rms_rope_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [GQA 分页 KV：tl.parallel bind_sub_block 双向量核并行 + KV 按组复用 + block_table 遍历](exp_gqa_paged_npu_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [UB tiling（BLOCK_Q×blockK 分块）+ tl.math.fma 复数乘法 + tl.interleave 交错布局](exp_rope_llama4_npu_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [标量类型转换仅用 .to(type)，禁用类型构造函数](exp_scalar_cast_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [反向 KV 梯度 kernel：HIVM CV 均衡 + tile_mix=2 + 双缓冲 + 256×64 tile + bf16 存储 fp32 累加](exp_sda_bwd_kv_ascend_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [前向 SDA kernel：CV 均衡 + 双缓冲 + bf16 乘 fp32 累加 + 64×256 tile 分相处理](exp_sda_fwd_ascend_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [禁用 Python 切片语法，改用 get_element/extract_slice/insert_slice](exp_slice_001.md) — 主题相关：共享 general_ascend
- 相关主题: [禁止对 tl.arange 索引表达式调用 get_element](exp_slice_002.md) — 主题相关：共享 general_ascend
- 相关主题: [稀疏注意力 prefill：tle.index_select_simd 按需取 Top-K + cube1 加速 + 工作集 tile=64](exp_sparse_flash_attn_prefill_001.md) — 主题相关：共享 跨芯片适配

<!-- okf:related:end -->
