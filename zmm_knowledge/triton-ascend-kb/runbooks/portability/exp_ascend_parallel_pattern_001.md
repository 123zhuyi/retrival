---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: portability
dimension: 4.3
title: 并行任务分配首选 for i in range(pid, total, core_num) 交错模式
description: 910b2/910b3/910b4 上总任务数远大于核心数时，kernel 内 for i in range(pid, total_items, core_num) 是最优并行分配：负载差 ≤1、无需手算每核配额、适配任意 shape，声称收益 10%
tags: [跨芯片适配, 通用优化, general_ascend]
confidence: provisional
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 并行任务分配首选 for i in range(pid, total, core_num) 交错模式

【优化维度】：跨芯片适配
【适用条件】：平台 910b2/910b3/910b4 ｜ triton-ascend 版本 未标注（按验证版本）｜ dtype 全部 ｜ shape 任意（nd），总任务数 ≥ 核心数
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 910b4；失效边界：总任务数小于核心数时交错无意义（部分核空转），应减小 grid 或合并任务；任务间存在依赖/需全局同步时不适用；与 exp_ascend_grid_large_shape_001 同源——本卡是通用分配模式，那张是大 shape 特化；910C/310P 未验证（unverified)）

### 原则
- **核心模式**：for i in range(pid, total_items, core_num)` 是Ascend后端最优并行工作分配模式
- **核心优势**：代码简洁、负载天然均衡（各核心任务数差≤1）、无需计算ELEMENTS_PER_GRID等复杂参数、适配任意大小输入shape
- **场景适配**：VEC核心用于逐行独立处理算子（逐行softmax/归一化/reduce/element-wise）；CUBE核心用于矩阵乘法类算子（matmul/attention）

### 代码/骨架出处
- `Triton`:
```python
@triton.jit
def parallel_assign_kernel(
    input_ptr, output_ptr, M, N,
    stride_m, stride_n,
    blockN: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    pid = tl.program_id(0)
    # 经典并行分配模式: 每个核心处理 pid, pid+CORE_NUM, pid+2*CORE_NUM... 行
    for row_idx in range(pid, M, CORE_NUM):
        row_ptr = input_ptr + row_idx * stride_m
        out_row_ptr = output_ptr + row_idx * stride_m
        
        for col_start in range(0, N, blockN):
            col_offsets = col_start + tl.arange(0, blockN)
            mask = col_offsets < N
            data = tl.load(row_ptr + col_offsets * stride_n, mask=mask)
            result = tl.nn.softmax(data, axis=0)  # 逐行softmax（VEC核心适配场景）
            tl.store(out_row_ptr + col_offsets * stride_n, result, mask=mask)
```

**坏实践（反例）**：手动计算 `ELEMENTS_PER_GRID = cdiv(total, core_num)`、每核处理连续区间 `[pid*EPG, (pid+1)*EPG)` → total 不整除时尾核越界需额外 mask，且代码复杂；任务代价不均时连续分块倾斜更严重 → ✅ 正解即本条：交错取模天然均衡（各核任务数差 ≤1）且代码量少约 30%

**已知实例**：general_ascend §raw (`experiences/general_ascend/exp_ascend_parallel_pattern_001.yaml`)
- 声称性能收益：10%
- 备注：该模式相比手动分块的并行方案，负载均衡性提升10%，且代码量减少30%；适配所有总任务数远大于核心数的大shape场景

<!-- okf:related:start -->

# 相关

- 相关主题: [Scheduling Optimization: pid grouping strategy](../compute-pipeline/exp_mm_grouped_reorder_001.md) — 同属优化知识，主题相关
- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](../memory-sram/exp_ascend_large_shape_block_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [内存初始化优化：输出张量创建 API 选择](../memory-sram/exp_tensor_create_001.md) — 主题相关：共享 general_ascend
- 相关主题: [内存访问三查：mask 防越界、stride 匹配布局、索引上限判断](../meta-skills/exp_debug_001.md) — 主题相关：共享 通用优化
- 相关主题: [控制流避坑：禁 return/break/continue，复杂条件分步 mask 组合](../meta-skills/exp_debug_002.md) — 主题相关：共享 通用优化
- 相关主题: [Grid/Block 配置校验：乘积 ≤65535、≤3 维 tuple、大 shape 交错循环](../meta-skills/exp_debug_003.md) — 主题相关：共享 通用优化
- 相关主题: [并发写入必须用原子操作且类型与目标内存严格匹配](../meta-skills/exp_debug_004.md) — 主题相关：共享 通用优化
- 相关主题: [切片与索引语法校验：禁 Python 切片、arange 结果禁 get_element](../meta-skills/exp_debug_005.md) — 主题相关：共享 通用优化
- 相关主题: [静态上限循环：while 改 for range（上限须为 tl.constexpr）](exp_api_003.md) — 同属优化知识，主题相关
- 相关主题: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](exp_ascend_blockptr_boundary_scalar_001.md) — 主题相关：共享 general_ascend, 跨芯片适配, 通用优化
- 相关主题: [移除确定性访存上的冗余 compile_hint（mayDiscretememaccess）](exp_ascend_compile_hint_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [核心数在 __init__ 获取一次，按算子类型选 VEC/CUBE 核数](exp_ascend_core_num_001.md) — 主题相关：共享 general_ascend, 跨芯片适配, 通用优化
- 相关主题: [Grid 必须是 ≤3 维 tuple 且各维乘积 ≤65535](exp_ascend_grid_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [大 shape 用交错循环：grid 固定为核心数，kernel 内按步长遍历](exp_ascend_grid_large_shape_001.md) — 主题相关：共享 general_ascend, 跨芯片适配, 通用优化
- 相关主题: [地址计算全程保持 int32，避免 int64 标量降级](exp_ascend_int64_address_001.md) — 主题相关：共享 general_ascend, 跨芯片适配, 通用优化
- 相关主题: [整数比较标量降级：比较操作数转 fp32 或编译期消除](exp_ascend_int_compare_scalar_001.md) — 主题相关：共享 general_ascend, 跨芯片适配, 通用优化
- 相关主题: [BLOCK_SIZE <65536 且子块内存 ≤L1 缓存，超限按维度多次嵌套切分](exp_ascend_split_001.md) — 主题相关：共享 general_ascend, 跨芯片适配, 通用优化
- 相关主题: [禁止用 tl.where 动态计算内存偏移，改 if-else 静态分支](exp_ascend_where_001.md) — 主题相关：共享 general_ascend
- 相关主题: [Cube/Vector 流水线编排 + 序列分半（down kernel）适配片上内存](exp_attn_bwd_q_down_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [Ascend 专用 tiling 配置：tile_mix cube/vector loop + multibuffer + HIVM CV 自动均衡](exp_attn_sda_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [tl.constexpr 仅限内核参数声明，Host 侧禁用](exp_constexpr_001.md) — 主题相关：共享 general_ascend
- 相关主题: [标签索引转 int32 比较 + GROUP_SIZE_M 分组调度 + 大 K tile](exp_entropy_backward_npu_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [反向 KV kernel：HIVM CV 均衡 + 双缓冲 + 非对称 tile（128×64）+ 序列分半](exp_flashattn_bwd_kv_ascend_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [异构并行融合：MatMul 派 Cube、归一化+位置编码派 Vector，dl.async_task 异步掩盖](exp_fused_matmul_rms_rope_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [GQA 分页 KV：tl.parallel bind_sub_block 双向量核并行 + KV 按组复用 + block_table 遍历](exp_gqa_paged_npu_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [MatMul 调度：对角遍历 grouped_launch + dot_pad_only_k hint + 寄存器原位累加](exp_matmul_ascend_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [波前调度 MatMul：对角遍历 + dot_pad_only_k + autotune tile 长宽比](exp_matmul_ascend_wavefront_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [UB tiling（BLOCK_Q×blockK 分块）+ tl.math.fma 复数乘法 + tl.interleave 交错布局](exp_rope_llama4_npu_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [标量类型转换仅用 .to(type)，禁用类型构造函数](exp_scalar_cast_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [反向 KV 梯度 kernel：HIVM CV 均衡 + tile_mix=2 + 双缓冲 + 256×64 tile + bf16 存储 fp32 累加](exp_sda_bwd_kv_ascend_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [前向 SDA kernel：CV 均衡 + 双缓冲 + bf16 乘 fp32 累加 + 64×256 tile 分相处理](exp_sda_fwd_ascend_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [禁用 Python 切片语法，改用 get_element/extract_slice/insert_slice](exp_slice_001.md) — 主题相关：共享 general_ascend
- 相关主题: [禁止对 tl.arange 索引表达式调用 get_element](exp_slice_002.md) — 主题相关：共享 general_ascend
- 相关主题: [稀疏注意力 prefill：tle.index_select_simd 按需取 Top-K + cube1 加速 + 工作集 tile=64](exp_sparse_flash_attn_prefill_001.md) — 主题相关：共享 跨芯片适配

<!-- okf:related:end -->
