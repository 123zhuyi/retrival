---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: portability
dimension: 4.3
title: BLOCK_SIZE <65536 且子块内存 ≤L1 缓存，超限按维度多次嵌套切分
description: 910b2/910b3 上 BLOCK_SIZE 必须 <65536，且线程块内存须 ≤L1 缓存（910b2 32KB / 910b3 64KB）；超限时按维度多次嵌套切分（先 M 后 N），声称收益 20%
tags: [跨芯片适配, 通用优化, general_ascend]
confidence: provisional
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## BLOCK_SIZE <65536 且子块内存 ≤L1 缓存，超限按维度多次嵌套切分

【优化维度】：跨芯片适配
【适用条件】：平台 910b2/910b3 ｜ triton-ascend 版本 未标注（按验证版本）｜ dtype 全部 ｜ shape 任意（nd），大 shape 触发
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：单次切分已满足 BLOCK_SIZE <65536 且子块内存 ≤L1 时无需嵌套——多余循环层只增控制开销；L1 容量因机型而异（910b2 32KB / 910b3 64KB），切分上界须按目标机型计算；切分后子块仍需满足对齐要求（如 32B）；910C/310P 缓存规格未验证（unverified)）

### 原则
- **硬性限制**：Ascend后端BLOCK_SIZE必须严格小于65536，且线程块占用的总内存需符合硬件缓存限制（910b2 L1缓存32KB、910b3 L1缓存64KB）
- **超限处理**：若张量shape过大导致单次切分超硬件缓存，或BLOCK_SIZE触达65536上限，需对循环进行**多次嵌套切分**（先按行/列拆分子块，再逐块处理）
- **切分原则**：多次切分优先按维度拆分（如先拆M维度，再拆N维度），保证每个子块内存占用≤硬件缓存上限

### 代码/骨架出处
- `Triton`:
```python
# Ascend后端多次切分示例（BLOCK_SIZE超限场景）
@triton.jit
def multi_split_kernel(
    input_ptr, output_ptr, M, N,
    stride_m, stride_n,
    blockM: tl.constexpr=128,  # <65536 符合硬件限制
    blockN: tl.constexpr=128,
    MAX_SPLITS: tl.constexpr=100  # 多次切分上限
):
    pid = tl.program_id(0)
    # 一级切分：按M维度拆分子块（避免单次切分超缓存）
    for split_m in range(pid, M, MAX_SPLITS):
        if split_m >= M:
            break
        # 二级切分：按N维度拆分子块（符合BLOCK_SIZE限制）
        for split_n in range(0, N, blockN):
            col_offsets = split_n + tl.arange(0, blockN)
            mask = col_offsets < N
            # 加载当前子块数据（内存占用符合硬件限制）
            row_ptr = input_ptr + split_m * stride_m
            data = tl.load(row_ptr + col_offsets * stride_n, mask=mask)
            # 计算并存储
            result = data * 2
            out_row_ptr = output_ptr + split_m * stride_m
            tl.store(out_row_ptr + col_offsets * stride_n, result, mask=mask)
```

**坏实践（反例）**：910b2/910b3平台下，blockSize≥65536会直接编译失败，线程块内存超缓存会导致访存效率下降50%以上

**已知实例**：general_ascend §raw (`experiences/general_ascend/exp_ascend_split_001.yaml`)
- 声称性能收益：20%
- 备注：910b2/910b3平台下，blockSize≥65536会直接编译失败，线程块内存超缓存会导致访存效率下降50%以上；多次切分可在符合限制的前提下充分利用硬件算力

<!-- okf:related:start -->

# 相关

- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](../memory-sram/exp_ascend_large_shape_block_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [内存初始化优化：输出张量创建 API 选择](../memory-sram/exp_tensor_create_001.md) — 主题相关：共享 general_ascend
- 相关主题: [Tiling 预算与布局驱动的块大小选择](../memory-sram/tiling-budget-and-layout-driven-selection.md) — 同属优化知识，主题相关
- 相关主题: [内存访问三查：mask 防越界、stride 匹配布局、索引上限判断](../meta-skills/exp_debug_001.md) — 主题相关：共享 通用优化
- 相关主题: [控制流避坑：禁 return/break/continue，复杂条件分步 mask 组合](../meta-skills/exp_debug_002.md) — 主题相关：共享 通用优化
- 相关主题: [Grid/Block 配置校验：乘积 ≤65535、≤3 维 tuple、大 shape 交错循环](../meta-skills/exp_debug_003.md) — 主题相关：共享 通用优化
- 相关主题: [并发写入必须用原子操作且类型与目标内存严格匹配](../meta-skills/exp_debug_004.md) — 主题相关：共享 通用优化
- 相关主题: [切片与索引语法校验：禁 Python 切片、arange 结果禁 get_element](../meta-skills/exp_debug_005.md) — 主题相关：共享 通用优化
- 相关主题: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](exp_ascend_blockptr_boundary_scalar_001.md) — 主题相关：共享 general_ascend, 跨芯片适配, 通用优化
- 相关主题: [移除确定性访存上的冗余 compile_hint（mayDiscretememaccess）](exp_ascend_compile_hint_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [核心数在 __init__ 获取一次，按算子类型选 VEC/CUBE 核数](exp_ascend_core_num_001.md) — 主题相关：共享 general_ascend, 跨芯片适配, 通用优化
- 相关主题: [Grid 必须是 ≤3 维 tuple 且各维乘积 ≤65535](exp_ascend_grid_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [大 shape 用交错循环：grid 固定为核心数，kernel 内按步长遍历](exp_ascend_grid_large_shape_001.md) — 主题相关：共享 general_ascend, 跨芯片适配, 通用优化
- 相关主题: [地址计算全程保持 int32，避免 int64 标量降级](exp_ascend_int64_address_001.md) — 主题相关：共享 general_ascend, 跨芯片适配, 通用优化
- 相关主题: [整数比较标量降级：比较操作数转 fp32 或编译期消除](exp_ascend_int_compare_scalar_001.md) — 主题相关：共享 general_ascend, 跨芯片适配, 通用优化
- 相关主题: [并行任务分配首选 for i in range(pid, total, core_num) 交错模式](exp_ascend_parallel_pattern_001.md) — 主题相关：共享 general_ascend, 跨芯片适配, 通用优化
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
