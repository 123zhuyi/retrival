---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: 大 shape 张量连续分块访存（缩小 grid、单核处理连续段）
description: 大 shape 场景下缩小 grid 尺寸、让每个核处理连续一段数据以提升缓存命中率，收益 12%
tags: [访存效率, 通用优化, general_ascend]
confidence: provisional
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 大 shape 张量连续分块访存

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3/910b4，元素数 ≥1e6 的 1d/2d/3d 大张量；失效边界：小 shape（<1e6）时 grid 收缩带来的循环调度开销可能抵消缓存收益；gather/scatter 等地址不连续场景不适用）

### 原则
- **适用场景**：适用于对内存连续性有要求的大shape场景（需优化缓存命中率）
- **核心逻辑**：缩小grid尺寸，让每个grid处理连续的一段数据，提升内存访问的缓存友好性
- **计算规则**：ELEMENTS_PER_GRID需向上取整到BLOCK_SIZE的倍数，确保循环能完整处理所有数据
- **grid计算**：实际grid数量 = triton.cdiv(总元素数, 每个grid处理的元素数)

### 代码/骨架出处
- `Triton`:
```python
import torch_npu

# 大shape连续分块处理kernel
@triton.jit
def large_vector_kernel(
    input_ptr, output_ptr, n_elements,
    blockSize: tl.constexpr,
    ELEMENTS_PER_GRID: tl.constexpr,
):
    pid = tl.program_id(0)
    # 计算当前grid负责的连续数据范围
    grid_start = pid * ELEMENTS_PER_GRID
    grid_end = min(grid_start + ELEMENTS_PER_GRID, n_elements)
    
    # 分块处理连续数据（缓存友好）
    for block_start in range(grid_start, grid_end, blockSize):
        offsets = block_start + tl.arange(0, blockSize)
        mask = offsets < grid_end
        
        data = tl.load(input_ptr + offsets, mask=mask)
        result = data * 2 + 1  # 示例计算逻辑
        tl.store(output_ptr + offsets, result, mask=mask)

# 连续分块处理启动逻辑
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
        except:
            self.VEC_CORE_NUM = 40

    def forward(self, input_tensor):
        n_elements = input_tensor.numel()
        blockSize = 256  # 适配UB大小的最优值
        MAX_GRID_SIZE = self.VEC_CORE_NUM
        
        # 计算每个grid处理的元素数（向上取整到BLOCK_SIZE倍数）
        ELEMENTS_PER_GRID = triton.cdiv(n_elements, MAX_GRID_SIZE)
        ELEMENTS_PER_GRID = triton.cdiv(ELEMENTS_PER_GRID, blockSize) * blockSize
        
        # 计算实际grid数量
        grid_size = triton.cdiv(n_elements, ELEMENTS_PER_GRID)
        
        output_tensor = torch.empty_like(input_tensor)
        large_vector_kernel[grid_size,](
            input_tensor, output_tensor, n_elements,
            blockSize=blockSize,
            ELEMENTS_PER_GRID=ELEMENTS_PER_GRID,
        )
        return output_tensor
```

**坏实践（反例）**：grid-stride 交错循环——每个 grid 以 `pid + i * num_grids` 的步长跳着访问全局内存
```python
for i in range(pid, num_blocks, num_grids):
    offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 各核访问地址互相交错
```
后果：同一时刻各核访问的地址分散，GM→UB 的缓存局部性差，内存密集型场景损失本卡约 12% 的收益。✅ 正解：缩小 grid 至核数规模，每核用内层循环顺序处理 `pid * ELEMENTS_PER_GRID` 起始的一段连续数据。

**已知实例**：general_ascend §raw (`experiences/general_ascend/exp_ascend_large_shape_block_001.yaml`)
- 声称性能收益：12%
- 备注：该方案通过连续内存访问提升缓存命中率，相比交错循环方案，内存密集型算子性能提升12%；优先用于大向量element-wise、连续分块matmul等对内存连续性敏感的场景

<!-- okf:related:start -->

# 相关

- 相关主题: [内存访问三查：mask 防越界、stride 匹配布局、索引上限判断](../meta-skills/exp_debug_001.md) — 主题相关：共享 通用优化
- 相关主题: [控制流避坑：禁 return/break/continue，复杂条件分步 mask 组合](../meta-skills/exp_debug_002.md) — 主题相关：共享 通用优化
- 相关主题: [Grid/Block 配置校验：乘积 ≤65535、≤3 维 tuple、大 shape 交错循环](../meta-skills/exp_debug_003.md) — 主题相关：共享 通用优化
- 相关主题: [并发写入必须用原子操作且类型与目标内存严格匹配](../meta-skills/exp_debug_004.md) — 主题相关：共享 通用优化
- 相关主题: [切片与索引语法校验：禁 Python 切片、arange 结果禁 get_element](../meta-skills/exp_debug_005.md) — 主题相关：共享 通用优化
- 相关主题: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../meta-skills/quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
- 相关主题: [形状驱动算子决策树](../meta-skills/shape-driven-operator-decision-tree.md) — 同属优化知识，主题相关
- 相关主题: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](../portability/exp_ascend_blockptr_boundary_scalar_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [移除确定性访存上的冗余 compile_hint（mayDiscretememaccess）](../portability/exp_ascend_compile_hint_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [核心数在 __init__ 获取一次，按算子类型选 VEC/CUBE 核数](../portability/exp_ascend_core_num_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [Grid 必须是 ≤3 维 tuple 且各维乘积 ≤65535](../portability/exp_ascend_grid_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [大 shape 用交错循环：grid 固定为核心数，kernel 内按步长遍历](../portability/exp_ascend_grid_large_shape_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [地址计算全程保持 int32，避免 int64 标量降级](../portability/exp_ascend_int64_address_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [整数比较标量降级：比较操作数转 fp32 或编译期消除](../portability/exp_ascend_int_compare_scalar_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [并行任务分配首选 for i in range(pid, total, core_num) 交错模式](../portability/exp_ascend_parallel_pattern_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [BLOCK_SIZE <65536 且子块内存 ≤L1 缓存，超限按维度多次嵌套切分](../portability/exp_ascend_split_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [禁止用 tl.where 动态计算内存偏移，改 if-else 静态分支](../portability/exp_ascend_where_001.md) — 主题相关：共享 general_ascend
- 相关主题: [tl.constexpr 仅限内核参数声明，Host 侧禁用](../portability/exp_constexpr_001.md) — 主题相关：共享 general_ascend
- 相关主题: [标量类型转换仅用 .to(type)，禁用类型构造函数](../portability/exp_scalar_cast_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [禁用 Python 切片语法，改用 get_element/extract_slice/insert_slice](../portability/exp_slice_001.md) — 主题相关：共享 general_ascend
- 相关主题: [禁止对 tl.arange 索引表达式调用 get_element](../portability/exp_slice_002.md) — 主题相关：共享 general_ascend
- 相关主题: [Memory Access Optimization: kv tile reuse](exp_attn_bwd_hstu_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: vectorized offset generation](exp_attn_causal_mask_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fp8 value casting](exp_attn_fp8_ub_slice_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sparse gather optimization](exp_attn_sparse_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: flattened grid scheduling](exp_causal_conv1d_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_causal_conv1d_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sliding window register reuse](exp_causal_conv1d_prefill_002.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: compile time branching](exp_causal_conv1d_states_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused state update](exp_causal_conv1d_update_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: make block ptr hardware acceleration](exp_causal_conv1d_update_003.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: overlapping t tiling](exp_causal_conv1d_update_004.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: 2D Tiling Strategy](exp_embedding_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: simd index select](exp_flashattn_gqa_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_fused_add_layernorm_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_fused_norm_rope_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: Tiled Two-Pass Reduction](exp_gn_backward_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: dynamic vectorization tuning](exp_groupnorm_001.md) — 主题相关：共享 访存效率
- 相关主题: [内存访问优化：转置 load 行优先化与跨阶段复用](exp_kda_bwd_intra_memory_opt_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: adaptive tiling strategy](exp_layernorm_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion](exp_lightning_attn_decode_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache](exp_mla_paged_attn_001.md) — 主题相关：共享 访存效率
- 相关主题: [内存访问优化：2D 张量块指针访存（make_block_ptr）](exp_perf_002.md) — 主题相关：共享 访存效率
- 相关主题: [内存访问优化：非连续张量连续化 + 一维遍历](exp_perf_003.md) — 主题相关：共享 访存效率
- 相关主题: [内存访问优化：stride 参数一律传 tensor.stride() 原生值](exp_perf_004.md) — 主题相关：共享 访存效率
- 相关主题: [内存初始化优化：输出张量创建优选 torch.empty_like](exp_perf_008.md) — 主题相关：共享 访存效率
- 相关主题: [内存访问优化：行优先连续加载 + tl.trans 寄存器转置](exp_perf_015.md) — 主题相关：共享 访存效率
- 相关主题: [内存访问优化：跨阶段寄存器复用消除重复 load](exp_perf_016.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: single-pass tiling via UB residency](exp_rmsnorm_grad_dx_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: grid-stride scheduling with two-pass reduction](exp_rmsnorm_infer_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: preload extract slice eliminate hbm read](exp_solve_tril_002.md) — 主题相关：共享 访存效率
- 相关主题: [内存初始化优化：输出张量创建 API 选择](exp_tensor_create_001.md) — 主题相关：共享 general_ascend, 访存效率

<!-- okf:related:end -->
