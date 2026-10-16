---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: portability
dimension: 4.3
title: 大 shape 用交错循环：grid 固定为核心数，kernel 内按步长遍历
description: 910b2/910b3 上元素数 ≥1e6 的大 shape 算子，grid 固定为硬件核心数、kernel 内 for row in range(pid, M, CORE_NUM) 交错处理，负载均衡最优且规避 grid 乘积 ≤65535 限制，声称收益 15%
tags: [跨芯片适配, 通用优化, general_ascend]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 大 shape 用交错循环：grid 固定为核心数，kernel 内按步长遍历

【优化维度】：跨芯片适配
【适用条件】：平台 910b2/910b3 ｜ triton-ascend 版本 未标注（按验证版本）｜ dtype 全部 ｜ shape 2d/3d，元素数 ≥1e6
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：仅适用于按行/按块独立处理的场景（行间无依赖）；小 shape（元素数 <1e6）直接切分 grid 更简单，交错循环的循环控制开销可能得不偿失；行处理代价差异大（如变长行）时交错均衡退化为按步长取模，尾部倾斜仍存在；910C/310P 未验证（unverified)）

### 原则
- **核心策略**：大shape算子优先使用**交错循环处理**（强烈推荐），将Grid固定为硬件核心数，每个核心以步长方式交错处理数据
- **优势特性**：代码简洁、负载均衡最优（每个核心处理的数据量基本一致），完全规避Grid乘积超限问题
- **适配场景**：适用于按行/按块独立处理的场景（如逐行计算、分块矩阵运算）

### 代码/骨架出处
- `Triton`:
```python
import torch_npu

# 大shape张量交错循环处理核心kernel
@triton.jit
def row_processing_kernel(
    input_ptr, output_ptr, 
    M, N,
    stride_m, stride_n,
    blockN: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # 核心优化：交错处理（每个核心处理 pid, pid+CORE_NUM, pid+2*CORE_NUM... 行）
    # 负载均衡：所有行被恰好处理一次，无核心闲置
    for row_idx in range(pid, M, CORE_NUM):
        # 计算当前行指针偏移
        row_ptr = input_ptr + row_idx * stride_m
        out_row_ptr = output_ptr + row_idx * stride_m
        
        # 列维度分块处理（符合BLOCK_SIZE限制）
        for col_start in range(0, N, blockN):
            col_offsets = col_start + tl.arange(0, blockN)
            mask = col_offsets < N
            
            data = tl.load(row_ptr + col_offsets * stride_n, mask=mask)
            result = data * 2 + 1  # 示例计算逻辑
            tl.store(out_row_ptr + col_offsets * stride_n, result, mask=mask)

# 大shape算子启动逻辑
class LargeShapeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 获取Ascend硬件VEC核心数（适配不同机型）
        try:
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 48)
        except:
            self.VEC_CORE_NUM = 48  # Ascend 910B2默认值：24个AI Core × 2个VEC/Core

    def forward(self, input_tensor):
        M, N = input_tensor.shape  # M可能超大（如327680）
        output_tensor = torch.empty_like(input_tensor)
        
        # Grid固定为核心数，规避乘积超限
        grid = (self.VEC_CORE_NUM,)
        
        row_processing_kernel[grid](
            input_tensor, output_tensor,
            M, N,
            input_tensor.stride(0), input_tensor.stride(1),
            blockN=256,
            CORE_NUM=self.VEC_CORE_NUM,
        )
        return output_tensor
```

**坏实践（反例）**：大 shape 直接按任务数切 grid（如 `grid=(M,)`，M 达数十万）→ 超过 grid 乘积 ≤65535 限制直接启动失败，或未超限但调度开销大、尾部负载不均 → ✅ 正解即本条：`grid=(CORE_NUM,)` + kernel 内 `range(pid, M, CORE_NUM)` 交错遍历

**已知实例**：general_ascend §raw (`experiences/general_ascend/exp_ascend_grid_large_shape_001.yaml`)
- 声称性能收益：15%
- 备注：该策略是大shape算子（元素数≥1e6）的最优解，相比直接切分Grid的方案，负载均衡提升15%；Ascend 910B3默认VEC核心数为64，可按需调整

<!-- okf:related:start -->

# 相关

- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](../memory-sram/exp_ascend_large_shape_block_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [内存初始化优化：输出张量创建 API 选择](../memory-sram/exp_tensor_create_001.md) — 主题相关：共享 general_ascend
- 相关主题: [内存访问三查：mask 防越界、stride 匹配布局、索引上限判断](../meta-skills/exp_debug_001.md) — 主题相关：共享 通用优化
- 相关主题: [控制流避坑：禁 return/break/continue，复杂条件分步 mask 组合](../meta-skills/exp_debug_002.md) — 主题相关：共享 通用优化
- 相关主题: [Grid/Block 配置校验：乘积 ≤65535、≤3 维 tuple、大 shape 交错循环](../meta-skills/exp_debug_003.md) — 主题相关：共享 通用优化
- 相关主题: [并发写入必须用原子操作且类型与目标内存严格匹配](../meta-skills/exp_debug_004.md) — 主题相关：共享 通用优化
- 相关主题: [切片与索引语法校验：禁 Python 切片、arange 结果禁 get_element](../meta-skills/exp_debug_005.md) — 主题相关：共享 通用优化
- 相关主题: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../meta-skills/quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
- 相关主题: [形状驱动算子决策树](../meta-skills/shape-driven-operator-decision-tree.md) — 同属优化知识，主题相关
- 相关主题: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](exp_ascend_blockptr_boundary_scalar_001.md) — 主题相关：共享 general_ascend, 跨芯片适配, 通用优化
- 相关主题: [移除确定性访存上的冗余 compile_hint（mayDiscretememaccess）](exp_ascend_compile_hint_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [核心数在 __init__ 获取一次，按算子类型选 VEC/CUBE 核数](exp_ascend_core_num_001.md) — 主题相关：共享 general_ascend, 跨芯片适配, 通用优化
- 相关主题: [Grid 必须是 ≤3 维 tuple 且各维乘积 ≤65535](exp_ascend_grid_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [地址计算全程保持 int32，避免 int64 标量降级](exp_ascend_int64_address_001.md) — 主题相关：共享 general_ascend, 跨芯片适配, 通用优化
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
