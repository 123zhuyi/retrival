---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: portability
dimension: 4.3
title: 核心数在 __init__ 获取一次，按算子类型选 VEC/CUBE 核数
description: 910b2/910b3/910b4 上 grid 核心数须按算子类型匹配（向量类用 VEC 核数、矩阵类用 CUBE 核数），且在 Module.__init__ 阶段获取一次——forward 中调用 get_device_limit 会触发设备同步，损耗 10%-15%
tags: [跨芯片适配, 通用优化, general_ascend]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 核心数在 __init__ 获取一次，按算子类型选 VEC/CUBE 核数

【优化维度】：跨芯片适配
【适用条件】：平台 910b2/910b3/910b4 ｜ triton-ascend 版本 未标注（按验证版本）｜ dtype 全部 ｜ shape 任意（nd）
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 910b4；失效边界：核心数硬编码后跨机型迁移会失真——910B2 默认 VEC=48/CUBE=24、910B3 VEC=64/CUBE=32、910B4 VEC=40/CUBE=20，必须以 get_device_limit 实测为准、硬编码仅作 try-except 兜底；单 kernel 微基准等无 nn.Module 结构的场景无 __init__ 可用，应在模块加载期而非 kernel 启动路径上获取；910C/310P 未验证（unverified)）

### 原则
- **核心数选择**：按算子类型匹配核心数——向量计算类（element-wise/softmax/归一化）用VEC核心数，矩阵计算类（matmul/attention）用CUBE核心数
- **获取时机**：核心数必须在Module的`__init__`阶段获取，避免在forward中调用导致设备同步开销
- **风险规避**：torch_npu 的import和`get_device_limit`调用会触发设备同步，禁止在forward中执行该操作
- **容错适配**：增加try-except兜底，适配不同Ascend机型的默认核心数（910B4默认VEC=40、CUBE=20）

### 代码/骨架出处
- `Triton`:
```python
import torch_npu

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 核心优化: 在__init__中仅执行一次核心数获取，避免forward同步开销
        try:
            # 向量计算类算子 → VEC核心数
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
            # 矩阵计算类算子 → CUBE核心数
            self.CUBE_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("cube_core_num", 20)
        except:
            # 兜底值: 适配Ascend 910B4（20个AI Core × 2 VEC/Core = 40；20个AI Core ×1 CUBE/Core=20）
            self.VEC_CORE_NUM = 40
            self.CUBE_CORE_NUM = 20
    
    def forward(self, input_tensor):
        # forward中直接使用初始化好的核心数，无同步开销
        grid = (self.VEC_CORE_NUM,)
        # 后续kernel调用逻辑...
        return output_tensor
```

**坏实践（反例）**：该规则无直接性能提升，但可规避forward中设备同步导致的10%-15%性能损耗

**已知实例**：general_ascend §raw (`experiences/general_ascend/exp_ascend_core_num_001.yaml`)
- 声称性能收益：0%
- 备注：该规则无直接性能提升，但可规避forward中设备同步导致的10%-15%性能损耗；910B2默认VEC=48、CUBE=24，910B3默认VEC=64、CUBE=32，可按需调整兜底值

<!-- okf:related:start -->

# 相关

- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](../memory-sram/exp_ascend_large_shape_block_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [内存初始化优化：输出张量创建 API 选择](../memory-sram/exp_tensor_create_001.md) — 主题相关：共享 general_ascend
- 相关主题: [内存访问三查：mask 防越界、stride 匹配布局、索引上限判断](../meta-skills/exp_debug_001.md) — 主题相关：共享 通用优化
- 相关主题: [控制流避坑：禁 return/break/continue，复杂条件分步 mask 组合](../meta-skills/exp_debug_002.md) — 主题相关：共享 通用优化
- 相关主题: [Grid/Block 配置校验：乘积 ≤65535、≤3 维 tuple、大 shape 交错循环](../meta-skills/exp_debug_003.md) — 主题相关：共享 通用优化
- 相关主题: [并发写入必须用原子操作且类型与目标内存严格匹配](../meta-skills/exp_debug_004.md) — 主题相关：共享 通用优化
- 相关主题: [切片与索引语法校验：禁 Python 切片、arange 结果禁 get_element](../meta-skills/exp_debug_005.md) — 主题相关：共享 通用优化
- 相关主题: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](exp_ascend_blockptr_boundary_scalar_001.md) — 主题相关：共享 general_ascend, 跨芯片适配, 通用优化
- 相关主题: [移除确定性访存上的冗余 compile_hint（mayDiscretememaccess）](exp_ascend_compile_hint_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [Grid 必须是 ≤3 维 tuple 且各维乘积 ≤65535](exp_ascend_grid_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [大 shape 用交错循环：grid 固定为核心数，kernel 内按步长遍历](exp_ascend_grid_large_shape_001.md) — 主题相关：共享 general_ascend, 跨芯片适配, 通用优化
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
- 相关主题: [dot_pad_only_k hint + 对角 block 调度 + batch/head 循环内置融合](exp_rope_matmul_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [标量类型转换仅用 .to(type)，禁用类型构造函数](exp_scalar_cast_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [反向 KV 梯度 kernel：HIVM CV 均衡 + tile_mix=2 + 双缓冲 + 256×64 tile + bf16 存储 fp32 累加](exp_sda_bwd_kv_ascend_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [前向 SDA kernel：CV 均衡 + 双缓冲 + bf16 乘 fp32 累加 + 64×256 tile 分相处理](exp_sda_fwd_ascend_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [禁用 Python 切片语法，改用 get_element/extract_slice/insert_slice](exp_slice_001.md) — 主题相关：共享 general_ascend
- 相关主题: [禁止对 tl.arange 索引表达式调用 get_element](exp_slice_002.md) — 主题相关：共享 general_ascend
- 相关主题: [稀疏注意力 prefill：tle.index_select_simd 按需取 Top-K + cube1 加速 + 工作集 tile=64](exp_sparse_flash_attn_prefill_001.md) — 主题相关：共享 跨芯片适配

<!-- okf:related:end -->
