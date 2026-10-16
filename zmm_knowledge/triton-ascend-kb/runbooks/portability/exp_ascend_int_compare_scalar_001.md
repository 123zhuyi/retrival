---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: portability
dimension: 4.3
title: 整数比较标量降级：比较操作数转 fp32 或编译期消除
description: 910b2/910b3 上 i64 的 CMP/ADD/MUL 与 i32 的 LT/GT/LE/GE 全部标量降级（仅 f16/f32 比较和 i32 EQ/NE 走向量），tl.arange 默认 int64 是常见陷阱；比较数转 fp32 或用 constexpr 编译期消除，声称收益 40%
tags: [跨芯片适配, 通用优化, general_ascend]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 整数比较标量降级：比较操作数转 fp32 或编译期消除

【优化维度】：跨芯片适配
【适用条件】：平台 910b2/910b3 ｜ triton-ascend 版本 未标注（按验证版本）｜ dtype 涉及 int32/int64 索引与比较处 ｜ shape 任意（nd）
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：int 转 fp32 仅在整数值 <2^24 时精确——索引/边界超过 2^24（约 1677 万）时 fp32 比较可能判错，此时只能用 constexpr 编译期消除或保留标量代价；i32 的 EQ/NE 与 f16/f32 比较本走向量、无需处理；该降级是 Ascend 910B 特有行为，GPU 上无此问题；910C/310P 未验证（unverified)）

### 原则
- **降级规则**：Ascend 910B上，i64的CMP/ADD/MUL全部标量降级；i32的LT/GT/LE/GE比较也标量降级；只有f16/f32比较和i32的EQ/NE走向量
- **常见陷阱**：tl.arange(0, N)默认产生int64，直接用其生成mask会触发int64比较降级
- **解决方案**：将比较操作数转换为fp32，或使用constexpr在编译期避免运行时比较
- **与data types.md的关系**：本经验聚焦i32比较降级，data_types.md主要覆盖int64；两者互补
- **诊断方法**：profiling中aiv_scalar_ratio高，检查代码中所有<、>、<=、>=比较

### 代码/骨架出处
- `Triton`:
```python
# 降级：int64/int32 比较
offs = tl.arange(0, BLOCK)   # int64 by default
mask = offs < N              # int64 LT → scalar

# 优化1：先转 int32（若范围允许）
offs_i32 = offs.to(tl.int32)
mask = offs_i32 < N          # 仍是 i32 LT → scalar！

# 优化2：转 fp32 比较
mask = offs.to(tl.float32) < N.to(tl.float32)  # 向量

# 优化3：编译期已知边界
if ALIGNED:  # constexpr
    mask = None  # 无需比较
```

**坏实践（反例）**：把 `tl.arange(0, BLOCK)`（默认 int64）直接 `< N` 生成 mask，或"优化"为 `.to(tl.int32) < N` → i64/i32 的 LT 都标量降级，aiv_scalar_ratio 飙升，后者是常见假修复 → ✅ 正解即本条：比较数转 fp32（值 <2^24 时）或以 `ALIGNED` 等 constexpr 分支在编译期消除比较

**已知实例**：general_ascend §raw (`experiences/general_ascend/exp_ascend_int_compare_scalar_001.yaml`)
- 声称性能收益：40%
- 备注：即使将tl.arange转为int32，LT/GT/LE/GE仍会标量降级。必须转fp32或消除比较。这是Ascend 910B特有的行为，与GPU不同。

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
- 相关主题: [核心数在 __init__ 获取一次，按算子类型选 VEC/CUBE 核数](exp_ascend_core_num_001.md) — 主题相关：共享 general_ascend, 跨芯片适配, 通用优化
- 相关主题: [Grid 必须是 ≤3 维 tuple 且各维乘积 ≤65535](exp_ascend_grid_001.md) — 主题相关：共享 general_ascend, 通用优化
- 相关主题: [大 shape 用交错循环：grid 固定为核心数，kernel 内按步长遍历](exp_ascend_grid_large_shape_001.md) — 主题相关：共享 general_ascend, 跨芯片适配, 通用优化
- 相关主题: [地址计算全程保持 int32，避免 int64 标量降级](exp_ascend_int64_address_001.md) — 主题相关：共享 general_ascend, 跨芯片适配, 通用优化
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
