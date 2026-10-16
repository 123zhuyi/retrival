---
kind: index
type: section_index
title: runbooks/portability/（占位）
description: 分类4 规范与移植优化点库（OPT-*/AP-*/CT-*，跨算子单一共享、增量合并）。
---

# 分类4 规范与移植（占位）

覆盖二级维度：4.1 动态 shape/兼容 · 4.2 编译器约束/hints · 4.3 910B/910C/310P。

生产 skill：`triton-knowledge-portability-ingest` 对应分类 skill（见插件 `skills/` 目录）。条目骨架见对应 skill 的 `STRUCTURE-runbook.md`。


## 新增条目（raw experiences 导入，待 review）

- [Ascend 后端禁用语法及替代写法（控制流/lambda/链式布尔/直接索引/负偏移）](exp_api_001.md) — 910b2/910b3 上 Ascend 后端禁止 return/break/continue、lambda、链式布尔、张量直接索引与负偏移，需用 mask 与 tl.where 替代（正确性约束，无性能收益）
- [复杂嵌套条件下 tl.where 改 if-else 分支](exp_api_002.md) — 910b2/910b3 上 Ascend 后端在多分支/嵌套条件场景下 tl.where 易触发执行错误，需改用 if-else 分支实现；简单单分支条件仍可用 tl.where
- [静态上限循环：while 改 for range（上限须为 tl.constexpr）](exp_api_003.md) — 910b2/910b3 上 Ascend 后端不支持 while 循环，循环上限为编译时常量时直接以 for range 替代，上限须声明为 tl.constexpr
- [动态上限循环：for + if + MAX_ITERS 编译期上界替代 while](exp_api_004.md) — 910b2/910b3 上 Ascend 后端不支持 while，循环上限为运行时值时设足够大的编译期常量 MAX_ITERS 作 for 上界、循环内 if i < n_iters 控制实际执行，代价约 5% 空转开销
- [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](exp_ascend_blockptr_boundary_scalar_001.md) — 910b2/910b3 上 tl.make_block_ptr + boundary_check 会逐元素生成标量边界判断，导致 aiv/aic scalar_ratio 升高；小 tile 频繁访存场景改手动指针+mask，完整 tile 用 ALIGNED constexpr 走无检查路径，声称收益 60%
- [移除确定性访存上的冗余 compile_hint（mayDiscretememaccess）](exp_ascend_compile_hint_001.md) — 910b2/910b3 上对 block_ptr 确定性访存或普通 load/store 加 tl.extra.cann.extension.compile_hint(..., "mayDiscretememaccess") 会限制编译器优化，移除冗余 hint 声称收益约 25%
- [核心数在 __init__ 获取一次，按算子类型选 VEC/CUBE 核数](exp_ascend_core_num_001.md) — 910b2/910b3/910b4 上 grid 核心数须按算子类型匹配（向量类用 VEC 核数、矩阵类用 CUBE 核数），且在 Module.__init__ 阶段获取一次——forward 中调用 get_device_limit 会触发设备同步，损耗 10%-15%
- [Grid 必须是 ≤3 维 tuple 且各维乘积 ≤65535](exp_ascend_grid_001.md) — 910b2/910b3 上 Triton kernel 的 grid 必须为 tuple、最多 3 维、各维乘积 ≤65535，违反直接编译/启动失败；优先用 1 维 grid (core_num,)
- [大 shape 用交错循环：grid 固定为核心数，kernel 内按步长遍历](exp_ascend_grid_large_shape_001.md) — 910b2/910b3 上元素数 ≥1e6 的大 shape 算子，grid 固定为硬件核心数、kernel 内 for row in range(pid, M, CORE_NUM) 交错处理，负载均衡最优且规避 grid 乘积 ≤65535 限制，声称收益 15%
- [地址计算全程保持 int32，避免 int64 标量降级](exp_ascend_int64_address_001.md) — 910b2/910b3 上 int64 通用算术（VAddOp/VMulOp 等）全部降级为标量循环，地址偏移中显式/隐式 int64 会显著拉高 scalar_ratio；确认偏移乘积 <2^31 后移除所有 .to(tl.int64)，声称收益 50%
- [整数比较标量降级：比较操作数转 fp32 或编译期消除](exp_ascend_int_compare_scalar_001.md) — 910b2/910b3 上 i64 的 CMP/ADD/MUL 与 i32 的 LT/GT/LE/GE 全部标量降级（仅 f16/f32 比较和 i32 EQ/NE 走向量），tl.arange 默认 int64 是常见陷阱；比较数转 fp32 或用 constexpr 编译期消除，声称收益 40%
- [并行任务分配首选 for i in range(pid, total, core_num) 交错模式](exp_ascend_parallel_pattern_001.md) — 910b2/910b3/910b4 上总任务数远大于核心数时，kernel 内 for i in range(pid, total_items, core_num) 是最优并行分配：负载差 ≤1、无需手算每核配额、适配任意 shape，声称收益 10%
- [BLOCK_SIZE <65536 且子块内存 ≤L1 缓存，超限按维度多次嵌套切分](exp_ascend_split_001.md) — 910b2/910b3 上 BLOCK_SIZE 必须 <65536，且线程块内存须 ≤L1 缓存（910b2 32KB / 910b3 64KB）；超限时按维度多次嵌套切分（先 M 后 N），声称收益 20%
- [禁止用 tl.where 动态计算内存偏移，改 if-else 静态分支](exp_ascend_where_001.md) — 910b2/910b3 上 Ascend 后端对 tl.where 生成的动态指针偏移支持不完全，易导致 kernel 执行错误；复杂条件的偏移计算改 if-else 静态分支提前确定偏移
- [Cube/Vector 流水线编排 + 序列分半（down kernel）适配片上内存](exp_attn_bwd_q_down_001.md) — 910b2/910b3 上用 tile_mix_vector_loop/tile_mix_cube_loop 编排指令流水线（Cube 跑 GEMM、Vector 跑逐元素），enable_hivm_auto_cv_balance 交给编译器动态均衡，set_workspace_multibuffer 2 双缓冲掩盖 GM 延迟；长序列按 SEQ_LEN=S//2 拆 down kernel 适配 L0C/L1，声称收益 35%
- [Ascend 专用 tiling 配置：tile_mix cube/vector loop + multibuffer + HIVM CV 自动均衡](exp_attn_sda_001.md) — 910b2/910b3/91093/91095 上用 tile_mix_cube_loop/tile_mix_vector_loop 显式映射计算循环到 Cube/Vector 单元，配 set_workspace_multibuffer 2 双缓冲与 enable_hivm_auto_cv_balance，在线 softmax 累加用 FP32、IO 保持 BF16，声称收益 35%
- [tl.constexpr 仅限内核参数声明，Host 侧禁用](exp_constexpr_001.md) — 910b2/910b3 上 tl.constexpr 只能出现在 Triton 内核参数声明中标记编译期常量，Host 侧（启动函数/forward）使用会直接语法报错
- [标签索引转 int32 比较 + GROUP_SIZE_M 分组调度 + 大 K tile](exp_entropy_backward_npu_001.md) — 910b2/910b3/91093/91095 上标签索引做相等比较前先转 int32 避免 int64 标量降级；GROUP_SIZE_M=4 分组调度提升 L2 局部性；BLOCK_SIZE_K=512 让 Cube 单元数据量充足，声称收益 25%
- [反向 KV kernel：HIVM CV 均衡 + 双缓冲 + 非对称 tile（128×64）+ 序列分半](exp_flashattn_bwd_kv_ascend_001.md) — 910b2/910b3 上反向 attention KV kernel 用 enable_hivm_auto_cv_balance + set_workspace_multibuffer 2 + tile_mix loop=2 均衡 Cube/Vector 流水线，BLOCK_R=128/BLOCK_C=64 非对称 tile 匹配 KV 访存粒度，SEQ_LEN=S//2 分半减少因果掩码开销，声称收益 30%
- [异构并行融合：MatMul 派 Cube、归一化+位置编码派 Vector，dl.async_task 异步掩盖](exp_fused_matmul_rms_rope_001.md) — 910b2/910b3 上把计算密集的 MatMul 派给 Cube 单元、访存密集的归一化与旋转位置编码派给 Vector 单元，用 dl.async_task 异步调度使向量操作隐藏在 GEMM 之后，三算子单 kernel 融合减少启动开销，声称收益 25%
- [GQA 分页 KV：tl.parallel bind_sub_block 双向量核并行 + KV 按组复用 + block_table 遍历](exp_gqa_paged_npu_001.md) — 910b2/910b3/91093/91095 上用 tl.parallel(0, 2, bind_sub_block=True) 把 query 头映射到 AICore 内两个向量单元，GQA 每 KV 头加载一次供组内多 query 头复用（带宽降为 1/组大小），block_table 遍历分页 KV，声称收益 45%
- [MatMul 调度：对角遍历 grouped_launch + dot_pad_only_k hint + 寄存器原位累加](exp_matmul_ascend_001.md) — 910b2/910b3 上 MatMul 用 range(pid, NUM_BLOCKS, NUM_CORES) 静态调度 + grouped_launch_diagonal 对角遍历提升 L2 局部性；dl.compile_hint(..., "dot_pad_only_k") 只 pad K 维避免整 tile padding 开销；tl.dot 第三参传累加器做寄存器原位累加，声称收益 35%
- [波前调度 MatMul：对角遍历 + dot_pad_only_k + autotune tile 长宽比](exp_matmul_ascend_wavefront_001.md) — 910b2/910b3/91093/91095 上大矩阵乘法用 grouped_launch_diagonal 波前遍历提升 cache 局部性，dl.compile_hint dot_pad_only_k 只 pad K 维，@triton.autotune 扫 tile 长宽比（128x256/256x128）与分组参数，声称收益 35%
- [UB tiling（BLOCK_Q×blockK 分块）+ tl.math.fma 复数乘法 + tl.interleave 交错布局](exp_rope_llama4_npu_001.md) — 910b2/910b3/91093/91095 上 RoPE 类 kernel 按 BLOCK_Q/blockK 分块保持工作集在 UB 内；复数乘法用 tl.math.fma 融合乘加减指令数；reshape can_reorder=True + tl.interleave 处理交错复数布局，避免显式转置/gather，声称收益 25%
- [dot_pad_only_k hint + 对角 block 调度 + batch/head 循环内置融合](exp_rope_matmul_001.md) — 910b2/910b3/91093/91095 上矩阵乘用 dl.compile_hint "dot_pad_only_k" 只 pad K 维对齐 Cube 单元，grouped_launch_diagonal 对角遍历提升 L2 命中，BATCH/HEADS 循环内置 kernel 体减少启动开销，声称收益 35%
- [标量类型转换仅用 .to(type)，禁用类型构造函数](exp_scalar_cast_001.md) — 910b2/910b3 上 Triton 标量转换只支持 scalar.to(type)，tl.float16(scalar) 等类型构造函数仅支持张量、对标量会编译报错；constexpr 常量须先赋临时变量再转换
- [反向 KV 梯度 kernel：HIVM CV 均衡 + tile_mix=2 + 双缓冲 + 256×64 tile + bf16 存储 fp32 累加](exp_sda_bwd_kv_ascend_001.md) — 910b2/910b3/91093/91095 上反向 KV 梯度 kernel 用 enable_hivm_auto_cv_balance 均衡 Vector/Cube 指令派发，tile_mix loop=2 提升 ILP，multibuffer=2 双缓冲，BLOCK_R=256/BLOCK_C=64 匹配 L0C/L1，bf16 存储 fp32 累加保精度，声称收益 45%
- [反向 Q kernel：CV 均衡 + 双缓冲 + 代数掩码（(mask-1)*1e6）+ 64×128 三角遍历](exp_sda_bwd_q_001.md) — 910b2/910b3 上反向 Q kernel 用 tile_mix loop=2 + enable_hivm_auto_cv_balance 均衡 Cube/Vector，multibuffer=2 双缓冲；掩码用代数写法 (mask.to(fp32)-1.0)*1e6 替代 if-else 避免分支分化；64×128 tile 按对角-下三角-前列结构化遍历，声称收益 35%
- [前向 SDA kernel：CV 均衡 + 双缓冲 + bf16 乘 fp32 累加 + 64×256 tile 分相处理](exp_sda_fwd_ascend_001.md) — 910b2/910b3 上前向滑窗/局部注意力 kernel 用 enable_hivm_auto_cv_balance + tile_mix loop 均衡 Cube/Vector，multibuffer=2 双缓冲，bf16 矩阵乘 + fp32 在线 softmax 累加，BLOCK_R=64/BLOCK_C=256 且序列分半（S//2）分相处理局部/因果/全局模式，声称收益 35%
- [禁用 Python 切片语法，改用 get_element/extract_slice/insert_slice](exp_slice_001.md) — 910b2/910b3 上 Triton 内核禁止 b[0]/b[i:j] 等 Python 风格切片，单元素用 tl.get_element、切片用 tl.extract_slice、插入用 tl.insert_slice
- [禁止对 tl.arange 索引表达式调用 get_element](exp_slice_002.md) — 910b2/910b3 上 tl.arange 生成的是编译期索引表达式而非内存张量，对其调用 tl.get_element 会编译报错，指定位置值应直接用数值计算（如 base + i）获得
- [迁移 kernel 须按 CANN 版本审计 launch 参数（8.5.0 删 no-limit/sync_solver/num_stages）](exp_solve_tril_005.md) — 910b2/910b3 + CANN 8.5.0 上从第三方仓库（fla/vllm 等）迁移 kernel 时须删除不兼容参数：limit_auto_multi_buffer_of_local_buffer='no-limit'（未实现）、sync_solver=True（不支持）、tl.range num_stages（依赖 TMA，Ascend USE_TMA=False），保留会编译报错/警告
- [稀疏注意力 prefill：tle.index_select_simd 按需取 Top-K + cube1 加速 + 工作集 tile=64](exp_sparse_flash_attn_prefill_001.md) — 910b2/910b3 上稀疏注意力用 tle.index_select_simd 按预算索引只从 GM gather Top-K key（带宽大降），cube1 内建调 Cube 单元做 QK^T，在线 softmax LSE 分块更新，手工 workspace 按 CACHE_BLOCK=64 tile，tle.sub_vec_id 核内向量道并行，声称收益 60%

## trajectory 实例（P5 提取）

- [attention Hardware Adaptation 实践聚合](traj-attention-hardware_adaptation.md) — 5 条 attention 硬件适配 experience 聚合
- [general_ascend Hardware Adaptation 实践聚合](traj-general_ascend-hardware_adaptation.md) — 2 条 general_ascend 硬件适配 experience 聚合
- [general_ascend Hardware Constraint 实践聚合](traj-general_ascend-hardware_constraint.md) — 3 条 general_ascend 硬件约束 experience 聚合
- [loss Hardware Adaptation 实践聚合](traj-loss-hardware_adaptation.md) — 1 条 loss 硬件适配 experience 聚合
- [matmul Hardware Adaptation 实践聚合](traj-matmul-hardware_adaptation.md) — 4 条 matmul 硬件适配 experience 聚合
- [misc Hardware Adaptation 实践聚合](traj-misc-hardware_adaptation.md) — 3 条 misc 硬件适配 experience 聚合
- [rope Hardware Adaptation 实践聚合](traj-rope-hardware_adaptation.md) — 1 条 rope 硬件适配 experience 聚合
- [solve_tril Api Compat 实践聚合](traj-solve_tril-api_compat.md) — 1 条 solve_tril API 兼容 experience 聚合

## 新增条目（GDN 三篇优化实践蒸馏，2026-08-27）

- [昇腾 950 平台特性：CV 直通数据通路与 RegBase（含 Triton 可达性边界）](ascend-950-cv-datapath-regbase.md) — 950PR/DT 平台知识，CV 融合额外 +12%~37%，RegBase 为 Triton→AscendC 升级判据

## 新增条目（ApplyRotaryPosEmb P4 后端契约沉淀，2026-09-09）

- [Triton-Ascend mask 纪律与编译期陷阱](triton-ascend-mask-codegen-traps.md) — mask 保持 int32 比较（fp32 mask 并发 store 污染）；stride-0 memref 常量折叠崩溃用 constexpr 分支规避；kernel 内禁 % 运算符

## 新增条目（grid_sample 后端陷阱沉淀，2026-09-17）

- [tl.floor(x).to(int32) 直接喂 gather 地址链在边界点取址偏移一整行](floor-gather-address-chain-hazard.md) — 进入地址计算的 floor 一律手工 floor（截断+负向修正）；triton-ascend-mask-codegen-traps 追加 §5 段粒度 gather 忽略逐 lane mask

## 新增条目（GELU 回流沉淀，2026-09-18）

- [libdevice 命名空间为 tl.extra.ascend.libdevice，文档 xxx 占位符须替换](exp_ascend_libdevice_namespace_001.md) — 照抄 tl.extra.cann 占位符致 AttributeError 全 case 编译失败；910 平台实测 ascend 命名空间
