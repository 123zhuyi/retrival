---
query_type: codegen_init
query_keywords: [groupnorm, fp32, fp16, normalization, elementwise, affine]
created: 2026-06-05
last_updated: 2026-06-05T14:32:00
hit_count: 2
context_summary: "为 GroupNorm 算子生成高性能 Triton-Ascend kernel，支持 fp32/fp16，NCHW 格式，含 weight/bias 仿射变换"
top_units: [C1, C2, P7, A1, A4, B_norm_1, E_groupnorm_001, S_rmsnorm_fused, T1, T2]
stale: false
---

## 检索问题（Problem Statement）

为 GroupNorm 算子生成高性能 Triton-Ascend kernel。

**触发上下文**：代码生成初始化，从零构建算子知识包
**检索范围**：groupnorm_fp32_fp16_NCHW，num_groups 可变，channels_per_group = C/num_groups，支持 weight/bias 仿射变换

---

## 知识单元列表

### 硬件约束知识单元（C 系列）

**C1** `[硬件约束]` UB 容量上限与 tile 规划
- **内容**：UB 192 KB，fp16 最大 ~98304 elements；fp32 最大 ~49152 elements。GroupNorm 每 program 处理一个 group 的 channels_per_group × HW 个元素。当 C=64, HW=3136(56×56), num_groups=8 时，channels_per_group=8, group_size=8×3136=25088 elements (fp16=50KB, fp32=100KB)，均在 UB 内。当 HW 很大时需分块处理。
- **来源**：`pages/hardware/constraints_cheatsheet.md C1`

**C2** `[硬件约束]` Grid 大小与 multi-round 调度
- **内容**：grid 维度乘积 ≤ 65535。GroupNorm 的 grid = N × num_groups。当 N=1, num_groups=8 时 grid=8；当 N 较大时可能超限，需用 persistent kernel 或 grid-stride loop。
- **来源**：`pages/hardware/constraints_cheatsheet.md C2`

**C3** `[硬件约束]` 512B 行宽对齐
- **内容**：matmul 场景需要 512B 对齐。GroupNorm 不涉及 matmul，但连续内存访问仍需尽量对齐以优化带宽。
- **来源**：`pages/hardware/constraints_cheatsheet.md C3`

**C4** `[硬件约束]` 禁用 tl.swizzle2d
- **内容**：Ascend 上行为未定义，不使用。
- **来源**：`pages/hardware/constraints_cheatsheet.md C4`

**C5** `[性能约束]` int64 CMP/ADD scalar fallback
- **内容**：int64 比较和加法退化为标量路径。`tl.arange` 返回 int64，比较前需 `.to(tl.float32)`。
- **来源**：`pages/hardware/constraints_cheatsheet.md P1/P2`

### 算子分类（OP 系列）

**OP1** `[算子分类]` GroupNorm 属于 Normalization 类型
- **内容**：对输入 x[N,C,H,W] 按 num_groups 分组，每组计算 mean 和 var，然后标准化并应用仿射变换 weight[C] 和 bias[C]。
  公式：
  - x_reshaped = x.view(N, num_groups, channels_per_group, H, W)
  - mean = x_reshaped.mean(dim=[2,3,4])  # per group
  - var = x_reshaped.var(dim=[2,3,4], unbiased=False)
  - x_norm = (x - mean) / sqrt(var + eps)
  - y = x_norm * weight + bias
- **来源**：`pages/ops/normalization.md`

### API 知识单元（A 系列）

**A1** `[API用法]` tl.load / tl.store 与 care_padding=False
- **内容**：`tl.load(ptr, mask=mask, other=0.0, care_padding=False)` 可消除 MTE2→Vector 同步屏障，提升 ILP。GroupNorm 中当 padding 值被 mask 过滤时安全使用。
- **来源**：`pages/api/triton_api_cheatsheet.md`

**A2** `[API用法]` tl.arange 返回 int64，需转 fp32
- **内容**：`cols = tl.arange(0, BLOCK_SIZE).to(tl.float32)` 避免 int64 比较导致的 scalar fallback。
- **来源**：`pages/api/triton_api_cheatsheet.md`

**A3** `[API用法]` tl.sum 归约操作
- **内容**：`tl.sum(x, axis=0)` 用于计算 group 内元素的和。fp16/bf16 输入必须先转 fp32 再 sum。
- **来源**：`pages/api/triton_api_cheatsheet.md`

**A4** `[API用法]` tl.rsqrt 替代 1/sqrt
- **内容**：`inv_std = tl.rsqrt(var + eps)` 比 `1.0 / tl.sqrt(var + eps)` 更快，使用硬件 rsqrt 指令。
- **来源**：`pages/ops/normalization.md`

**A5** `[API用法]` tl.constexpr 编译期分支消除
- **内容**：用 `HAS_WEIGHT: tl.constexpr` 和 `HAS_BIAS: tl.constexpr` 在编译期消除 weight/bias 的条件分支。
- **来源**：`pages/api/triton_api_cheatsheet.md`

**A6** `[API用法]` tl.cdiv 向上取整除法
- **内容**：`tl.cdiv(total, BLOCK_SIZE)` 计算循环次数。
- **来源**：`pages/api/triton_api_cheatsheet.md`

### 瓶颈预判（B 系列）

**B_norm_1** `[潜在瓶颈]` fp16 累加溢出
- **内容**：`sum(x^2)` 和 `sum(x)` 在 fp16 下对大 group_size 易溢出。必须在 fp32 下累加。
- **来源**：`pages/ops/normalization.md`

**B_norm_2** `[潜在瓶颈]` 非连续内存访问
- **内容**：NCHW 格式下，同一 group 的 channel 在内存中不连续（间隔 HW 个元素）。需通过 stride 计算正确偏移，或考虑将输入 reshape 为 [N, num_groups, channels_per_group, H, W]。
- **来源**：`pages/ops/normalization.md`

**B_norm_3** `[潜在瓶颈]` weight/bias 重复加载
- **内容**：每个 group 处理时从 HBM 加载 weight/bias，造成冗余访问。可通过 persistent kernel 缓存。
- **来源**：`pages/ops/normalization.md`

### 调优经验（E 系列）

**E_groupnorm_001** `[调优经验]` GroupNorm 三趟算法
- **内容**：
  - dynamic_vectorization_tuning: 使用 autotune 搜索 BLOCK_SUB_HW_SIZE（32-16384）
  - coalesced_memory_access_pattern: 以 HW 维度为内层 stride，确保连续访存
  - high_precision_accumulation: fp32 累加 X_sum 和 X_var_sum
  - three_pass_algorithm: 分三趟（Mean → Variance → Normalization），降低寄存器压力
  - 提升：25%
- **来源**：`raw/experiences/normalization/exp_groupnorm_001.yaml`

**E_rmsnorm_001** `[调优经验]` fp32 混合精度累加
- **内容**：fp16/bf16 输入转 fp32 后累加，最终 cast 回输出类型。
- **来源**：`raw/experiences/normalization/exp_rmsnorm_001.yaml`

**E_online_mean_var_001** `[调优经验]` Welford 在线算法
- **内容**：单遍稳定计算均值和方差，避免数值不稳定。
- **来源**：`raw/experiences/normalization/exp_online_mean_var_001.yaml`

### 调优策略（T 系列）

**T1** `[调优策略]` BLOCK_SIZE 选择
- **内容**：GroupNorm 中 BLOCK_SIZE 用于处理 HW 维度。推荐值 {128, 256, 512, 1024, 2048, 4096}，根据 HW 大小选择，尽量一次处理完整 HW 或 channels_per_group*HW。
- **来源**：`pages/techniques/tiling.md`

**T2** `[调优策略]` Persistent Kernel
- **内容**：当 N × num_groups >> num_aicores 时，使用 persistent kernel（grid = num_aicores），grid-stride 处理所有 group。weight/bias 只加载一次。
- **来源**：`pages/techniques/persistent_kernels.md`

**T3** `[调优策略]` care_padding=False 启用 ILP
- **内容**：在 tl.load 中设置 care_padding=False，消除 MTE2→Vector 屏障，提升指令级并行。
- **来源**：`pages/techniques/ilp.md`

### 代码骨架（S 系列）

**S_rmsnorm_fused** `[代码骨架]` RMSNorm 融合骨架
- **内容**：参考 `pages/skeletons/rmsnorm_fused.py`，包含：
  - Persistent kernel 模式（grid = num_aicores）
  - 权重只加载一次
  - Two-pass 算法（Pass 1 计算 var，Pass 2 归一化）
  - 大 hidden dim 列分块变体
- **来源**：`pages/skeletons/rmsnorm_fused.py`

---

## 复用提示

**可部分复用本缓存的 query_keywords 变体**：
- 含 `layernorm` + `fp32/fp16` → 复用 C 系列、A 系列、T 系列；替换 OP 和 E 系列为 LayerNorm 专用
- 含 `rmsnorm` + `fp32/fp16` → 复用 C 系列、A 系列、T 系列；替换 OP 和 E 系列为 RMSNorm 专用
- 含 `groupnorm` + `bf16` → 复用全部，增量更新 dtype 相关约束
