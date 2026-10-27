---
query_type: codegen_init
query_keywords: [matmul, fp16, groupnorm, swish]
created: 2026-04-27
last_updated: 2026-04-27T00:00:00
hit_count: 1
context_summary: "Linear → Swish → Bias Add → GroupNorm，x[32768,1024]→[32768,4096]，groups=64，fp16，910b3"
top_units: [X1, E1, E2, E3, E4]
stale: false
---

## 检索问题（Problem Statement）

为 Linear(x) → Swish → Bias Add → GroupNorm(groups=64) 算子生成高性能 Triton-Ascend kernel。

**触发上下文**：代码生成初始化，从零构建算子知识包  
**检索范围**：fused (matmul + elementwise + normalization)，x[32768,1024]→[32768,4096]，num_groups=64，channels_per_group=64，dtype=fp16，910b3

---

## 知识单元列表

### 硬件约束知识单元（C 系列）

**C1** `[硬件约束]` UB 容量上限与 tile 规划
- **内容**：UB 192 KB，fp16 最大 ~98304 elements。matmul 同时持有 A/B/C tile，实际约 64KB per buffer。推荐 BLOCK_M=128, BLOCK_N=256（满足512B对齐），BLOCK_K=256。GroupNorm 每 program 处理 channels_per_group=64 个元素（64×2B=128B），远低于 UB，完全在寄存器内完成。
- **来源**：`pages/hardware/constraints_cheatsheet.md C1`

**C2** `[硬件约束]` Grid 大小与 multi-round 调度
- **内容**：M=32768，BLOCK_M=128 → 256 tiles；N=4096，BLOCK_N=256 → 16 tiles；总 grid=4096，910b3 仅 24 核，约 170 轮调度。GroupNorm 的 grid = 32768×64 groups = 2097152（必须 persistent kernel）。
- **来源**：`pages/hardware/constraints_cheatsheet.md C2`

**C3** `[硬件约束]` 512B 行宽对齐（matmul 强制）
- **内容**：BLOCK_N × 2(fp16) 必须是 512B 整数倍，即 BLOCK_N 须为 256 的倍数。本算子 N=4096，BLOCK_N=256 满足。GroupNorm channels_per_group=64 仅 128B，不满足 512B，但 GroupNorm 是归约算子（非 matmul），无此约束。
- **来源**：`pages/hardware/constraints_cheatsheet.md C3`

**C4** `[硬件约束]` 禁用 tl.swizzle2d
- **内容**：Ascend 上行为未定义，matmul 核映射必须用 grouped_launch_diagonal 或手动对角 pid 映射。
- **来源**：`pages/hardware/constraints_cheatsheet.md C4`

**C5** `[性能约束]` int64 scalar fallback
- **内容**：行偏移 `row_id × K`：32768×1024=33M，在 int32 范围内安全。GroupNorm 偏移 `group_id × CPG`：32768×64×64=134M，在 int32 范围内。仍建议显式 `.to(tl.int32)` 防编译器退化。
- **来源**：`pages/hardware/constraints_cheatsheet.md P1`

**C6** `[硬件约束]` fp16 累加精度（GroupNorm + Swish 关键）
- **内容**：①Swish 激活：sigmoid 在 fp16 下对大值精度不足，**必须**在 fp32 accumulator 上施加 Swish，不提前 downcast；②GroupNorm：mean/var 计算在 fp32 中进行。
- **来源**：`pages/hardware/constraints_cheatsheet.md P7`

### 算子分类知识单元（OP 系列）

**OP1** `[性能约束]` 算子融合必要性分析
- **内容**：PyTorch 4 个独立核总 HBM 流量 ≈ 1.3GB（读+写），融合后约 0.5GB，减少约 60%。GEMM + Swish + Bias 合一：消除 Swish 的 HBM 中间写（256MB 节省）。
- **来源**：`pages/techniques/operator_fusion.md`, `pages/ops/fused_ops.md §Fusion-7`

**OP2** `[性能约束]` 算子分类：两段融合
- **内容**：
  - **Kernel 1**：fused_linear_swish_bias（Fusion-7 变体）— GEMM(Cube) + bias add(Vector) + Swish(Vector)
  - **Kernel 2**：groupnorm — two-pass mean/var + normalize + affine
  - GroupNorm 无法内联进 GEMM（见 X1）
- **来源**：`index.md §2 Step 2`, `pages/ops/fused_ops.md`

### API 知识单元（A 系列）

**A1** `[API用法]` `tl.dot` — matmul Cube 核心
- **内容**：`acc += tl.dot(a, b)`，rank-2 only，默认 fp32 accumulator。K 循环内多次累加，BLOCK_K=256 匹配 L0C 容量。
- **来源**：`pages/api/triton_api_cheatsheet.md`, `raw/triton-ascend-develop-guide/triton_api/Linear_Algebra_Ops/`

**A2** `[API用法]` `tl.load / tl.store` — 分段加载与带 mask 写回
- **内容**：linear.bias 在 K 循环前加载（broadcast 至 BLOCK_M×BLOCK_N）；extra bias 在 K 循环结束后加；GroupNorm weight/bias 为 channels_per_group=64 elements。
- **来源**：`pages/api/triton_api_cheatsheet.md`

**A3** `[API用法]` `tl.sigmoid` — Swish 激活数值精度
- **内容**：`swish(acc_fp32) = acc_fp32 * tl.sigmoid(acc_fp32)`，必须在 fp32 accumulator 上施加（不提前 downcast 至 fp16）。Swish 在 K 循环结束后、store 前对 acc 直接施加。
- **来源**：`pages/techniques/compute_and_branch.md §2`, `raw/experiences/activation/exp_swiglu_fwd_001.yaml`

**A4** `[API用法]` `tl.sum` — GroupNorm 方差计算（fp32）
- **内容**：`mean = tl.sum(x_fp32, axis=0) / CHANNELS_PER_GROUP`；`var = tl.sum((x_fp32 - mean)**2, axis=0) / CHANNELS_PER_GROUP`。channels_per_group=64，可一次性全装寄存器，two-pass 完全在寄存器内。
- **来源**：`pages/techniques/compute_and_branch.md §2.1`, `exp_groupnorm_001.yaml`

**A5** `[API用法]` `tl.rsqrt` — GroupNorm 归一化
- **内容**：`rstd = tl.rsqrt(var + eps)`，硬件单指令。`x_norm = (x - mean) * rstd * gn_weight + gn_bias`。
- **来源**：`pages/techniques/compute_and_branch.md §2.3`

**A6** `[API文档]` `@triton.autotune` — matmul tile size 自动调优
- **内容**：推荐搜索空间：`BLOCK_M in [128, 256]`，`BLOCK_N=256`（固定满足对齐），`BLOCK_K=256`，`GROUP_SIZE in [8, 16, 32]`。GroupNorm autotune：`BLOCK_SUB in [32, 64]`。
- **来源**：`pages/hardware/constraints_cheatsheet.md §Autotune 搜索空间`

**A7** `[API文档]` `tl.constexpr` — 编译时分支消除
- **内容**：推荐 constexpr：`HAS_LINEAR_BIAS`，`HAS_EXTRA_BIAS`，`ACTIVATION="swish"`，`CHANNELS_PER_GROUP=64`，`BLOCK_M/N/K/GROUP_SIZE`。
- **来源**：`pages/techniques/compute_and_branch.md §1.1`

**A8** `[API文档]` `dl.async_task` — Cube/Vector 异构调度
- **内容**：GEMM 在 Cube 单元（scope="cube"），Swish+Bias+GroupNorm 在 Vector 单元（scope="vector"），实现流水并行。
- **文档路径**：`raw/triton-ascend-develop-guide/triton_api/Extension_Ops/`

### 潜在瓶颈知识单元（B 系列）

**B1** `[潜在瓶颈]` GEMM L2 Bank Conflict（M,N≥512 大矩阵）
- **触发条件**：M=32768, N=4096，均远超 512，行主序 tiling 相邻核访问同列，L2 bank conflict 严重
- **症状**：`CubeUtilization` < 70%；实测吞吐低于峰值
- **关联经验**：`exp_matmul_diag_tiling_001`（35%），`exp_matmul_001`（40%）
- **修复**：对角 tiling（pid → (m_id, n_id) 对角映射），GROUP_SIZE=8 or 16

**B2** `[潜在瓶颈]` Swish 激活 fp16 精度损失
- **触发条件**：GEMM acc 提前 downcast 至 fp16 后做 Swish
- **症状**：精度验证失败，max_err > 1e-2
- **关联经验**：`exp_swiglu_fwd_001`（fp32 sigmoid）
- **修复**：Swish 必须在 fp32 acc 上施加，不提前 downcast

**B3** `[潜在瓶颈]` GroupNorm 两遍 HBM 读
- **触发条件**：GroupNorm 若两遍算法则 HBM 读两次（共 ~512MB）
- **关联经验**：`exp_groupnorm_001`（连续访问，25%）
- **修复**：channels_per_group=64 时，two-pass 完全在寄存器内，无额外 HBM 读写

**B4** `[潜在瓶颈]` GroupNorm weight 重复加载
- **触发条件**：GroupNorm total programs = 32768×64 = 2097152，每 program 各加载 64 fp16 weight
- **关联经验**：`exp_rmsnorm_001`（persistent weight 复用），`exp_groupnorm_001`
- **修复**：GroupNorm Persistent kernel，weight/bias 寄存器缓存

### 调优经验知识单元（E 系列）

**E1** `[调优经验]` `exp_matmul_001` — 大矩阵标准 GEMM L0 分块
- **适用条件**：M≥256, N≥256, K≥1024, fp16 — 完全匹配
- **性能提升**：40%
- **经验文件**：`raw/experiences/matmul/exp_matmul_001.yaml`

**E2** `[调优经验]` `exp_matmul_diag_tiling_001` — 对角 tiling
- **适用条件**：M≥512, N≥512 — 完全匹配（M=32768, N=4096）
- **性能提升**：25-35%
- **经验文件**：`raw/experiences/matmul/exp_matmul_diag_tiling_001.yaml`

**E3** `[调优经验]` GEMM + Bias + Swish（Fusion-7 变体）
- **适用条件**：GEMM 后接 bias add + Swish elementwise
- **性能提升**：35%
- **关键代码**：
  ```python
  acc += tl.dot(a, b)           # K 循环内
  # K 循环结束后
  lin_bias = tl.load(lin_bias_ptr + n_off, mask=n_mask)
  ext_bias = tl.load(ext_bias_ptr + n_off, mask=n_mask)
  acc += lin_bias[None, :] + ext_bias[None, :]
  acc = acc * tl.sigmoid(acc)   # Swish in fp32
  tl.store(c_ptr + ..., acc.to(tl.float16), mask=...)
  ```
- **来源**：`pages/ops/fused_ops.md §Fusion-7`

**E4** `[调优经验]` `exp_groupnorm_001` — GroupNorm 向量化 + fp32 累加
- **适用条件**：GroupNorm，fp16/bf16，channels_per_group=64 — 完全匹配
- **性能提升**：25%
- **经验文件**：`raw/experiences/normalization/exp_groupnorm_001.yaml`

**E5** `[调优经验]` `exp_swiglu_fwd_001` — Swish fp32 中间精度策略
- **适用条件**：Swish/SiLU 激活，fp16/bf16
- **性能提升**：15%（独立激活），融合进 GEMM 后叠加效果
- **经验文件**：`raw/experiences/activation/exp_swiglu_fwd_001.yaml`

### 调优策略知识单元（T 系列）

**T1** `[调优策略]` 两段融合架构
- **核心方法**：Kernel 1 `fused_linear_swish_bias`（GEMM→fp32 acc→bias×2→Swish→store fp16）；Kernel 2 `groupnorm_persistent`（two-pass→normalize→gn_w/b→store fp16）
- **来源**：`pages/ops/fused_ops.md §Fusion-7`

**T2** `[调优策略]` 对角 Core Mapping（M=32768, N=4096）
- **核心代码**：
  ```python
  num_pid_m = tl.cdiv(M, BLOCK_M)  # 256
  num_pid_n = tl.cdiv(N, BLOCK_N)  # 16
  pid = tl.program_id(0)
  num_pid_in_group = GROUP_SIZE * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * GROUP_SIZE
  group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  ```
- **来源**：`pages/techniques/core_mapping.md`

**T3** `[调优策略]` GroupNorm Two-pass 寄存器内完成（channels_per_group=64）
- **核心代码**：
  ```python
  row = tl.program_id(0)
  offs = tl.arange(0, CHANNELS_PER_GROUP)
  x = tl.load(x_ptr + row * CHANNELS_PER_GROUP + offs).to(tl.float32)
  mean = tl.sum(x) / CHANNELS_PER_GROUP
  var  = tl.sum((x - mean) ** 2) / CHANNELS_PER_GROUP
  rstd = tl.rsqrt(var + eps)
  y = (x - mean) * rstd * tl.load(gn_w_ptr + offs).to(tl.float32) + tl.load(gn_b_ptr + offs).to(tl.float32)
  tl.store(y_ptr + row * CHANNELS_PER_GROUP + offs, y.to(tl.float16))
  ```
- **来源**：`pages/ops/normalization.md`

**T4** `[调优策略]` Swish 融合进 GEMM accumulator（无中间 HBM 写）
- **核心方法**：K 循环结束后：`acc += (lin_bias + extra_bias)[None, :]`；`acc = acc * tl.sigmoid(acc)`；`tl.store(..., acc.to(fp16))`
- **来源**：`pages/ops/fused_ops.md §Fusion-7`

### 代码骨架知识单元（S 系列）

**S1** `[代码骨架]` GEMM + Bias + Swish skeleton（Fusion-7 变体）
- **骨架路径**：`pages/ops/fused_ops.md §Fusion-7`
- **关键结构**：K 循环结束后 acc（fp32）→ bias add × 2 → Swish → store fp16

**S2** `[代码骨架]` GroupNorm kernel（two-pass，channels_per_group=64，寄存器内完成）
- **骨架路径**：参考 `pages/ops/normalization.md §标准 RMSNorm 模板`

### 矛盾提示知识单元（X 系列）

**X1** `[矛盾提示]` GroupNorm 无法融合进 GEMM kernel（对齐冲突）
- **内容**：GroupNorm 需要对 channels_per_group=64 列做行归约，而 GEMM BLOCK_N=256（满足 512B 对齐）≠ 64。若令 BLOCK_N=64 则违反 512B 对齐。因此**必须分两个 kernel**。
- **来源**：`pages/hardware/constraints_cheatsheet.md C3`, `exp_groupnorm_001`

**X2** `[矛盾提示]` Swish vs SwiGLU 语义区分
- **内容**：本算子 Swish = `x * sigmoid(x)`（单输入），非 SwiGLU（双输入）。`exp_swiglu_fwd_001` 的 fp32 sigmoid 策略可复用，但不要复用其双输入的 gate 分支结构。
- **来源**：`raw/experiences/activation/exp_swiglu_fwd_001.yaml`

---

## 复用提示

- `matmul_gelu_groupnorm`：Swish → GELU，A3/E3/T4 局部更新，精度策略相同
- `matmul_silu_groupnorm`：SiLU=Swish，完全相同
- `matmul_swish_layernorm`：GroupNorm → LayerNorm，E4 换为 exp_skip_layernorm_001，T3 类似
- `matmul_swish_rmsnorm`：GroupNorm → RMSNorm，去掉 mean 计算，T3 更简单
- `matmul_bias_groupnorm`：去掉 Swish，kernel 1 退化为 GEMM + bias only
- `linear_swish_groupnorm_large`：batch 更大或 N 更大，C2 grid 约束重新评估，T5 更关键
