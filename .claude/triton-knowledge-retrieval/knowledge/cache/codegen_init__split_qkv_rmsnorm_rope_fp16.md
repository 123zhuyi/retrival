---
query_type: codegen_init
query_keywords: [split_qkv, rmsnorm, rope, fp16]
created: 2026-04-27
last_updated: 2026-04-27T00:00:00
hit_count: 1
context_summary: "split QKV → RMSNorm(Q,K) → RoPE(Q,K)，qkv[12,18432]，head_dim=128，fp16，910b3"
top_units: [E1, E2, E3, T1, C4]
stale: false
---

## 检索问题（Problem Statement）

为 split QKV → RMSNorm(Q, K) → RoPE(Q, K) → V passthrough 算子生成高性能 Triton-Ascend kernel。

**触发上下文**：代码生成初始化，从零构建算子知识包  
**检索范围**：fused (normalization + rope)，qkv[12, 18432]，head_dim=128，half_rope_dim=64，num_heads=48，dtype=fp16，910b3

---

## 知识单元列表

### 硬件约束知识单元（C 系列）

**C1** `[硬件约束]` UB 容量上限
- **内容**：UB 大小 192 KB，fp16 最大元素数 ~98304。head_dim=128，推荐每个 block 处理 BLOCK_T=64 行：64×128×2=16 KB fp16 + 64×128×4=32 KB fp32 累加，远低于上限，安全。
- **来源**：`pages/hardware/constraints_cheatsheet.md`

**C2** `[硬件约束]` Grid 维度约束
- **内容**：Grid 最大 3D，单维度 ≤65535。本算子 grid = (batch × num_heads,) = 12×48 = 576，远低于上限。若处理更大 batch，注意单维度不超过 65535。
- **来源**：`pages/hardware/constraints_cheatsheet.md`, `pages/hardware/ascend_patterns.md`

**C3** `[硬件约束]` 512B 行宽对齐
- **内容**：Vector 指令的 load/store 建议行宽对齐 512B。head_dim=128，fp16 → 128×2=256B，不满足 512B。两种处理方式：① 使用 boundary_check 模式（性能略低）；② 单 program 同时处理 Q+K 各 128 维拼为 256 维（满足 512B 对齐）。推荐先用方式①验证正确性。
- **来源**：`pages/hardware/constraints_cheatsheet.md`

**C4** `[硬件约束]` fp16 精度风险
- **内容**：RMSNorm 的方差计算（sum of squares）在 fp16 下因动态范围不足会导致精度损失。**必须**使用 fp32 累加器计算方差，最后 downcast 回 fp16 输出。
- **来源**：`pages/techniques/compute_and_branch.md §2.1`

**C5** `[硬件约束]` int64 scalar fallback
- **内容**：Ascend 910B 上 int64 比较/加法退化为标量路径。batch=12 时最大偏移 12×18432=221184 在 int32 范围内，但仍推荐显式 `.to(tl.int32)` 以避免编译器退化，防止 `aiv_scalar_ratio` 升高。
- **来源**：`pages/techniques/compute_and_branch.md §3.1`, `exp_scalar_cast_001`

### 算子分类知识单元（OP 系列）

**OP1** `[性能约束]` 算子融合必要性分析
- **内容**：PyTorch 基线拆成 5 个独立核（split/RMSNorm_Q/RMSNorm_K/RoPE_Q/RoPE_K），每步读写 HBM，共约 2.2MB 带宽。融合后单次读入 qkv、单次写出 q/k/v，HBM 流量降至 ~0.9MB，减少 ~60% 带宽。
- **来源**：`pages/techniques/operator_fusion.md`, `pages/ops/fused_ops.md §Fusion-2`

**OP2** `[性能约束]` 算子分类：归一化 + 旋转位置编码融合
- **内容**：本算子属于 `normalization + rope` 融合类型。经验目录优先查：`raw/experiences/normalization/`（33条）、`raw/experiences/rope/`（10条）、`raw/experiences/misc/`（40条，含 transformer 融合组合）。
- **来源**：`index.md §2 Step 2`, `pages/experience_index.md`

### API 知识单元（A 系列）

**A1** `[API用法]` `tl.load` — 带 mask 的非连续分段加载
- **内容**：QKV split 需从连续内存段分别加载三段。推荐用 offset 计算：`q_ptr = qkv_ptr + pid * stride_batch`，然后 `tl.load(q_ptr + col_offs, mask=col_offs < q_hidden)`。支持 `other=0.0` 填充超界区域。
- **来源**：`pages/api/triton_api_cheatsheet.md`, `raw/triton-ascend-develop-guide/triton_api/Memory_Pointer_Ops/`

**A2** `[API用法]` `tl.sum` — fp32 累加器归约
- **内容**：`variance = tl.sum(x_fp32 * x_fp32, axis=0) / n`，其中 `x_fp32 = x.to(tl.float32)`。注意：`tl.sum` 返回标量，需配合 `tl.rsqrt(var + eps)` 使用。
- **来源**：`pages/techniques/compute_and_branch.md §2.1`, `raw/experiences/normalization/exp_rmsnorm_001.yaml`

**A3** `[API用法]` `tl.rsqrt` — 归一化单指令倒数平方根
- **内容**：`rstd = tl.rsqrt(variance + eps)`，比 `1.0 / tl.sqrt(variance + eps)` 更快（单条硬件指令）。RMSNorm 步骤直接用 `x_norm = x * rstd`。
- **来源**：`pages/techniques/compute_and_branch.md §2.3`, `exp_stability_004`

**A4** `[API文档]` `tl.constexpr` 参数特化
- **内容**：将以下参数提升为 `tl.constexpr`：`HAS_BIAS`（消除零 bias 路径），`INVERSE`（RoPE前/反向共用），`IS_NEOX_STYLE`（RoPE布局），`ALIGNED`（head_dim对齐），`IS_VARLEN`（变长序列）。
- **文档路径**：`pages/techniques/compute_and_branch.md §1.1`, `raw/triton-ascend-develop-guide/triton_api/triton/`

**A5** `[API用法]` `@triton.autotune` — 自动调优 BLOCK_T
- **内容**：对 `BLOCK_T`（每个 program 处理的 token 数）和 `num_stages` 做 autotune。推荐配置：`[triton.Config({'BLOCK_T': bt, 'num_stages': ns}) for bt in [32, 64, 128] for ns in [1, 2]]`。head_dim=128 固定，可作为 constexpr。
- **来源**：`pages/api/triton_api_cheatsheet.md`, `raw/experiences/normalization/exp_fused_qkv_rmsnorm_rope_001.yaml`

**A6** `[API文档]` `tl.make_block_ptr` + `tl.advance` — 2D 块指针
- **内容**：处理 QKV 分段时，可用 2D block_ptr 简化边界处理：`qkv_block_ptr = tl.make_block_ptr(base=qkv_ptr, shape=(T, H), strides=(stride_t, 1), offsets=(pid*BLOCK_T, q_start), block_shape=(BLOCK_T, head_dim), order=(1,0))`，然后 `tl.advance(ptr, (0, q_hidden))` 跳到 K 段。
- **文档路径**：`raw/triton-ascend-develop-guide/triton_api/Memory_Pointer_Ops/`

**A7** `[API用法]` `triton.jit(do_not_specialize=["batch_size"])` — 避免过度特化
- **内容**：batch_size 在推理时可能频繁变化（1/4/8/12），加 `do_not_specialize=["batch_size"]` 避免为每个值重编译。head_dim=128 固定，应作为 constexpr。
- **来源**：`pages/techniques/compute_and_branch.md §1.2`, `exp_rope_forward_001`

### 潜在瓶颈知识单元（B 系列）

**B1** `[潜在瓶颈]` fp16 精度损失 — RMSNorm 方差计算
- **触发条件**：x 值较大时（激活值大于 ±32），fp16 sum of squares 会溢出或丢失精度
- **症状**：精度验证失败，输出与 PyTorch baseline 差值 > 1e-2
- **关联经验**：`exp_stability_001`（LayerNorm fp32），`exp_fused_split_qk_norm_001`（fp32 variance）
- **修复**：load 后立即 `.to(tl.float32)`，variance 计算全程 fp32

**B2** `[潜在瓶颈]` RoPE cos/sin 重复加载 — 多头场景带宽浪费
- **触发条件**：48 个 head 共享同一组 cos/sin，若每个 program 独立 load 则重复加载 48 次
- **症状**：HBM 带宽利用率高但 compute 利用率低；profiling 显示 MTE2 时间占比大
- **关联经验**：`exp_rope_npu_001`（cos/sin 复用 + software pipeline，35%），`exp_rope_001`（persistent grid-stride）
- **修复**：每个 program 按 head_id 分配，cos/sin 按 token 偏移加载一次后复用所有 head

**B3** `[潜在瓶颈]` ILP 不足 — RMSNorm 单路 for 循环串行
- **触发条件**：head_dim=128，单次全量处理无 for-loop tiling，AI core 指令队列利用率 < 60%
- **症状**：profiling 中 `pipe_util` 低，cube 和 vector 单元交替等待
- **关联经验**：`exp_fused_split_qk_norm_001`（for-loop tiling），`exp_fused_qkv_rmsnorm_rope_001`
- **修复**：将 head_dim=128 拆分为 2-4 个子块（每块 32-64 元素），用 `tl.static_range` 展开

**B4** `[潜在瓶颈]` 小 batch 调度效率低
- **触发条件**：batch=12，num_heads=48，total programs=576，910B3 有 64 个 AI core，tail wave 问题
- **症状**：部分 core idle；实测带宽远低于峰值（1.8 TB/s）
- **关联经验**：`exp_rope_npu_001`（head blocking），`exp_rmsnorm_001`（persistent kernel grid-stride）
- **修复**：启用 persistent kernel，单个 core 用 grid-stride 处理多个 (batch, head) 对

**B5** `[潜在瓶颈]` bias 加法路径冗余
- **触发条件**：q_bias / k_bias 可能全零（某些模型无 bias），无 constexpr 控制时每次执行 bias load+add
- **关联经验**：`exp_fused_split_qk_norm_001`（constexpr HAS_BIAS）
- **修复**：`HAS_BIAS: tl.constexpr`，编译时消除 bias 路径

### 调优经验知识单元（E 系列）

**E1** `[调优经验]` `exp_fused_qkv_rmsnorm_rope_001` — QKV split + RMSNorm + RoPE 三合一
- **适用条件**：QKV 分割后接 RMSNorm + RoPE，head_dim≥64
- **性能提升**：25%
- **核心优化**：单 kernel 完成 split/norm/rope 三步，消除 2 次 HBM 中间写；fp32 variance；cos/sin 按 token 加载复用所有 head
- **关键代码**：`rstd = tl.rsqrt(var + eps); x_hat = x * rstd * weight + bias; x1, x2 = x_hat[:hd//2], x_hat[hd//2:]; o1 = x1*cos - x2*sin; o2 = x2*cos + x1*sin`
- **经验文件**：`raw/experiences/normalization/exp_fused_qkv_rmsnorm_rope_001.yaml`

**E2** `[调优经验]` `exp_fused_split_qk_norm_001` — Q/K 分别 RMSNorm 单核融合
- **适用条件**：Q 和 K 分别做 RMSNorm（可不同 weight），head_dim=128
- **性能提升**：35%
- **核心优化**：同一 program 连续处理 Q_norm 和 K_norm，fp32 variance 分开计算，constexpr HAS_BIAS 消除零 bias 路径；for-loop 展开提升 ILP
- **经验文件**：`raw/experiences/normalization/exp_fused_split_qk_norm_001.yaml`

**E3** `[调优经验]` `exp_rope_npu_001` — Ascend NPU RoPE（cos/sin 复用 + software pipeline）
- **适用条件**：Ascend NPU，head_dim≥64，前/反向共用
- **性能提升**：35%
- **核心优化**：cos/sin 寄存器驻留（加载一次，所有 head 复用）；`num_stages=2` 软件流水；`INVERSE: tl.constexpr` 前/反向共用；head blocking 聚合多 head 处理
- **经验文件**：`raw/experiences/rope/exp_rope_npu_001.yaml`

**E4** `[调优经验]` `exp_rmsnorm_001` — Persistent RMSNorm
- **适用条件**：n_cols=head_dim=128，fp16 输入，多 batch
- **性能提升**：35%
- **核心优化**：persistent kernel（grid-stride loop 覆盖所有行），weight broadcast（单次 load 复用所有 token），full-row vectorization（整行一次 load/store）
- **经验文件**：`raw/experiences/normalization/exp_fused_rms_norm_001.yaml`

**E5** `[调优经验]` `exp_fused_norm_rope_001` — RMSNorm + RoPE 融合
- **适用条件**：RMSNorm 后直接接 RoPE，无中间 HBM 写
- **性能提升**：35%
- **核心优化**：norm 结果直接在寄存器内传入 rope 旋转计算，消除一次完整的 HBM 中间写
- **经验文件**：`raw/experiences/normalization/exp_fused_norm_rope_001.yaml`

### 调优策略知识单元（T 系列）

**T1** `[调优策略]` 算子融合策略 — 消除中间 HBM 写
- **适用场景**：多个顺序算子（split → norm → rope），中间结果不被其他算子消费
- **核心方法**：将 split/RMSNorm_Q/RMSNorm_K/RoPE_Q/RoPE_K 合并为单一 kernel；寄存器内传递中间结果
- **来源**：`pages/techniques/operator_fusion.md`, `pages/ops/fused_ops.md`

**T2** `[调优策略]` Persistent Kernel + Grid-stride Loop
- **核心代码**：
  ```python
  pid = tl.program_id(0)
  num_programs = batch * num_heads
  while pid < num_programs:
      row = pid // num_heads; head = pid % num_heads
      pid += tl.num_programs(0)
  ```
- **来源**：`pages/techniques/persistent_kernels.md`

**T3** `[调优策略]` cos/sin 寄存器驻留 — 多头复用
- **适用场景**：num_heads=48，cos/sin shape=[seq_len, half_rope_dim=64]，按 token 加载后所有 head 共享
- **来源**：`pages/techniques/memory_access_optimization.md §5`, `exp_rope_npu_001`

**T4** `[调优策略]` ILP 提升 — for-loop 展开 head_dim 方向
- **核心代码**：
  ```python
  acc = tl.zeros([1], dtype=tl.float32)
  for i in tl.static_range(N_ITER):
      x = tl.load(ptr + i * BLOCK_D + offs)
      acc += tl.sum(x.to(tl.float32) ** 2)
  rstd = tl.rsqrt(acc / head_dim + eps)
  ```
- **来源**：`pages/techniques/ilp.md`

**T5** `[调优策略]` constexpr 分支消除 — HAS_BIAS + INVERSE
- **来源**：`pages/techniques/compute_and_branch.md §1.1`

### 代码骨架知识单元（S 系列）

**S1** `[代码骨架]` RMSNorm Fused Skeleton
- **骨架路径**：`pages/skeletons/rmsnorm_fused.py`
- **关键结构**：persistent kernel，grid-stride loop，fp32 variance，weight broadcast

**S2** `[代码骨架]` RoPE 旋转内联片段
- **骨架路径**：参考 `raw/experiences/rope/exp_rope_npu_001.yaml`

### 矛盾提示知识单元（X 系列）

**X1** `[矛盾提示]` head_dim=128 的 512B 对齐问题
- **内容**：512B 对齐要求每行 256 个 fp16 元素，但 head_dim=128 只有 256B。推荐先用 boundary_check 模式验证正确性，再考虑 Q+K 拼接优化。
- **来源**：`pages/hardware/constraints_cheatsheet.md`

**X2** `[矛盾提示]` care_padding 安全条件
- **内容**：`care_padding=False` 可消除 MTE2→Vector 同步屏障（+10-15% 性能），但需确保所有 mask 区域不产生写冲突。head_dim=128 为 constexpr 且 BLOCK_D=128 无 mask 时满足安全条件。
- **来源**：`pages/techniques/ilp.md §care_padding`

---

## 复用提示

- `split_qkv_rmsnorm`：去掉 RoPE（E3/T3/S2 不需要），其余复用
- `rmsnorm_rope`：去掉 split 加载逻辑（A1/OP1 部分调整），其余复用
- `split_qkv_rope`：去掉 RMSNorm（A2/A3/B1/E2/E4 不需要），其余复用
- `split_qkv_rmsnorm_rope_backward`：加 `INVERSE=True`，补充反向 grad 经验
- head_dim≠128 变体：重新评估 C1/C3 约束，更新 autotune 搜索空间
