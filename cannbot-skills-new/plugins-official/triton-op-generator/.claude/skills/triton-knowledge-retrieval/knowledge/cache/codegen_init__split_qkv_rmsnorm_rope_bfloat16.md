---
query_type: codegen_init
query_keywords: [split_qkv, rmsnorm, rope, bfloat16]
created: 2026-05-05
last_updated: 2026-06-02T10:54:00
hit_count: 4
context_summary: "split QKV → RMSNorm(Q,K) → RoPE(Q,K)，qkv[12,8192]，head_dim=128，bfloat16，910b2"
top_units: [E1, E2, E3, T1, C4]
stale: false
---

## 检索问题（Problem Statement）

为 split QKV → RMSNorm(Q, K) → RoPE(Q, K) → V passthrough 算子生成高性能 Triton-Ascend kernel。

**触发上下文**：代码生成初始化，从零构建算子知识包  
**检索范围**：fused (split + normalization + rope)，qkv[12, 8192]，head_dim=128，half_rope_dim=64，q_hidden=6144, kv_hidden=1024，dtype=bfloat16，910b2

---

## 知识单元列表

（部分命中：4/5 与 codegen_init__split_qkv_rmsnorm_rope_fp16.md 重叠，差异关键词 bfloat16 做增量补充）

### 硬件约束知识单元（C 系列）

**C1** `[硬件约束]` UB 容量上限
- **内容**：UB 大小 192 KB，bfloat16 最大元素数 ~98304。head_dim=128，推荐每个 block 处理 BLOCK_T=64 行：64×128×2=16 KB bf16 + 64×128×4=32 KB fp32 累加，远低于上限，安全。
- **来源**：`pages/hardware/constraints_cheatsheet.md`

**C2** `[硬件约束]` Grid 维度约束
- **内容**：Grid 最大 3D，单维度 ≤65535。本算子基线 grid = (batch × num_q_heads,) = 12×48 = 576（q_hidden=6144, head_dim=128 → 48 heads），远低于上限。
- **来源**：`pages/hardware/constraints_cheatsheet.md`

**C3** `[硬件约束]` 512B 行宽对齐
- **内容**：head_dim=128，bf16 → 128×2=256B，不满足 512B。推荐先用 boundary_check 模式验证正确性；进阶优化可同时处理 Q+K 拼为 256 维（512B对齐）。
- **来源**：`pages/hardware/constraints_cheatsheet.md`

**C4** `[硬件约束]` bfloat16 精度特性
- **内容**：bfloat16（1+8+7 位）相比 fp16（1+5+10 位）动态范围更大，但小数精度较低。RMSNorm 方差计算**必须**使用 fp32 累加器，load 后立即 `.to(tl.float32)`。bfloat16 不会像 fp16 那样容易溢出，但精度仍需 fp32 中间计算保证。
- **来源**：`pages/techniques/compute_and_branch.md §2.1`

**C5** `[硬件约束]` int64 scalar fallback
- **内容**：Ascend 910B 上 int64 退化为标量路径。推荐显式 `.to(tl.int32)` 以避免编译器退化。
- **来源**：`pages/techniques/compute_and_branch.md §3.1`

### 算子分类知识单元（OP 系列）

**OP1** `[性能约束]` 算子融合必要性分析
- **内容**：PyTorch 基线拆成 5 个独立核（split/RMSNorm_Q/RMSNorm_K/RoPE_Q/RoPE_K），融合后单次读入 qkv、单次写出 q/k/v，HBM 流量降低约 60%。qkv[12,8192] bf16 = 196KB，每步 HBM 读写显著。
- **来源**：`pages/techniques/operator_fusion.md`

**OP2** `[性能约束]` 算子分类：split + 归一化 + 旋转位置编码融合
- **内容**：本算子属于 `split + normalization + rope` 融合类型。Q有48 heads，K有8 heads（kv_hidden=1024, head_dim=128），注意 Q/K heads 数量不同，并行策略需分别处理。
- **来源**：`index.md §2`

### API 知识单元（A 系列）

**A1** `[API用法]` `tl.load` — 分段加载 QKV
- **内容**：Q段从 offset 0 加载（q_hidden=6144），K段从 q_hidden 加载（kv_hidden=1024），V段从 q_hidden+kv_hidden 加载。推荐用 offset 计算：`q_ptr = qkv_ptr + pid_batch * stride_batch`，然后按 q_start / k_start / v_start 加载各段。
- **来源**：`pages/api/triton_api_cheatsheet.md`

**A2** `[API用法]` `tl.sum` — fp32 累加器归约
- **内容**：`variance = tl.sum(x_fp32 * x_fp32, axis=0) / n`，其中 `x_fp32 = x.to(tl.float32)`。注意：`tl.sum` 返回标量，需配合 `tl.rsqrt(var + eps)` 使用。
- **来源**：`pages/techniques/compute_and_branch.md §2.1`

**A3** `[API用法]` `tl.rsqrt` — 归一化单指令倒数平方根
- **内容**：`rstd = tl.rsqrt(variance + eps)`，比 `1.0 / tl.sqrt(variance + eps)` 更快。RMSNorm 步骤直接用 `x_norm = x * rstd`。
- **来源**：`pages/techniques/compute_and_branch.md §2.3`

**A4** `[API文档]` `tl.constexpr` 参数特化
- **内容**：将 `HEAD_DIM=128`、`Q_HIDDEN`、`KV_HIDDEN`、`HAS_BIAS=True` 提升为 constexpr，消除编译时不确定分支。
- **来源**：`pages/techniques/compute_and_branch.md §1.1`

**A5** `[API用法]` `@triton.autotune` — 自动调优 BLOCK_T
- **内容**：对 `BLOCK_T`（每个 program 处理的 token 数，即 batch 行数）做 autotune。推荐配置：`[triton.Config({'BLOCK_T': bt}) for bt in [1, 2, 4, 8, 12]]`。batch=12 时可直接设 BLOCK_T=1（每 program 处理1行）。
- **来源**：`pages/api/triton_api_cheatsheet.md`

### 潜在瓶颈知识单元（B 系列）

**B1** `[潜在瓶颈]` fp32 精度损失 — RMSNorm 方差计算
- **触发条件**：即使 bf16 动态范围大，sum of squares 仍需 fp32 保证精度
- **修复**：load 后立即 `.to(tl.float32)`，variance 计算全程 fp32，输出 cast 回 bf16

**B2** `[潜在瓶颈]` Q/K heads 数量不对称
- **触发条件**：q_hidden=6144(48 heads) vs kv_hidden=1024(8 heads)，不同数量 heads，并行策略需兼顾
- **修复**：以 batch×q_num_heads 为 grid，K 段按 head 索引取模映射到 kv_num_heads

**B3** `[潜在瓶颈]` sin/cos 广播维度处理
- **触发条件**：sin/cos shape=[bsz,1,1,128]，需广播到每个 head
- **修复**：按 batch_idx 加载对应 sin/cos 行，所有 head 共享同一 batch 的 sin/cos

**B4** `[潜在瓶颈]` 小 batch 调度效率低
- **触发条件**：batch=12，q_heads=48，total programs=576，910B2 有 24 个 AI core（tail wave）
- **修复**：576 / 24 = 24 waves，整除无 tail，调度均匀

### 调优经验知识单元（E 系列）

**E1** `[调优经验]` `exp_fused_qkv_rmsnorm_rope_001` — QKV split + RMSNorm + RoPE 三合一
- **适用条件**：QKV 分割后接 RMSNorm + RoPE，head_dim≥64
- **性能提升**：25%
- **核心优化**：单 kernel 完成 split/norm/rope 三步；fp32 variance；cos/sin 按 token 加载复用所有 head
- **关键代码**：`rstd = tl.rsqrt(var + eps); x_hat = x * rstd * weight + bias; x1, x2 = x_hat[:hd//2], x_hat[hd//2:]; o = torch.cat([-x2, x1]) * sin + x_hat * cos`

**E2** `[调优经验]` `exp_fused_split_qk_norm_001` — Q/K 分别 RMSNorm 单核融合
- **适用条件**：Q 和 K 分别做 RMSNorm（可不同 weight），head_dim=128
- **性能提升**：35%
- **核心优化**：同一 program 处理1个 Q head 的 RMSNorm+RoPE；constexpr HAS_BIAS；fp32 variance

**E3** `[调优经验]` `exp_rope_npu_001` — Ascend NPU RoPE（cos/sin 复用）
- **适用条件**：Ascend NPU，head_dim≥64
- **性能提升**：35%
- **核心优化**：cos/sin 寄存器驻留（加载一次，所有 head 复用）；`num_stages=2`

### 调优策略知识单元（T 系列）

**T1** `[调优策略]` 算子融合策略 — 消除中间 HBM 写
- **核心方法**：split/RMSNorm_Q/RMSNorm_K/RoPE_Q/RoPE_K 合并为单一 kernel；寄存器内传递中间结果

**T2** `[调优策略]` 并行策略 — batch × q_num_heads
- **内容**：grid = (bsz * q_num_heads,)，每个 program 处理1个 (batch, head) 对。对 K 段：head_idx_k = head_idx_q // (q_num_heads // kv_num_heads)（若 GQA），或直接用 head_idx 取模。注意本算子 Q/K weights 独立，需分别 normalize。

**T3** `[调优策略]` cos/sin 广播处理
- **内容**：sin/cos shape=[bsz,1,1,128]，按 batch_idx 定位基址，head 维度广播。每 program 按 batch_idx = pid // q_num_heads 加载一次 sin/cos，对该 batch 所有 head 复用。

**T4** `[调优策略]` V passthrough 不归一化
- **内容**：V 段不做 RMSNorm 和 RoPE，直接原样输出。kernel 内只需将 V 段 load 后 store 到输出，或 passthrough（写回同一地址时可跳过）。注意输出格式：q[bsz,-1], k[bsz,-1], v 保持原始切片。

### 代码骨架知识单元（S 系列）

**S1** `[代码骨架]` split_qkv_rmsnorm_rope 融合骨架（bfloat16 适配）

```python
@triton.jit
def split_qkv_rmsnorm_rope_kernel(
    qkv_ptr, q_out_ptr, k_out_ptr, v_out_ptr,
    sin_ptr, cos_ptr,
    q_weight_ptr, k_weight_ptr, q_bias_ptr, k_bias_ptr,
    bsz, q_hidden, kv_hidden,
    stride_batch,
    Q_NUM_HEADS: tl.constexpr, KV_NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr, HALF_DIM: tl.constexpr,
    eps,
):
    pid = tl.program_id(0)  # 每个 program 处理1个 (batch, q_head)
    batch_idx = pid // Q_NUM_HEADS
    q_head_idx = pid % Q_NUM_HEADS
    k_head_idx = q_head_idx % KV_NUM_HEADS  # GQA 映射或等比

    # 偏移计算
    q_base = qkv_ptr + batch_idx * stride_batch + q_head_idx * HEAD_DIM
    k_base = qkv_ptr + batch_idx * stride_batch + q_hidden + k_head_idx * HEAD_DIM
    v_base = qkv_ptr + batch_idx * stride_batch + q_hidden + kv_hidden + ...

    # 加载 Q head 数据
    offs = tl.arange(0, HEAD_DIM)
    q = tl.load(q_base + offs).to(tl.float32)

    # RMSNorm on Q
    var_q = tl.sum(q * q) / HEAD_DIM
    rstd_q = tl.rsqrt(var_q + eps)
    w_q = tl.load(q_weight_ptr + offs).to(tl.float32)
    b_q = tl.load(q_bias_ptr + offs).to(tl.float32)
    q_norm = q * rstd_q * w_q + b_q

    # RoPE on Q
    sin_base = sin_ptr + batch_idx * HEAD_DIM
    cos_base = cos_ptr + batch_idx * HEAD_DIM
    sin_v = tl.load(sin_base + offs).to(tl.float32)
    cos_v = tl.load(cos_base + offs).to(tl.float32)
    q1 = tl.where(offs < HALF_DIM, q_norm, 0.0)
    q2 = tl.where(offs >= HALF_DIM, q_norm, 0.0)
    neg_q2_cat_q1 = tl.where(offs < HALF_DIM, -tl.load(q_base + offs + HALF_DIM).to(tl.float32), tl.load(q_base + offs - HALF_DIM).to(tl.float32))
    q_rope = neg_q2_cat_q1 * sin_v + q_norm * cos_v

    # Store Q output
    tl.store(q_out_ptr + pid * HEAD_DIM + offs, q_rope.to(tl.bfloat16))

    # Similar for K ...
    # V passthrough ...
```

### 矛盾提示知识单元（X 系列）

**X1** `[矛盾提示]` head_dim=128 的 512B 对齐问题
- **内容**：512B 对齐要求每行 256 个 bf16 元素，但 head_dim=128 只有 256B。先用 boundary_check 模式验证正确性。

**X2** `[矛盾提示]` V passthrough 输出格式
- **内容**：原始 baseline 中 v 直接作为 qkv.split 的输出，不做 reshape，保持 [bsz, kv_hidden] 形状。确保 kernel 中 v_out 与输入 v 段相同（可直接 load+store，或 passthrough 原 qkv 地址段）。

---

## 复用提示

- 差异关键词 `bfloat16` 相比 `fp16`：动态范围更大，精度要求一致（fp32 中间计算），API 使用相同
- Q/K heads 不对称（48 vs 8）是本算子的重要特征，需在 grid 设计中体现
