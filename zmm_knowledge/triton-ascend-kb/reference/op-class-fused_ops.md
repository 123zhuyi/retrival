---
schema_version: okf.v1
kind: guide
type: programming_guide
source_family: reference
title: 融合算子综合页
description: 融合算子综合页
tags: [fusion, gemm, rmsnorm, rope, attention, moe, fused_ops, 910b2, 910b3, operator_class]
confidence: inferred
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/pages/ops/fused_ops.md
created_at: '2026-08-10T02:56:36Z'
updated_at: '2026-08-10T02:56:36Z'
---

sources:
  - url: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/pages/ops/fused_ops.md
    role: primary

> 迁移备注：本卡由 `cannbot-skills_5171/ops/triton-knowledge-retrieval/knowledge/pages/ops/fused_ops.md` 迁移而来，原始为算子类别设计页，未做内容改写，仅转 OKF frontmatter。

# 融合算子综合页

> 汇总最常见的多算子融合组合，每个组合给出完整 kernel 结构、参数建议和性能收益。融合的核心目标：消除中间 tensor 的 HBM 读写。

---

## 融合组合速查

| 融合组合 | 典型场景 | 性能提升 | 跳转 |
|---------|---------|---------|------|
| Linear + GELU/SiLU | FFN 第一层 | 30% | [→ Fusion-1](#fusion-1-linear--gelusilu) |
| Linear + RMSNorm + RoPE | Attention QKV 投影 | 45% | [→ Fusion-2](#fusion-2-linear--rmsnorm--rope) |
| Skip + LayerNorm | Transformer 残差块 | 30% | [→ Fusion-3](#fusion-3-skip--layernorm) |
| RMSNorm + Quantization | 量化推理归一化 | 50% | [→ Fusion-4](#fusion-4-rmsnorm--quantization) |
| Flash Attention (QK^T + Softmax + AV) | 所有 attention | 50% | [→ Fusion-5](#fusion-5-flash-attention) |
| MoE Scatter + Expert GEMM | MoE FFN | 35%+ | [→ Fusion-6](#fusion-6-moe-scatter--expert-gemm) |
| GEMM + Bias + Activation | 通用线性层 | 35% | [→ Fusion-7](#fusion-7-gemm--bias--activation) |
| Phase Split + Barrier 外提 | 多阶段反向 kernel | 10-30% | [→ Fusion-8](#fusion-8-phase-split--barrier-外提多阶段融合通用模式) |

---

## Fusion-1: Linear + GELU/SiLU

**场景**：FFN 第一层 `y = GELU(W @ x + b)`

**结构**：
```
GEMM (Cube) → [L0C 寄存器] → GELU/SiLU (Vector) → Store
```

**关键参数**：
- `BLOCK_M`: 128 或 256
- `BLOCK_N`: 128 或 256（满足 512B 对齐）
- `BLOCK_K`: 256
- `GROUP_SIZE`: 8/16/32（autotune）
- 后处理激活在 GEMM 累加之后、store 之前执行

**实现要点**：
1. 在 matmul 的 K 轴累加循环结束后，对 accumulator 直接施加激活函数
2. 激活函数在 L0C 寄存器上操作，无额外 HBM 读写
3. 使用 `dl.async_task(scope="cube")` 与 `scope="vector"` 分离 GEMM 和激活的 Cube/Vector 调度

**代码骨架**：见 skeletons/gemm_activation.py

**性能提升**：~30%（消除 1 次 HBM 写 + 1 次 HBM 读）

**约束**：
- 激活函数必须为 element-wise（GELU、SiLU、ReLU、Sigmoid 均可）
- 不适用于需要跨元素通信的激活（如 Softmax）

---

## Fusion-2: Linear + RMSNorm + RoPE

**场景**：`Q = RoPE(RMSNorm(W_q @ x))`，QKV 投影后归一化并加旋转位置编码

**结构**：
```
GEMM (Cube) → [L0C] → RMSNorm (Vector, fp32) → RoPE (Vector) → Store
```

**关键参数**：
- GEMM 参数同 Fusion-1
- RMSNorm 在 Vector 单元上，n_cols = head_dim（通常 64-128），可单次装入寄存器
- RoPE 需要 cos/sin 表，预加载进寄存器（1 次 HBM 读）

**实现要点**：
1. GEMM 累加结束后，accumulator 即为 Q/K 投影结果
2. 在同一 kernel 内对 head 维度做 RMSNorm（fp32 variance）
3. 接着做 RoPE 复数旋转：`[x0,x1,...] → [x0*cos-x1*sin, x0*sin+x1*cos,...]`
4. Q、K 共享同一 RMSNorm weight（或各自独立），V 直接 passthrough

**性能提升**：~45%（消除 2 次 HBM 中间存储：GEMM output + RMSNorm output）

**代码骨架**：见 skeletons/gemm_rmsnorm_rope.py

**约束**：
- head_dim 须能装入寄存器（通常 64-128 无问题）
- cos/sin 表尺寸 = seq_len × head_dim，若 seq_len 大需分块加载

---

## Fusion-3: Skip + LayerNorm

**场景**：Transformer 残差连接 `y = LayerNorm(x + residual)`

**结构**：
```
Load x, residual → [寄存器] x + residual (fp32) → mean → variance → normalize → weight/bias → Store
```

**关键参数**：
- 整行 hidden_dim 须装入 BLOCK_SIZE_N（若太大需 two-pass）
- 全程 fp32 中间计算

**实现要点**：
1. 同时 load x 和 residual，在 fp32 寄存器中相加
2. 对 fp32 结果做 pass-1：计算 mean
3. pass-2：计算 variance，归一化，应用 weight/bias
4. 使用 `constexpr HAS_RESIDUAL: tl.constexpr` 编译时消除分支

**性能提升**：~30%（消除 1 次中间 tensor 存储）

**代码骨架**：见 [skeletons/rmsnorm_fused.py](skeleton-rmsnorm_fused.md)（含 skip 变体）

**约束**：
- 若 n_cols > UB 容量（fp32: ~49K elements），需列分块 two-pass，此时 residual add 在 first pass 内完成

---

## Fusion-4: RMSNorm + Quantization

**场景**：量化推理中 `y_int8 = quantize(RMSNorm(x))`

**结构**：
```
Load x (fp16/bf16) → variance (fp32) → rstd → normalize (fp32) → compute scale → quantize → Store (int8/fp8)
```

**关键参数**：
- 量化 scale 可以 per-tensor 或 per-channel
- per-channel scale 需在 normalize 之后对每个 channel 计算 abs_max

**实现要点**：
1. 计算 fp32 方差和 rstd（同普通 RMSNorm）
2. 归一化后的输出为 fp32
3. 计算量化 scale：`scale = max(abs(y)) / 127.0`（对称量化）
4. 量化：`y_int8 = round(y / scale).clamp(-128, 127).to(int8)`

**性能提升**：~50%（消除 RMSNorm 和量化之间的 HBM 读写）

**约束**：
- 静态量化 scale 需提前知道；动态 per-tensor scale 可在 kernel 内计算
- 输出 int8 须对齐 128B（64 个 int8 elements）

---

## Fusion-5: Flash Attention

**场景**：完整 attention `output = softmax(QK^T / sqrt(d_k)) @ V`

**结构**（在 Q 的 outer loop, K/V 的 inner loop 内）：
```
Q tile (寄存器) × K tile (寄存器) → scores → online softmax update → × V tile → acc update
```

**Online Softmax 核心公式**：
```python
# 初始化
m_i = -float('inf')    # running max
l_i = 0.0              # running sum of exp
acc = zeros([BLOCK_M, HEAD_DIM])

# 对每个 K/V 块
for j in range(0, seq_len, BLOCK_N):
    k_tile = load_k(j)
    scores = q_tile @ k_tile.T / scale  # [BLOCK_M, BLOCK_N]
    if causal: scores += causal_mask(q_pos, j)
    
    m_ij = max(m_i, scores.max(axis=1))  # per-row max
    p = exp(scores - m_ij[:, None])
    l_ij = p.sum(axis=1)
    
    alpha = exp(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None] + p @ load_v(j)
    m_i = m_ij

output = acc / l_i[:, None]
```

**性能提升**：~50%（O(S²)→O(S) 显存，消除完整 attention matrix 存储）

**代码骨架**：见 [skeletons/flash_attention_causal.py](skeleton-flash_attention_causal.md)

**约束**：
- BLOCK_N 须满足 512B 对齐
- GQA/MQA：用 `tl.load(k_ptr + kv_head_idx * stride)` 共享 KV head

---

## Fusion-6: MoE Scatter + Expert GEMM

**场景**：MoE 层 `output[token] = Expert[routing[token]] @ input[token]`

**结构**：
```
top-k routing (CPU/host) → token_binning → [per-expert] Scatter + Expert GEMM → Gather output
```

**两阶段实现**：

**Phase 1 — Token Scatter（gather_scatter kernel）**：
```python
# 按 expert 分组，每 expert 一个 bin
# bin[e] = indices of tokens routed to expert e
for expert in range(num_experts):
    tokens_for_expert = gather(tokens, routing_table[expert])
    write to expert_buffer[expert * max_capacity:]
```

**Phase 2 — Expert GEMM（matmul kernel，按 expert 批量执行）**：
```python
# grouped GEMM：每个 expert 一个矩阵乘
for expert in range(num_experts):
    token_count = bin_count[expert]
    if token_count == 0: continue
    output[expert] = expert_weight[expert] @ expert_input[expert, :token_count]
```

**性能关键**：
- `exp_gs_bin_003`：Binned dispatch，expert_capacity × num_experts ≥ 1024 时效果最佳
- Dynamic work-stealing（`exp_gs_bin_005`）解决专家负载不均
- Expert GEMM 使用对角 tiling（M=token_count 通常较小，K=hidden, N=ffn_dim）

**约束**：
- expert_capacity 需预先设定（静态）或动态分配
- 空 expert（token_count=0）需 skip 避免无效 GEMM

---

## Fusion-7: GEMM + Bias + Activation

**场景**：`y = activation(W @ x + b)`，通用带 bias 的线性层

**结构**：
```
GEMM (Cube) → [L0C] → bias add (Vector) → activation (Vector) → Store
```

**实现要点**：
1. bias 向量（长度 N）在 K 循环开始前加载进寄存器
2. GEMM 累加完成后，对 accumulator 加 bias
3. 施加激活函数
4. Store

**代码片段**：
```python
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptr + ...)
    b = tl.load(b_ptr + ...)
    acc += tl.dot(a, b)

# 融合 bias + activation
bias = tl.load(bias_ptr + n_offs, mask=n_mask)
acc += bias[None, :]           # broadcast over M
acc = tl.sigmoid(acc) * acc    # SiLU = x * sigmoid(x)

tl.store(c_ptr + ..., acc.to(OUT_DTYPE), mask=...)
```

**性能提升**：~35%

---

## Fusion-8: Phase Split + Barrier 外提（多阶段融合通用模式）

**场景**：一个 kernel 内包含多个逻辑阶段，阶段间存在数据依赖，但阶段内部循环存在条件分支。

**典型结构**：
```
原始：
  for i_k:
    for i_v:
      if i_k == 0:          # 每个 i_v 迭代都判断
        compute dA/dv2/db
      compute dq/dk/dw/dgk
      tl.debug_barrier()    # 每个 i_v 迭代都 barrier

优化后：
  for i_v:                  # Phase 1：V-only
    compute dA/dv2/db
  tl.debug_barrier()        # 只在 Phase 间 barrier 一次
  for i_k:
    for i_v:                # Phase 2：干净内循环
      compute dq/dk/dw/dgk
```

**关键点**：
- 将 `if i_k == 0` 分支外提为独立循环，消除内循环条件分支
- 将 `tl.debug_barrier()` 从循环内移到循环外，减少 barrier 次数
- 确保 Phase 1 的写入在 Phase 2 读取前完成

**性能提升**：10-30%（取决于原始循环中条件分支和 barrier 的频次）

经验：`exp_kda_bwd_wy_structure_opt_001`；详见 [ops/ssm.md](op-class-ssm.md)

---

## 相关页面

- techniques/operator_fusion.md — 融合策略原理与 Cube/Vector 异构调度
- [ops/matmul.md](op-class-matmul.md) — GEMM 基础经验和 tiling 策略
- [ops/attention.md](op-class-attention.md) — Flash Attention 详细实现
- [ops/normalization.md](op-class-normalization.md) — RMSNorm/LayerNorm 精度管理
- [ops/gather_scatter.md](op-class-gather_scatter.md) — MoE scatter/gather 实现
- [ops/ssm.md](op-class-ssm.md) — KDA/Linear Attention 反向 Phase split + barrier 外提实例
- techniques/online_algorithms.md — Online Softmax 完整公式
- techniques/compute_and_branch.md — 分支外提、循环不变量外提
- skeletons/README.md — 各融合组合的完整可运行骨架代码

## 更多融合经验（misc 类目）

### Transformer Block 级融合

| 场景 | 经验ID | 关键技术 | 经验文件 |
|------|--------|----------|----------|
| 完整 Transformer Block | exp_fused_transformer_block_001 | GEMM + Sigmoid + RMSNorm + RoPE 四合一 | yaml |
| SDA 前向（Ascend） | exp_sda_fwd_ascend_001 | Scaled Dot Attention 前向，Ascend 910b2/b3 路径 | yaml |
| SDA 反向 Q | exp_sda_bwd_q_001 | SDA 反向 dQ | yaml |
| SDA 反向 KV（Ascend） | exp_sda_bwd_kv_ascend_001 | SDA 反向 dK/dV，Ascend 专用 | yaml |
| MHC 后融合 | exp_mhc_post_fusion_001 | Multi-Head 计算后融合后处理 | yaml |

### 推理专用融合

| 场景 | 经验ID | 关键技术 | 经验文件 |
|------|--------|----------|----------|
| Spec Dec 验证 | exp_spec_dec_verify_001 | 投机解码（Speculative Decoding）token 验证 | yaml |
| Spec Dec 拒绝采样 | exp_spec_dec_reject_001 | 投机解码拒绝采样 | yaml |
| 推测恢复 | exp_speculative_recovery_001 | Speculative Decoding recovery token 生成 | yaml |
| Top-P 采样 | exp_top_p_sampling_001 | Top-P nucleus sampling | yaml |
| Top-K Pad | exp_topk_pad_001 | Top-K 结果 padding 对齐 | yaml |
| Top-P 过滤 | exp_topp_filter_001 | Top-P logits 过滤 | yaml |
| Logits Penalty 融合 | exp_logits_penalty_fusion_001 | Logits frequency/presence penalty 融合 | yaml |
| 多项式采样 | exp_multinomial_001 | 基于概率的多项式采样 | yaml |
| Rejection Sampler | exp_rejection_sample_001 | 拒绝采样器 | yaml |

### 排序与辅助算子

| 场景 | 经验ID | 关键技术 | 经验文件 |
|------|--------|----------|----------|
| Bitonic Sort | exp_sort_bitonic_001 | N≤1024 的双调排序（Bitonic sort），全 SRAM 完成 | yaml |
| Digit Histogram | exp_digit_hist_001 | 基数排序的 digit histogram 计算 | yaml |
| 数字直方图计数 | exp_hist_count_001 | 直方图计数 | yaml |
| Radix Convert | exp_radix_convert_001 | 基数转换辅助操作 | yaml |
| Shuffle Blockwise | exp_shuffle_blockwise_001 | 块级洗牌操作 | yaml |

### 矩阵专项算子

| 场景 | 经验ID | 关键技术 | 经验文件 |
|------|--------|----------|----------|
| 下三角矩阵求解 | exp_solve_tril_001 | 下三角线性方程组求解 | yaml |
| 下三角求解（另一版） | exp_tril_solve_001 | tril 求解变体 | yaml |
| Outer Product | exp_outer_product_001 | 向量外积（rank-1 update） | yaml |
| Masked Fill | exp_masked_fill_001 | 带 mask 的元素填充 | yaml |
| Element Mul | exp_element_mul_001 | 逐元素乘法变体 | yaml |
| Expand Serial | exp_expand_serial_001 | 串行展开操作 | yaml |

### 训练专项算子

| 场景 | 经验ID | 关键技术 | 经验文件 |
|------|--------|----------|----------|
| GRPO Loss | exp_grpo_loss_001 | GRPO 强化学习策略损失（LogSumExp 归约） | yaml |
| GRPO 反向 | exp_grpo_bwd_001 | GRPO 损失反向传播 | yaml |
| TV Distance | exp_tv_distance_001 | 全变差距离（Total Variation Distance） | yaml |
| GN WB 反向 | exp_gn_wb_backward_001 | GroupNorm weight/bias 梯度 | yaml |
| Cache Store | exp_cache_store_001 | KV cache 写入优化 | yaml |
| Fused Index | exp_fused_index_001 | 索引融合操作 | yaml |
| Fused QKVzba Split | exp_fused_qkvzba_split_001 | QKV+z+b+a 联合分割 | yaml |
| Linear Persistent | exp_linear_persistent_001 | 持久化线性层 | yaml |
| Recompute W/U | exp_recompute_w_u_001 | 激活重计算以节省显存 | yaml |
| Launch Diagonal | exp_launch_diagonal_001 | 对角线启动模式（balanced launch） | yaml |
| Bonus Renew | exp_bonus_renew_001 | RLHF bonus 更新 | yaml |
| Fuse TLX | exp_fuse_tlx_001 | TLX 融合操作 | yaml |

## 被以下页面引用
- workflows/optimization_flow.md — 融合场景路由
- experience_index.md — 融合经验 ID 汇总
- techniques/operator_fusion.md — 反向引用此页列举的融合模式

## 已知实例（P3 挂载）

### Extracted Kernel 反链
- `017-_fwd_grouped_split_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/017-_fwd_grouped_split_kernel.py`）
- `019-_fwd_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/019-_fwd_kernel.py`）
- `020-forward_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/020-forward_kernel.py`）
- `027-_fwd_kernel_stage1.py` — extracted kernel（source: `raw_experiences/extracted_kernels/027-_fwd_kernel_stage1.py`）
- `028-_fwd_kernel_stage2.py` — extracted kernel（source: `raw_experiences/extracted_kernels/028-_fwd_kernel_stage2.py`）
- `068-_fwd_loop_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/068-_fwd_loop_kernel.py`）
- `157-fused_qkvzba_split_reshape_cat_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/157-fused_qkvzba_split_reshape_cat_kernel.py`）
- `163-fused_gdn_gating_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/163-fused_gdn_gating_kernel.py`）
- `191-fused_split_qk_norm_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/191-fused_split_qk_norm_kernel.py`）
- `226-_fused_penalty_temp_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/226-_fused_penalty_temp_kernel.py`）
- `075-_mhc_post_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/075-_mhc_post_kernel.py`）
- `076-_k_fuse_tlx.py` — extracted kernel（source: `raw_experiences/extracted_kernels/076-_k_fuse_tlx.py`）
- `156-recompute_w_u_fwd_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/156-recompute_w_u_fwd_kernel.py`）
- `166-bonus_renew.py` — extracted kernel（source: `raw_experiences/extracted_kernels/166-bonus_renew.py`）
- `213-_store_label_cache_triton_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/213-_store_label_cache_triton_kernel.py`）

<!-- okf:related:start -->

# 相关

- 实现示例: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../ops/mla-preprocess.md) — 主题相关：共享 attention, fusion, rmsnorm, rope
- 实现示例: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../ops/mla-prolog-v3.md) — 主题相关：共享 attention, rmsnorm, rope
- 实现示例: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](../ops/mla-prolog.md) — 主题相关：共享 attention, fusion, rmsnorm, rope
- 实现示例: [RMSNorm-RoPE fused kernel on Ascend 910B1](../ops/rmsnorm-rope.md) — 主题相关：共享 rmsnorm, rope
- 讲解接口: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../runbooks/algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 主题相关：共享 rmsnorm
- 讲解接口: [GEMM 操作数保持 low_dtype，fp32 累加](../runbooks/algorithm-numerics/gemm-low-dtype-fp32-accum.md) — 主题相关：共享 gemm
- 讲解接口: [Compute Optimization: gemm based histogram](../runbooks/compute-pipeline/exp_digit_hist_001.md) — 同属优化知识，主题相关
- 讲解接口: [算子融合模式与约束决策](../runbooks/compute-pipeline/operator-fusion-patterns.md) — 主题相关：共享 fusion, gemm, rmsnorm, rope
- 讲解接口: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](../runbooks/memory-sram/exp_fused_rms_norm_001.md) — 同属优化知识，主题相关
- 讲解接口: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](../runbooks/memory-sram/exp_gemma_rmsnorm_001.md) — 同属优化知识，主题相关
- 配套样例: [Gemm Basic Skeleton](skeleton-gemm_basic.md) — 同属优化知识，主题相关
- 配套样例: [Rmsnorm Fused Skeleton](skeleton-rmsnorm_fused.md) — 同属优化知识，主题相关
- 相关主题: [Fused Attention / Flash Attention 高性能算子](op-class-attention.md) — 主题相关：共享 910b2, 910b3, attention, operator_class
- 相关主题: [卷积算子优化（Causal Conv1D）](op-class-convolution.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Elementwise 算子优化](op-class-elementwise.md) — 主题相关：共享 910b2, 910b3, fusion, operator_class
- 相关主题: [Embedding / 索引拷贝算子优化](op-class-embedding.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Gather/Scatter 算子优化](op-class-gather_scatter.md) — 主题相关：共享 910b2, 910b3, moe, operator_class
- 相关主题: [Loss 函数算子优化](op-class-loss.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [MatMul / GEMM 高性能算子](op-class-matmul.md) — 主题相关：共享 910b2, 910b3, gemm, operator_class
- 相关主题: [归一化算子优化](op-class-normalization.md) — 主题相关：共享 910b2, 910b3, operator_class, rmsnorm
- 相关主题: [归约算子优化](op-class-reduction.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [RoPE（旋转位置编码）算子优化](op-class-rope.md) — 主题相关：共享 910b2, 910b3, operator_class, rope
- 相关主题: [Softmax / LogSoftmax 算子优化](op-class-softmax.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [solve_tril 算子优化（三角矩阵求逆）](op-class-solve_tril.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [SSM / 状态空间模型算子优化](op-class-ssm.md) — 主题相关：共享 910b2, 910b3, operator_class

<!-- okf:related:end -->
