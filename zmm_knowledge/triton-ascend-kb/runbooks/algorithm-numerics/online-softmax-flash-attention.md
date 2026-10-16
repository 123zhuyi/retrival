---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: algorithm-numerics
dimension: "3.3"
title: Online Softmax / Flash Attention 单遍稳定计算
description: Flash Attention 用 online softmax 在 K/V 循环中维护 running max/sum/acc，将 attention 内存从 O(S²) 降到 O(S)，同时避免 exp 溢出。
tags: [online_softmax, flash_attention, running_max, running_sum, numerical_stability]
confidence: verified
status: verified
created_at: "2026-08-10T00:00:00Z"
updated_at: "2026-09-30"
---

## OPT-24 Online Softmax / Flash Attention 单遍稳定计算

【优化维度】：数值｜数值稳定
【泛化层级】：Triton通用（迁移条件：softmax、attention、log-sum-exp 等需要跨块归约的场景；失效边界：序列很短且可一次装入 UB 的简化场景）

<原则：标准 softmax 需要 O(S²) 的 attention 矩阵，大序列时无法放入 UB。Online softmax 在 K/V 循环中维护 running max、running sum 和 running acc，每块只加载当前 K/V block，将内存复杂度降到 O(S)，同时通过减最大值保证 exp 不溢出。>

**Online Softmax 更新公式**：

```
m_ij = max(m_i, max(scores_block))        # 新的 running max
p = exp(scores_block - m_ij)              # 稳定 exp
l_ij = sum(p)                             # 局部 denominator
alpha = exp(m_i - m_ij)                   # 旧和修正因子
l_i = l_i * alpha + l_ij                 # 修正 running denominator
acc = acc * alpha + (p @ V_block)        # 修正 running numerator
m_i = m_ij                               # 更新 running max
```

最终输出：`output = acc / l_i`

**内存对比**：

| 方法 | Attention 内存 | S=4096, fp16 |
|---|---|---|
| 标准 softmax | O(S²) | 32MB |
| Online softmax | O(S) | ~48KB |

**Triton 模式**：

```python
@triton.jit
def flash_attention_kernel(Q, K, V, Out, ...):
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for k_start in range(0, seq_len, BLOCK_N):
        k = tl.load(K_ptr + k_start * d + k_offs, mask=k_mask)
        scores = tl.dot(q, tl.trans(k)) * scale

        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(scores - m_new[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_new)

        l_i = l_i * alpha + l_ij
        v = tl.load(V_ptr + k_start * d + v_offs, mask=v_mask)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new

    output = acc / l_i[:, None]
    tl.store(Out_ptr + ..., output.to(tl.float16), ...)
```

**关键注意点**：

- `m_i`、`l_i`、`acc` 必须全程 fp32。
- `p` 与 `v` 的 `tl.dot` 前需 cast 到合适精度（如 fp16）。
- causal mask 可在 `scores` 处通过 `m_ij` 前的 mask 实现。

**坏实践（反例）**：
```python
scores = tl.dot(q, tl.trans(k))
probs = tl.softmax(scores)  # 物化整个 S×S attention 矩阵
```
后果：大序列 UB overflow，且 exp 可能溢出。✅ 正解即 online softmax 单遍计算。

## 已知实例（P3 挂载）

### Experience YAML 反链
- `exp_fused_attention_002` — 内存优化+数值稳定优化 （收益 50%）
  - 适用于大序列长度（seq_len≥1024）的Attention计算，分块计算减少内存占用（从O(S²)降至O(S)），在线Softmax提升数值稳定性，910b2/910b3硬件上性能提升显著

### 边界扩展（近期实测，架构特异）
- **CrossformerAttention 分组窗口 MHA（flash_attention，ascend910b，2026-09-21）**
  - **结构负向洞察（把 OT 边界扩展到「N 可物化时 flash 反而不劣化并劣化」）**：49 case 中 29 个
    full-window（G=N 最大 7921）、20 个 grouped。对 N≤7921 的可物化序列，flash 单 kernel 的
    per-KV-tile **跨核 CV 同步**成本（同一 FA §1.1：per-KV-iteration 固定 sync 与 tile 无关）随 G 线性增长，
    cube 空等；而 CANN（aclnn）`BatchMatMul(Q,K^T)→Softmax→BatchMatMul` 整张物化 N×N 让矩阵引擎
    一次吃下、无逐块同步，实测快 **~1000x**（G≥3000 时 impl 500~1660ms vs aclnn 0.07~2ms）。
  - **教训**：online softmax 的 O(S) 红利仅在 S 大到「物化 S² 不可承受」时成立；当 S 可承受
    （本 case ≤7921 → N² fp16 ≤125MB）时，「避免物化」红利为零，而逐块 CV 同步是纯负担。
    此类应接 CANN 的 **非 flash batched-matmul 注意力路径**（整块 QK^T→softmax→PV），
    而非强行走分块 online softmax。

    > 模板约束与教训冲突示例：此处 precheck L1.3 强制作分块 online softmax，在本 shape 集是反杠杆；
    > 需 kernel 分裂（优化点 17）打破模板分块约束才能接 batched-matmul 路径。

  - **证据**：opt 全-D 扁平结构亦回归——impl 128.536ms（`output/opt_iter_0/optimized_perf_result.json`）
    vs baseline 118.376ms（`output/iter_0/perf_result.json`），speedup_vs_baseline=1.086 <1.0 → 回退；
    完整分析见轨迹 `TRITON_VS_CANN.md` 差异 7。flash_attention 剩余适用场景：长序列大 head-dim / 大 S batch，
   或真正无法物化 N² 的极长上下文。

- **SelfAttention（flash_attention + 投影 GEMM，ascend910b，2026-09-21）**
  - **正向 tiling 洞察**：大 S attention 的 per-KV-iteration 跨核同步成本与 tile 无关（FA §1.1），
    迭代数 = B·H·ceil(S/BSQ)·ceil(S/BSK)。BSK 64→128、BSQ 收缩保面积 8192 得 **64×128**
    → KV 迭代减半，speedup_vs_torch 0.0797→0.1193（+49%，opt_iter_0）。
  - **GEMM K 归约独立杠杆**：D=4096 的投影/输出 GEMM 随 K 归约迭代数线性，half BK 64→128、
    fp32 单独较小 BK 32→64（acc 128x128 fp32=64KB UB 紧）→ 0.1193→0.1246→0.1250。
  - **天花板**：含 softmax 的 kernel 必为 MIX_AIC（~4.5us/发地板 ×3 kernel），vs aclnn 大 shape
    差距 ~40× 不可闭合；target 口径（vs torch eager / vs aclnn, perf.json 注入）决定天花板判据，须先行。
  - **证据**：speedup_vs_torch opt_iter_0=0.1193 / opt_iter_1=0.1246 / opt_iter_2=0.1250
    （output/opt_iter_{n}/optimized_perf_result.json，geomean；verify 50/50）。

- **SelfAttention 互补 merge：FULL_D 满块分裂（ascend910b，2026-09-30）**
  - **正向洞察（直击 per-KV-iteration 同步与 vbrc）**：将 K3 KV 主循环按
    `n_full=(S//BSK)*BSK` 分裂为「满块零-mask（FULL_D 时连 d-mask 也免）+ 尾块 KMASKED」。
    满块段消掉每迭代的 vbrc/幽灵列 where——不只是帮助大 shape 目标 case，也让 HD 对齐 16
    的主流 half case（64×128）普遍省一次向量分支，带来大量 0.90-0.93× 的全面小幅提升。
  - **互补 merge 收益超「逐 case 理想互补」预算**：当前版 hi/lo 2-dot（opt_iter_2, 0.5577）与
    早期版单 dot 骨架互补（FULL_D 分裂 + 候选 tile 阶梯优先保大 BSK + fp32 离线逐 shape 标定表
    + 编译降档回退——全部保留 hi/lo 精度主线、不搬单 dot）→ 同卡 today **0.5577→0.7151（+28.2%）**，
    超市前逐 case 取单点最小 0.6296（+12.9%）的上限，恰因 FULL_D 分裂惠及全体主流 case。
  - **证据**：speedup_vs_torch opt_iter_9=0.7151 / opt_iter_2(today)=0.5577，
    impl avg 4.80→3.74ms（output/opt_iter_9/optimized_perf_result.json 对
    output/opt_iter_2/optimized_perf_result_today.json；同卡 6；verify 50/50）。
    逐 case：9 更快 41 / 2 更快 8 / 平 1。

<!-- okf:related:start -->

# 相关

- 实践案例: [BlockSparseAttention (block-sparse mask) on Ascend NPU](../../ops/block-sparse-attention.md) — 主题相关：共享 flash_attention, online_softmax
- 实践案例: [FlashAttentionScoreGrad training-backward on Ascend NPU](../../ops/flash-attention-score-grad.md) — 主题相关：共享 flash_attention, online_softmax
- 实践案例: [FlashAttentionScore training-forward on Ascend NPU](../../ops/flash-attention-score.md) — 主题相关：共享 flash_attention, online_softmax
- 实践案例: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 同属优化知识，主题相关
- 实践案例: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](../../ops/fused-infer-attention-score.md) — 主题相关：共享 flash_attention, online_softmax
- 实践案例: [IncreFlashAttention inference-decode on Ascend NPU](../../ops/incre-flash-attention.md) — 主题相关：共享 flash_attention, online_softmax
- 实践案例: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](../../ops/kv-quant-sparse-flash-attention.md) — 主题相关：共享 flash_attention
- 实践案例: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](../../ops/moe-gating-top-k-softmax-v2.md) — 同属优化知识，主题相关
- 实践案例: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](../../ops/moe-gating-top-k-softmax.md) — 同属优化知识，主题相关
- 实践案例: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](../../ops/nsa-selected-attention.md) — 主题相关：共享 online_softmax
- 实践案例: [PromptFlashAttention inference/prefill on Ascend NPU](../../ops/prompt-flash-attention.md) — 主题相关：共享 flash_attention, online_softmax
- 实践案例: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 主题相关：共享 online_softmax
- 实践案例: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](../../ops/ring-attention-update.md) — 主题相关：共享 online_softmax
- 实践案例: [SparseFlashAttention inference (sparse KV) on Ascend NPU](../../ops/sparse-flash-attention.md) — 主题相关：共享 flash_attention, online_softmax
- 开发指南: [Fused Attention / Flash Attention 高性能算子](../../reference/op-class-attention.md) — 主题相关：共享 flash_attention
- 开发指南: [归一化算子优化](../../reference/op-class-normalization.md) — 主题相关：共享 numerical_stability
- 开发指南: [归约算子优化](../../reference/op-class-reduction.md) — 主题相关：共享 numerical_stability
- 开发指南: [Softmax / LogSoftmax 算子优化](../../reference/op-class-softmax.md) — 主题相关：共享 numerical_stability, online_softmax
- 相关主题: [Flash Attention Causal Skeleton](../../reference/skeleton-flash_attention_causal.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_attn_reduce_stage2_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_crossentropy_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: online softmax](../compute-pipeline/exp_sdpa_infer_ascend_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_softmax_non_inner_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_softmax_online_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: online softmax](../memory-sram/exp_attn_sinks_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: online softmax algorithm](../memory-sram/exp_flashattn_fwd_001.md) — 同属优化知识，主题相关
- 相关主题: [长序列注意力分块计算 + 在线 Softmax（内存 O(S²)→O(S)）](../memory-sram/exp_fused_attention_002.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: score-tile reuse with online softmax](../memory-sram/exp_sdpa_flash_001.md) — 同属优化知识，主题相关
- 相关主题: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../meta-skills/attention-fusion-failure-diagnosis.md) — 主题相关：共享 flash_attention
- 相关主题: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](exp-stable-softmax-rmsnorm.md) — 主题相关：共享 numerical_stability
- 相关主题: [exp 前减维度最大值防溢出（safe-exp/softmax 数值稳定）](exp_stability_001.md) — 同属优化知识，主题相关
- 相关主题: [统计归约必须用 fp32 累加器，禁止 fp16/bf16 直接累加](fp32-reduction-accumulator.md) — 主题相关：共享 numerical_stability

<!-- okf:related:end -->
