---
query_type: codegen_init
query_keywords: [decode_mla, attention, paged_kv_cache, softmax, bfloat16]
created: 2026-05-06
last_updated: 2026-06-10T14:13:00
hit_count: 2
context_summary: "Decode-phase MLA (Multi-Latent Attention) with paged KV cache, GQA, online softmax, bfloat16, Ascend 910B2"
top_units: [E1_exp_mla_paged_attn_001, E2_exp_paged_attn_decode_001, S1_flash_attention_causal, T1_online_softmax, C1_UB_192KB]
stale: false
---

## 检索问题（Problem Statement）

为 decode-phase MLA (Multi-Latent Attention) 生成高性能 Triton-Ascend kernel。
Q 分解为 nope(512) + rope(64)，paged KV cache block_size=128，GQA H_Q=128 H_KV=1，bfloat16。

**触发上下文**：代码生成初始化，从零构建算子知识包
**检索范围**：attention + paged_kv_cache + decode + MLA + bfloat16

---

## 知识单元列表

### 硬件约束

知识单元 C1 [硬件约束] UB 总用量 ≤ 192KB
  Ascend 910B2 UB = 192KB。MLA decode 需保持 Q(nope+rope) + K_block + V_block + softmax 状态在 UB 内。
  BLOCK_H heads × (BLOCK_DNOPE + BLOCK_DROPE) + BLOCK_N × BLOCK_DNOPE × 2 (K+V) ≤ 192KB
  来源: hardware/constraints_cheatsheet.md C1

知识单元 C2 [硬件约束] Grid ≤ 65535
  Grid B×H_Q_groups = 16×(128/BLOCK_H)。当 BLOCK_H=1 时 grid=2048，安全。
  来源: hardware/constraints_cheatsheet.md C2

知识单元 C5 [性能约束] int64 scalar fallback
  tl.arange 返回 int64，比较/加法退化标量。用 .to(tl.float32) 规避。
  来源: hardware/constraints_cheatsheet.md P1/P2

### 算子分类

知识单元 OP [算子分类] operator_class = "attention"
  判断依据: softmax(QK^T) @ V, paged KV cache, decode phase
  详细页面: pages/ops/attention.md
  经验目录: raw/experiences/attention/

### API 知识单元

知识单元 A1 [API用法] tl.load / tl.store
  Paged KV 加载需手动计算偏移 (page_loc * stride + offset)。
  来源: api/triton_api_cheatsheet.md

知识单元 A2 [API用法] tl.dot — 矩阵乘
  用于 Q@K^T 和 P@V。输入须 rank-2。fp32 accumulator 默认开启。
  来源: api/triton_api_cheatsheet.md

知识单元 A3 [API用法] tl.trans — 转置
  K 加载为 [BLOCK_N, D]，需 tl.trans 为 [D, BLOCK_N] 做 Q@K^T。
  来源: api/triton_api_cheatsheet.md

知识单元 A5 [API用法] tl.exp / tl.maximum / tl.sum — Online Softmax
  在线 softmax 需要 exp, max, sum。均在 fp32 下计算。
  来源: api/triton_api_cheatsheet.md

### 瓶颈预判

知识单元 B1 [潜在瓶颈] MTE2 带宽瓶颈（paged KV 非连续访问）
  Paged KV cache 物理块地址不连续，每页需查 block_table 获取物理地址。
  症状: MTE2 时间占比高
  修复: 减少页加载次数，增大 BLOCK_N

知识单元 B2 [潜在瓶颈] GQA head 间低利用率
  128 个 Q head 共享 1 个 KV head，KV 数据重复读取。
  修复: BLOCK_H 分组多个 Q head 共享同一 KV 加载（参考 exp_mla_paged_attn_001）

知识单元 B3 [潜在瓶颈] fp16/bf16 累加精度
  Online softmax 的 m_i, l_i, acc 必须 fp32。
  来源: techniques/data_types.md

### 调优经验

知识单元 E1 [调优经验] exp_mla_paged_attn_001 — MLA Paged Attention (45%)
  适用条件: MLA + paged KV + GQA, bf16/fp16, 910b2
  核心优化: Q 分解 nope/rope + 分页 block_table 索引 + online softmax + BLOCK_H 多头分组
  关键: V 与 K_nope 共享 buffer（MLA 特性）
  经验文件: raw/experiences/attention/exp_mla_paged_attn_001.yaml
  参考代码: raw/experiences/raw_experiences/extracted_kernels/175-_paged_mla_fwd_kernel.py

知识单元 E2 [调优经验] exp_paged_attn_decode_001 — Paged Decode (30%)
  适用条件: decode phase + paged KV, fp16/bf16, 910b2
  核心优化: make_block_ptr 加载 KV + online softmax + GQA head 映射
  经验文件: raw/experiences/attention/exp_paged_attn_decode_001.yaml
  参考代码: raw/experiences/raw_experiences/extracted_kernels/255-paged_decode_kernel.py

知识单元 E3 [调优经验] exp_flashattn_gqa_001 — Flash GQA (40%)
  适用条件: GQA, 大 seq
  核心优化: SIMD index select 多头复用
  经验文件: raw/experiences/attention/exp_flashattn_gqa_001.yaml

### 调优策略

知识单元 T1 [调优策略] Online Softmax
  核心: m_i(running max), l_i(running sum), acc(累加器) 全部 fp32
  每个 KV block: scores = Q@K^T*scale → m_ij = max(m_i, max(scores)) → p = exp(scores-m_ij)
  → alpha = exp(m_i-m_ij) → l_i = l_i*alpha + sum(p) → acc = acc*alpha + p@V → m_i = m_ij
  最终: out = acc / l_i
  来源: techniques/online_algorithms.md

知识单元 T2 [调优策略] BLOCK_H 多头分组（MLA 专用）
  将多个 Q head 打包到一个 program，共享同一 KV 加载。
  BLOCK_H = min(q_heads_per_kv_head, 合适值)
  效果: 减少 KV 重复读取，提升带宽利用率。
  来源: exp_mla_paged_attn_001

### 代码骨架

知识单元 S1 [代码骨架] flash_attention_causal.py
  骨架路径: pages/skeletons/flash_attention_causal.py
  需修改: 添加 paged KV 索引 + Q nope/rope 分解 + BLOCK_H 多头分组

知识单元 S2 [代码骨架] 175-_paged_mla_fwd_kernel.py (MLA 专用参考)
  路径: raw/experiences/raw_experiences/extracted_kernels/175-_paged_mla_fwd_kernel.py
  关键: 直接适用的 MLA paged attention kernel 参考

---

## 复用提示

含 decode + attention + paged → 本缓存（精确）
含 mla + attention → 本缓存（相似）
含 paged_kv_cache + softmax → 本缓存（部分复用 online softmax + paged 索引）

---

## 附加补丁 (compile_error)

知识单元 P0_broadcast_001 [调试建议] 显式维度扩展 [:, None] / [None, :]
  Triton Ascend 中，一维张量 [N] 不能自动广播到二维 [N, M]。
  修复: p[:, None] → [N, 1] 可与 [N, M] 广播。

---

## 附加补丁 (compile_error - UB overflow)

知识单元 P0_ub_overflow_001 [调试建议] 减小 BLOCK_N 或分块 D 维度
  BLOCK_N=64 时 k_nope+v+pv fp32 多缓冲后超 UB 192KB。
  修复: BLOCK_N 降至 16，单 program UB 占用约 70KB，多缓冲 2x 后约 140KB，安全。
