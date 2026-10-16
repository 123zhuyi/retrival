---
query_type: codegen_init
query_keywords: [lightningindexer, topk, score, sparse, bf16, ascend910b1]
created: 2026-07-25T03:35:00+08:00
last_updated: 2026-07-29T03:05:00+08:00
hit_count: 3
context_summary: "LightningIndexer 算子 — 双 kernel (score + topk)，支持 BSND/PA_BSND/TND 布局，按 index 升序输出 top-K 索引与值"
top_units: [C1, C2, C5, C6, OP1, A1, A2, A3, A5, A6, B1, B2, B5, E1, E2, E4, T1, T2, T3, S1]
stale: false
---

## 检索问题（Problem Statement）

为 LightningIndexer 算子生成高性能 Triton-Ascend kernel，实现：
- score = Σ_n weights * ReLU(query @ key^T) * (1 - mask)
- 每 (batch, s1) 行取 top-K
- 输出按 index 升序排列的 sparse_indices 与 sparse_values

## 知识单元列表

### 硬件约束知识单元

知识单元 C1  [硬件约束]  UB 总用量 ≤ 192KB
  每个 program 的 tile 数据量之和不超过 192KB（Atlas 800T A2）。
  score kernel: BLOCK_S1 * BLOCK_S2 * 4B + BLOCK_S2 * BLOCK_D * 2B + BLOCK_S1 * BLOCK_D * 2B ≤ 192KB。
  topk kernel: TILE_SORT * 4B + MAX_K * 8B ≤ 192KB。
  来源: hardware/constraints_cheatsheet.md C1

知识单元 C2  [硬件约束]  Grid 维度上界 65535
  grid 必须是 tuple，各维度乘积 ≤ 65535。本算子使用 persistent grid=(num_cores,)，通过 grid-stride loop 处理全部 task。
  来源: hardware/constraints_cheatsheet.md C2

知识单元 C3  [硬件约束]  512B 行宽对齐
  matmul 的块行宽（BLOCK_N × dtype_bytes）必须是 512B 整数倍。score kernel 中 q@k^T 的 K 维度 D=128，为 512B/2B=256 的约数，满足对齐；建议 BLOCK_D 取 128 的倍数。
  来源: hardware/constraints_cheatsheet.md C3

知识单元 C4  [硬件约束]  禁止 tl.swizzle2d
  Ascend 上行为未定义，本算子不涉及 swizzle。
  来源: hardware/constraints_cheatsheet.md C4

知识单元 C5  [性能约束]  int64 CMP/ADD scalar fallback
  int64 的比较和加法会退化为标量路径。解决：索引用 int32，比较前 .to(tl.float32)。
  来源: hardware/constraints_cheatsheet.md P1/P2

知识单元 C6  [性能约束]  fp16/bf16 accumulation 未升精度
  score 累加必须在 fp32 中进行，避免 bf16 累加导致 top-k 位置偏移。
  来源: hardware/constraints_cheatsheet.md P7

---

### 算子分类知识单元

知识单元 OP1  [算子分类]  operator_class = "fused_score_topk"
  判断依据：forward 包含 matmul(q,k) → ReLU → weighted reduction → mask → topk。
  详细页面：pages/ops/attention.md + pages/ops/gather_scatter.md
  经验目录：raw/experiences/attention/ + raw/experiences/gather_scatter/

---

### API 知识单元

知识单元 A1  [API用法]  tl.load / tl.store / tl.make_block_ptr
  score kernel 用 block pointer 加载 query/key tile；topk kernel 用指针+offset 加载 score 行片段。
  Ascend 特有：care_padding=False 可消除 MTE2→Vector 同步屏障（仅当越界值被 mask 过滤时安全）。
  API文档: raw/triton-ascend-develop-guide/triton_api/Memory_Pointer_Ops/

知识单元 A2  [API文档]  tl.dot / tl.maximum / tl.where
  score kernel 核心：tl.dot(q_fp32, k_fp32.T) → tl.maximum(qk, 0.0) → tl.where(mask, score, -inf)。
  API文档: raw/triton-ascend-develop-guide/triton_api/Math_Ops/

知识单元 A3  [API用法]  tl.max / tl.min / tl.sum
  topk kernel 用 tl.max/tl.min 实现迭代 argmax；等值处理用 tl.min 选择最小/最大 index。
  API文档: raw/triton-ascend-develop-guide/triton_api/Reduction_Ops/

知识单元 A4  [API用法]  tl.num_programs / tl.program_id
  persistent kernel 用 grid=(num_cores,) 配合 grid-stride loop。
  API文档: raw/triton-ascend-develop-guide/triton_api/triton/

知识单元 A5  [API文档]  tl.constexpr
  用于 BLOCK_S1, BLOCK_S2, BLOCK_D, TILE_SORT, MAX_K, IS_BF16, HAS_VARLEN 等编译期常量。
  API文档: raw/triton-ascend-develop-guide/triton_api/triton/

知识单元 A6  [API用法]  禁止 torch.topk / torch.sort 在 kernel 内
  Ascend Triton 对 tl.sort 支持有限，topk 必须手写迭代 argmax。
  来源: template/lightningindexer.md L1.7

---

### 瓶颈预判知识单元

知识单元 B1  [潜在瓶颈]  S1=1 时 score kernel 并行度不足
  触发条件：query shape [1,1,64,128]，仅 1 行。
  修复：使用 persistent grid-stride，单个 program 处理多行或多 S2 块；增大 BLOCK_S2 以提升单 program 工作量。
  关联经验: exp_sparse_attn_topk_001

知识单元 B2  [潜在瓶颈]  topk 小 k 固定开销大
  触发条件：k ≤ 256。
  修复：一次性 bitonic/sorting-network 处理完整行（若 UB 允许），或增大 TILE_SORT 减少 tile 循环。
  关联经验: exp_sort_bitonic_001

知识单元 B3  [潜在瓶颈]  PA_BSND host gather 引入额外拷贝
  触发条件：layout_key="PA_BSND"。
  修复：在 forward 中用 key_pa[block_table.long()].contiguous().view(...) 归约为 BSND；或 kernel 内 block gather。
  关联经验: exp_paged_kv_cache_001

知识单元 B4  [潜在瓶颈]  大 S2 时 score workspace 写带宽
  触发条件：S2=32768，score workspace [B*S1, S2] fp32。
  修复： score kernel 内按 BLOCK_S2 分块累加后写回，避免一次性大 tile。

知识单元 B5  [潜在瓶颈]  topk 等值处理不一致
  触发条件：score 存在大量相等值。
  修复：按 (value, index) 升序，等值淘汰 index 最大者。
  来源: template/lightningindexer.md L1.11

---

### 调优经验知识单元

知识单元 E1  [调优经验]  exp_sparse_attn_topk_001 — 稀疏注意力 TopK
  适用条件：Attention + TopK，长序列，fp16/bf16。
  性能提升：40%
  核心优化：仅迭代 topk 块、block pointer、online softmax、向量化 causal mask。
  适用阶段：score 计算与 topk 索引使用。
  经验文件：raw/experiences/attention/exp_sparse_attn_topk_001.yaml

知识单元 E2  [调优经验]  exp_sort_bitonic_001 — Bitonic Sort
  适用条件：N ≤ 1024，可完整装入 UB。
  性能提升：25%
  核心优化：padding 策略、batch 并行、masked memory access。
  适用阶段：小 k topk 替代迭代 argmax。
  经验文件：raw/experiences/misc/exp_sort_bitonic_001.yaml

知识单元 E3  [调优经验]  exp_moe_gather_001 — MoE Gather 优化
  适用条件：topk_num ≤ 8，BLOCK_D ≥ 128。
  性能提升：20%
  核心优化：vectorized metadata loading、memory alignment。
  适用阶段：PA_BSND block gather。
  经验文件：raw/experiences/gather_scatter/exp_moe_gather_001.yaml

知识单元 E4  [调优经验]  exp_stability_001 — 数值稳定性
  适用条件：所有含 exp/reduction 的算子。
  核心优化：fp32 accumulator、careful mask。
  来源：raw/experiences/general_stability/exp_stability_001.yaml

---

### 调优策略知识单元

知识单元 T1  [调优策略]  双 kernel 拆分（score + topk）
  适用场景：LightningIndexer
  核心方法：
    1. score kernel 写 fp32 workspace [B*S1, S2]
    2. topk kernel 读 workspace 取每行 top-K
  来源：template/lightningindexer.md L1.1

知识单元 T2  [调优策略]  persistent grid-stride loop
  适用场景：任务数不固定，需占满所有 core。
  核心方法：grid=(num_cores,)，kernel 内 for task_idx in range(pid, total_tasks, num_cores)。
  页面：techniques/persistent_kernels.md

知识单元 T3  [调优策略]  小 k loop unrolling / bitonic
  适用场景：topk k 较小。
  核心方法：维护 k 个 (value,index) 对；或用 bitonic sort 一次性排序完整行。
  页面：techniques/ilp.md

---

### 代码骨架知识单元

知识单元 S1  [代码骨架]  LightningIndexer score + topk 骨架
  适用任务：LightningIndexer
  骨架路径：template/lightningindexer.md
  关键结构：
    1. score kernel：persistent grid，BLOCK_S1×BLOCK_S2 tile，fp32 dot + ReLU + weighted sum + mask
    2. topk kernel：persistent grid，TILE_SORT 分块，迭代 argmax，index 升序输出
