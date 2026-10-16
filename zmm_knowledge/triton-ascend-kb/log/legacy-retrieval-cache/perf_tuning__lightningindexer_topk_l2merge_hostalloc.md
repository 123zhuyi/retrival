---
query_type: perf_tuning
query_keywords: [lightningindexer, topk, two-level-merge, l2-parallel, host-alloc, radix-hist, ascend910b1]
created: 2026-07-29T15:40:00+08:00
last_updated: 2026-07-29T15:40:00+08:00
hit_count: 2
context_summary: "LightningIndexer Phase4 opt6 — case9-12 (R=1,S2=8192) topk L2 单核串行合并洼地 + radix hist host 分配空转"
top_units: [C1, C2, C5, P4, OP1, B1, B2, B6, B7, E1, E2, E5, E6, T2, T3, T4]
stale: false
---

## 检索问题（Problem Statement）

LightningIndexer 两级 topk：R=B*S1=1 时 L1 grid=nch 小、L2 合并 grid=1 单核串行
sort 合并 nch 个 run（case9-12, S2=8192, K∈{64,100,512,1024}, 0.20-0.23x）；
radix 路径 hist 窗口内 torch host 分配（full×2/clone×2/zeros）造成 device 空转。
求：L2 合并并行化方案 + host 分配消除方案。

## 知识单元列表

### 硬件/性能约束

知识单元 C1  [硬件约束]  UB 总用量 ≤ 192KB
  topk L2 合并 tile：nch*K 候选 (值 fp32 + 索引 int32) 必须 ≤ 192KB。
  nch=4,K=1024 → 4096×8B=32KB 可行；nch=16,K=2048 → 256KB 超界，需分波次合并。
  来源: hardware/constraints_cheatsheet.md C1

知识单元 C2  [硬件约束]  Grid 维度上界 65535，grid_size ≤ AI Core 数
  L2 并行化后 grid=(R, nwave) 或 persistent (num_cores,)，不得超过核数（910b1=48? 以 npu-smi 为准）。
  来源: hardware/constraints_cheatsheet.md C2/C5

知识单元 C5  [性能约束]  int64/int32 CMP → scalar fallback
  L2 合并中的索引比较、hist bin 比较若用 int32/int64 会标量化。
  修复：比较前 .to(tl.float32)，索引用 int32。
  来源: hardware/constraints_cheatsheet.md P1/P2

知识单元 P4  [性能约束]  Block 数远少于核数 → 核利用率低
  R=1 时 L2 grid=1 → 1/48 核。修复：persistent kernel 或按合并波次/候选分块拆多 program。
  来源: hardware/constraints_cheatsheet.md P4

### 算子分类

知识单元 OP1  [算子分类]  operator_class = "fused_score_topk"（topk 子阶段 = Indexing/Sort）
  来源: cache/codegen_init__lightningindexer_topk_score_bf16.md OP1

### 瓶颈知识单元

知识单元 B1  [潜在瓶颈]  S1=1/R=1 时 topk 并行度不足
  修复：persistent grid-stride 或按候选分块并行；增大单 program 工作量。
  来源: cache/codegen_init__lightningindexer_topk_score_bf16.md B1

知识单元 B2  [潜在瓶颈]  topk 小 k 固定开销大（k≤256）
  修复：UB 允许时一次性 bitonic/sorting-network 处理完整候选集，替代迭代 argmax。
  关联经验: exp_sort_bitonic_001
  来源: cache/codegen_init__lightningindexer_topk_score_bf16.md B2

知识单元 B6  [潜在瓶颈]  两级 topk L2 串行合并（本轮诊断）
  触发条件：R=1 且 nch≥2，L2 单核串行合并 nch 个有序 run。
  修复方向 a：L2 候选集 (nch×K) 装入单 program UB 一次 bitonic 全排序（nch*K≤4096 时可行）。
  修复方向 b：多 program 各合并候选子集 → 部分 top-K 写 workspace → 最终单核小合并（波次归约）。
  修复方向 c：跳过 L2 —— L1 直接 atomic 提交全局候选（需等值 tie-break 一致性，风险高）。
  来源: 本会话瓶颈诊断（static_fallback）

知识单元 B7  [潜在瓶颈]  kernel 间 torch host 分配调度空转
  触发条件：forward 中 kernel 启动之间穿插 torch.full/clone/zeros。
  修复：预分配 workspace 一次创建复用（cache 在 self/_workspace dict），kernel 间只做指针传递。
  注意：benchmark 计时含 host 调度，分配消除直接降延迟；多 shape 时需按 shape 键缓存。
  来源: 本会话瓶颈诊断（static_fallback）+ techniques/persistent_kernels.md

### 调优经验

知识单元 E1  [调优经验]  exp_sparse_attn_topk_001 — 稀疏注意力 TopK（40%）
  仅迭代 topk 块、block pointer、向量化 mask。
  来源: raw/experiences/attention/exp_sparse_attn_topk_001.yaml

知识单元 E2  [调优经验]  exp_sort_bitonic_001 — Bitonic Sort（25%）
  N≤1024 可完整装 UB 时一次性排序；padding 用 iinfo 极值自然沉底；batch 并行。
  适用：L2 候选集 nch*K≤1024 时直接全排序；>1024 需分波次。
  来源: raw/experiences/misc/exp_sort_bitonic_001.yaml

知识单元 E5  [调优经验]  exp_hist_count_001 — Histogram 寄存器累加
  局部寄存器累加 partial histogram 再写全局，减少 atomic 竞争；
  2D tiling（bin 块 × 输入块）+ 向量化比较 x[:,None]==bins[None,:]。
  适用：radix hist pass（case4/7）若后续攻 hist 原子竞争。
  来源: raw/experiences/misc/exp_hist_count_001.yaml

知识单元 E6  [调优经验]  exp_topk_pad_001 — topk 索引 block_ptr + early_exit（25%）
  tl.make_block_ptr + boundary_check 处理 ragged；越界 program 立即 return。
  来源: raw/experiences/misc/exp_topk_pad_001.yaml

### 调优策略

知识单元 T2  [调优策略]  persistent grid-stride loop
  grid=(num_cores,)，kernel 内 for task in range(pid,total,nprog)。
  来源: techniques/persistent_kernels.md

知识单元 T3  [调优策略]  小 k bitonic 一次性排序
  候选集装 UB 后 sorting-network 全排序替代迭代 argmax，消除 k 次循环固定开销。
  来源: techniques/ilp.md + exp_sort_bitonic_001

知识单元 T4  [调优策略]  波次归约（wave reduction）并行合并
  nch 个 run → ceil(nch/w) 个 program 各合并 w 个 run → 部分结果 → 最终单核合并。
  每波候选数控制在 UB 内；波次数 = log_w(nch)。语义不变式：全局 top-K ⊆ 各 run top-K 之并。
  来源: 本会话设计（归约树标准模式）
