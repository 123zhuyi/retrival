---
query_type: perf_tuning
query_keywords: [MTE2_high, gather_scatter, moe, routing, memory_access]
created: 2026-06-08
last_updated: 2026-06-08T09:55:00
hit_count: 1
context_summary: "MoeInitRouting 性能调优 — host 侧 argsort 为主要瓶颈，kernel 侧优化 BLOCK_H 和 grid-stride"
top_units: [T_mem_1, T_ilp_1, T_perf_1, T_perf_4]
stale: false
---

## 检索问题（Problem Statement）

MoeInitRouting 算子性能瓶颈分析。

**触发上下文**：Phase 4 性能优化，baseline speedup = 0.1288x
**检索范围**：MTE2_high / gather_scatter / memory_access

---

## 瓶颈识别

**瓶颈类型**: MTE2_high + host overhead
**测量值**: framework=0.0163ms, implementation=0.23ms, speedup=0.1288x
**原因分析**:
1. Host 侧 torch.argsort(int32) 在 Ascend 上 fallback 到 CPU（主要瓶颈）
2. BLOCK_H 固定为 128，对于小 H 导致无效 mask 操作
3. 大 NK 时 grid 可能超过核数，导致多轮调度

---

## 知识单元列表

知识单元 T_mem_1  [调优策略]  索引预计算与地址复用
  适用场景：gather/scatter 中重复计算索引偏移
  核心方法：预计算 src_base_addrs = indices * hidden_dim
  页面：pages/techniques/memory_access_optimization.md

知识单元 T_ilp_1  [调优策略]  for-loop tiling + ILP 提升
  适用场景：hidden_dim 较大，单 program 需处理多行数据
  核心方法：以 BLOCK_H 为单位循环处理 hidden_dim
  页面：pages/techniques/ilp.md

知识单元 T_perf_1  [调优策略]  动态 BLOCK_SIZE 选择
  适用场景：不同 shape 的 H 差异大
  核心方法：使用 triton.next_power_of_2(H) 动态选择 BLOCK_H
  页面：pages/techniques/performance_tuning.md

知识单元 T_perf_4  [调优策略]  care_padding=False 优化
  适用场景：当 padding 不影响计算结果时
  核心方法：tl.load(..., care_padding=False) 消除 MTE2→Vector 同步屏障
  页面：pages/techniques/performance_tuning.md

---

## 复用提示

- 含 gather + scatter + moe       → 本缓存（精确命中）
- 含 MTE2_high + memory_access    → 本缓存（精确命中）
- 含 routing + index_permutation  → 本缓存（相似命中）
