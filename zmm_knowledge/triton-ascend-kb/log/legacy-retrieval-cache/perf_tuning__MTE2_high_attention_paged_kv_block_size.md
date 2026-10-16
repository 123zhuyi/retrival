---
query_type: perf_tuning
query_keywords: [MTE2_high, attention, paged_kv, block_size]
created: 2026-06-10
last_updated: 2026-06-10T14:45:00
hit_count: 1
context_summary: "decode_mla BLOCK_N=16 循环次数多，paged KV 非连续访问导致 MTE2 瓶颈"
top_units: [P2_dot_optimize_001, P2_block_h_group_001, P2_block_n_increase_001]
stale: false
---

## 检索问题

decode_mla BLOCK_N=16 循环次数多（256 次 per seq），paged KV 非连续访问导致 MTE2 瓶颈。

**触发上下文**: 瓶颈分析 — MTE2_high 预判 + VecUtil_low 预判
**检索范围**: MTE2_high

---

## 知识单元列表

知识单元 P2_dot_optimize_001 [调优经验] 使用 tl.dot 替代逐元素 sum
  适用条件: attention 算子中 Q@K^T 和 P@V 计算
  核心优化: 用 tl.dot 让 CUBE 单元加速矩阵乘，替代逐元素 sum 的 VEC 路径
  效果: CUBE 利用率提升，整体延迟降低 20-40%
  命中优化点: #3 (使用 tl.dot 替代低效循环)

知识单元 P2_block_h_group_001 [调优经验] BLOCK_H 多头分组共享 KV
  适用条件: GQA (H_Q > H_KV)，decode phase
  核心优化: 一个 program 处理多个 Q head，共享同一 KV 加载
  效果: 减少 KV 重复读取次数 H_Q/H_KV 倍
  命中优化点: #7 (多头分组)

知识单元 P2_block_n_increase_001 [调优经验] 在 UB 限制内增大 BLOCK_N
  适用条件: decode attention，BLOCK_N 过小导致循环次数过多
  核心优化: 优化数据加载模式，在 UB 192KB 限制内尽可能增大 BLOCK_N
  效果: 减少循环次数，降低 MTE2 调度开销
  命中优化点: #5 (增大 block size)
