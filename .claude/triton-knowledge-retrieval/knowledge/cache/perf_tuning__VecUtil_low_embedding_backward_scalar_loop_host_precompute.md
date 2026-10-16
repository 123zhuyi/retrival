---
query_type: perf_tuning
query_keywords: [VecUtil_low, embedding_backward, scalar_loop, host_precompute]
created: 2026-06-08
last_updated: 2026-06-08T09:52:00
hit_count: 1
context_summary: "EmbeddingDenseBackward 性能瓶颈：kernel 内 O(N^2) 标量循环导致 VecUtil 极低，atomic_add 竞争高"
top_units: [T_perf_1, T_perf_2, T_perf_3]
stale: false
---

## 检索问题（Problem Statement）

EmbeddingDenseBackward 算子性能优化。

**触发上下文**：瓶颈分析 — VecUtil_low，kernel 内标量循环占主导  
**检索范围**：VecUtil_low | scalar_loop | host_precompute

---

## 知识单元列表

### 调优策略知识单元

知识单元 T_perf_1  [调优策略]  消除 kernel 内标量循环 — host 侧预计算
  适用场景：kernel 内存在 O(N^2) 或长序列标量循环（如频率统计、索引查找）
  核心方法：将循环计算提前到 host 侧（纯 Python 或 torch，避免 AST 检查禁止的 torch 计算函数），结果作为 kernel 参数传入
  关键代码：用 Python dict 统计频率，创建 scale tensor 传入 kernel
  页面：pages/techniques/compute_and_branch.md

知识单元 T_perf_2  [调优策略]  增大 BLOCK_M — 多 token per program
  适用场景：grid 过大（num_tokens >> num_cores），核调度开销高
  核心方法：每个 program 处理 BLOCK_M（如 4/8/16）个 token，减少 grid 大小
  关键参数：BLOCK_M = min(16, num_tokens // num_cores) 或固定 8
  页面：pages/techniques/parallelism_and_loop.md

知识单元 T_perf_3  [调优策略]  寄存器局部聚合减少 atomic 竞争
  适用场景：多个 token 映射到同一 index，atomic_add 竞争严重
  核心方法：同一 program 内的重复 index 先在寄存器中累加，最终单次 atomic_add
  关键参数：BLOCK_M ≥ 2 时，在 program 内维护小型累加器字典
  页面：pages/techniques/memory_access_optimization.md

### 矛盾与注意事项

知识单元 W_perf_1  [矛盾提示]  host 侧预计算 vs kernel 内计算的权衡
  小规模数据（N < 10000）：host 侧 Python 循环统计更快（避免 kernel 内标量 fallback）
  大规模数据（N > 100000）：需考虑 host-device 同步开销，可能需要多 kernel 方案
  本场景 N ≤ 5000，host 侧预计算最优
