---
query_type: perf_tuning
query_keywords: [MTE2_high, groupnorm, normalization, memory_access, fp16]
created: 2026-06-08
last_updated: 2026-06-08T14:30:00
hit_count: 1
context_summary: "GroupNormSwish 性能瓶颈 — 三趟算法导致 MTE2 过高，speedup=0.22"
top_units: [P2_MTE2_1, P2_MTE2_2, P2_MTE2_3]
stale: false
---

## 检索问题

GroupNormSwish Triton 实现性能远低于 PyTorch NPU 原生算子（speedup=0.22）。

**触发上下文**: 性能基准测试，三趟算法导致内存访问瓶颈
**检索范围**: groupnorm + normalization + MTE2优化

---

## 知识单元列表

**P2_MTE2_1** `[调优经验]` 三趟→单趟 Welford 算法减少内存访问
- **内容**: 当前实现 Pass 1(mean) + Pass 2(var) + Pass 3(normalize+swish) 导致每个元素被读取 3 次。使用 Welford 在线算法可在单趟内同时计算 mean 和 variance，将内存访问减少 33%。
- **适用条件**: GroupNorm/RMSNorm/LayerNorm 等归约算子
- **性能提升**: 30-40%
- **来源**: raw/experiences/normalization/exp_online_mean_var_001.yaml

**P2_MTE2_2** `[调优经验]` 增大 BLOCK_SIZE 提升向量利用率
- **内容**: 当前 BLOCK_D=256，fp16 下仅 512 bytes。增大至 512/1024 可更好利用 VEC 单元 256B 处理能力，提升 MTE2→Vector 数据通路效率。
- **适用条件**: elementwise/归约算子，D >= BLOCK_SIZE
- **性能提升**: 15-25%
- **来源**: pages/techniques/tiling.md

**P2_MTE2_3** `[调优经验]` 融合 mean/var 计算消除中间同步
- **内容**: 将 mean 和 variance 的加载循环合并为一个循环，在加载数据的同时计算 mean_acc 和 sq_acc（sum of squares），然后 var = E[x^2] - mean^2。虽然数值稳定性略低于 Welford，但实现简单且内存访问最少。
- **适用条件**: 数据分布不太极端（避免 cancellation）
- **性能提升**: 20-30%
- **来源**: pages/techniques/operator_fusion.md

---

## 复用提示

**可部分复用本缓存的 query_keywords 变体**:
- 含 `MTE2_high` + `normalization` + `memory_access` → 精确命中
- 含 `groupnorm` + `performance` → 相似命中，复用 P2_MTE2_1/P2_MTE2_3
