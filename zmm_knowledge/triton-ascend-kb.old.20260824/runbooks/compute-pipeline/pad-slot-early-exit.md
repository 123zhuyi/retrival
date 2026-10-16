---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.1"
title: pad slot 单次检查提前退出，跳过无效填充序列
description: 对变长 batch 中的 padding 序列，在 kernel 入口通过间接索引 + 单次 load+cmp 判断 pad_slot_id，匹配则立即 return，避免执行完整计算循环。
tags: [pad_slot, early_exit, indirect_indexing, dynamic_batch, moe, kv_cache]
confidence: verified
status: verified
created_at: "2026-08-10T00:00:00Z"
updated_at: "2026-08-10T00:00:00Z"
---

## OPT-18 pad slot 单次检查提前退出，跳过无效填充序列

【优化维度】：计算｜Scalar 削减
【泛化层级】：条件性（迁移条件：变长 batch 中有 padding 序列，且可通过单次 load+cmp 判断；失效边界：padding 比例极低，或判断条件复杂）

<原则：变长 batch 中常有 padding 序列不需要任何计算。与其执行完整 kernel 循环后由 mask 过滤输出，不如在 kernel 入口处通过间接索引（batch_id → cache_slot）做一次 load + 整数比较，若命中 pad_slot_id 则立即 `return`。基础开销仅 ~10 cycles，远小于完整循环。>

**正例**：

```python
# indirect mapping: batch_id → cache_slot
cache_idx = tl.load(cache_indices_ptr + pid_batch)
if cache_idx == pad_slot_id:   # -1 = 无效填充槽
    return                      # 立即退出，释放 compute unit
```

**适用场景**：

| 场景 | 说明 |
|---|---|
| KV-cache page table | 无效 page 跳过 |
| Embedding lookup with dynamic indices | 无效索引跳过 |
| MoE routing | 未命中 expert 的 token 跳过 |
| 变长序列 batch | padding 序列比例 ≥10% 时收益显著 |

**与 constexpr 分支消除的区别**：

- `tl.constexpr` 分支：消除编译时已知路径（算子配置级）。
- pad slot 退出：运行时数据驱动跳过（batch 构成级）。

两者互补：constexpr 消除固定路径分支，pad slot 跳过动态无效数据。

**坏实践（反例）**：
```python
for i in range(seq_len):
    x = tl.load(ptr + i * stride, mask=mask)
    # 对 padding 序列也执行完整计算
    ...
```
后果：padding 序列浪费大量 compute cycle。✅ 正解即入口处单次检查提前退出。

## 已知实例（P3 挂载）

### Experience YAML 反链
- `exp_causal_conv1d_prefill_005` — Scheduling Optimization （收益 在有 padding 序列的 batch 中节省无效计算比例（如 25% padding→~25% 资源节省）；最小开销仅 1 load + 1 cmp）
  - pad_slot 机制不仅节省无效计算，更重要的是让 SM/SIMT 单元可被其他有用工作抢占——在 Ascend 的 grid-stride 调度下，提前退出的 program 对应的 tile 可被重新分配给其他 tasks。

<!-- okf:related:start -->

# 相关

- 实践案例: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](../../ops/fused-infer-attention-score.md) — 主题相关：共享 kv_cache
- 实践案例: [IncreFlashAttention inference-decode on Ascend NPU](../../ops/incre-flash-attention.md) — 主题相关：共享 kv_cache
- 实践案例: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](../../ops/kv-quant-sparse-flash-attention.md) — 主题相关：共享 kv_cache
- 实践案例: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../../ops/mla-preprocess.md) — 主题相关：共享 kv_cache
- 实践案例: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](../../ops/mla-prolog.md) — 主题相关：共享 kv_cache
- 实践案例: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](../../ops/nsa-compress-with-cache.md) — 主题相关：共享 kv_cache
- 实践案例: [PromptFlashAttention inference/prefill on Ascend NPU](../../ops/prompt-flash-attention.md) — 主题相关：共享 kv_cache
- 实践案例: [SparseFlashAttention inference (sparse KV) on Ascend NPU](../../ops/sparse-flash-attention.md) — 主题相关：共享 kv_cache
- 相关主题: [非 2 的幂的语义维度必须 pad 到 next_pow2 + 极值填充 + 截断](../memory-sram/pad-next-pow2-extremum.md) — 同属优化知识，主题相关
- 相关主题: [Causal Conv1D 调度优化：Prefill/Update 布局分野、2D Grid 与 pad_slot 早退](traj-conv-scheduling.md) — 主题相关：共享 pad_slot
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](traj-rope-scheduling.md) — 主题相关：共享 kv_cache

<!-- okf:related:end -->
