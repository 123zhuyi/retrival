---
title: Embedding / 索引拷贝算子优化
category: ops
tags: [embedding, gather, index-copy, lookup, backward]
hardware: [910b2, 910b3]
last_updated: 2026-04-27
---

# Embedding / 索引拷贝算子优化

## 概述

Embedding 查找是 NLP 模型的基础操作：`y[i] = weight[index[i]]`，本质是不规则 gather。  
反向传播为 scatter-add：`grad_weight[index[i]] += grad_y[i]`。

Triton-Ascend 中的优化重点：
1. **2D Tile 分块**：沿 (token, embedding_dim) 两个维度 tile，最大化向量化
2. **软件流水线**：多 stage 预取掩盖访存延迟
3. **反向原子加**：scatter-add 的并发原子更新

## 经验路由

| 场景 | 经验ID | 关键技术 | 经验文件 |
|------|--------|----------|----------|
| Embedding 前向查找 | exp_embedding_001 | 2D tile (M×N)；软件流水线 num_stages；向量化加载 embedding 行 | [yaml](../../raw/experiences/embedding/exp_embedding_001.yaml) |
| Embedding 反向（grad） | exp_embedding_backward_001 | scatter-add 原子操作；按 index 排序提升访问连续性 | [yaml](../../raw/experiences/embedding/exp_embedding_backward_001.yaml) |
| 索引拷贝（idx_copy） | exp_idx_copy_001 | gather + copy，支持非连续索引批量拷贝 | [yaml](../../raw/experiences/embedding/exp_idx_copy_001.yaml) |

## 关键技术

### 2D Tiling + 软件流水线

```python
@triton.jit
def embedding_fwd(weight_ptr, index_ptr, out_ptr, vocab_size, embed_dim,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, num_stages: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # 每个 program 处理 BLOCK_M 个 token 的 BLOCK_N 维 embedding
    token_ids = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    embed_ids = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # 加载 index
    indices = tl.load(index_ptr + token_ids, mask=token_ids < n_tokens)
    # 向量化加载 embedding 行
    for m in tl.range(BLOCK_M, num_stages=num_stages):
        row = tl.load(weight_ptr + indices[m] * embed_dim + embed_ids,
                      mask=embed_ids < embed_dim)
        tl.store(out_ptr + (pid_m * BLOCK_M + m) * embed_dim + embed_ids, row)
```

### 反向 scatter-add 与原子操作

Embedding 反向存在写冲突（同一 index 可能被多个 token 引用），需原子加：

```python
tl.atomic_add(grad_weight_ptr + index * embed_dim + embed_ids, grad_y_tile, mask=mask)
```

优化：提前对 indices 排序，使相同 index 连续出现，提升 L2 缓存命中率。  
参考：[exp_embedding_backward_001](../../raw/experiences/embedding/exp_embedding_backward_001.yaml)

## 适用条件

| 参数 | 推荐值 |
|------|--------|
| embed_dim | ≥64，BLOCK_N=64/128/256 |
| n_tokens | ≥32（过少时 launch overhead 占主导） |
| vocab_size | 任意，index 超出 vocab_size 需 mask |

## 常见陷阱

| 陷阱 | 说明 | 解决方案 |
|------|------|----------|
| 反向原子加冲突 | 多个 token 同 index，写冲突导致结果不一致 | 使用 tl.atomic_add |
| 非连续索引低效 | 随机 index 访问 weight，cache 命中率低 | 按 index 排序后处理，提升局部性 |
| BLOCK_N 过大 | 超出 UB 容量 | 910b2 fp16 单行最大 98304 元素，按需分块 |

## 相关页面

- [ops/gather_scatter.md](gather_scatter.md) — Embedding 反向是特殊的 scatter-add
- [techniques/tiling.md](../techniques/tiling.md) — 2D tile 大小选择
