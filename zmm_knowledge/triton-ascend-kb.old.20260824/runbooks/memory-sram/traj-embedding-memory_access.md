---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: embedding
dimension: "1.1"
title: Embedding 与索引拷贝内存访问优化：Block Pointer、变长 cumsum 与右 Padding 偏移
description: 聚合 2 条 embedding memory access 经验，覆盖 embedding 前向大维度向量化加载与 top-k 索引拷贝的 block_ptr、cumsum 变长处理、右 padding 偏移与无效 block 早退，收益 25%。
tags: [embedding, memory_access, block_pointer, ragged_tensor, cumsum, padding_offset, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/embedding/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# Embedding 与索引拷贝内存访问优化：Block Pointer、变长 cumsum 与右 Padding 偏移

## 概述

Embedding/索引拷贝类算子的核心是不规则 gather：`out[token, dim] = weight[index[token], dim]`。与 elementwise 不同，其瓶颈在于：
1. **索引不规则**：`index` 是离散的，无法像连续张量那样做完美向量化；
2. **变长序列**：rmpad 后每个 batch 的有效 token 数不同，需要 `cumsum` 元数据定位；
3. **右 padding**：某些场景下有效 token 位于每行右侧，需要 `n_len - c_len` 偏移才能正确寻址。

本卡 2 条 experience 分别覆盖：
- `exp_embedding_001`：大 embedding 维度下的前向向量化加载与 2D tile；
- `exp_idx_copy_001`：基于 `tl.make_block_ptr` 的 top-k 索引拷贝，支持 ragged tensor 和右 padding 偏移。

**核心 trade-off**：
- Block pointer 能生成更规则的向量化 load/store，但要求 stride 和形状信息明确；
- 变长处理引入 `cumsum` 加载和早退逻辑，带来少量标量开销，但能显著减少 padding 空跑；
- 右 padding 偏移省掉 host 侧显式 slice，但要求 kernel 内正确计算 `c_len`。

## 共同模式

### 1. Block Pointer：把不规则 gather 包装成规则 block

**问题根因**：`embedding` 或 `index_copy` 的读取按 `index` 分散在 weight 表中，直接使用 `tl.load(weight_ptr + index * stride + dim)` 容易生成标量 gather，带宽利用率低。

**收益明显条件**：
- 每次 gather 的连续长度（embedding_dim 或 BLOCK_SIZE_N）较大；
- 索引虽然离散，但在一个 block 内连续存放；
- 输入/输出张量的 stride 已知且固定。

**实现要点**：
```python
p_ptrs = tl.make_block_ptr(
    base=p_ptr + pid_b * stride_pb + pid_h * stride_ph + (n_len - c_len) * stride_pn,
    shape=(c_len,),
    strides=(stride_pn,),
    offsets=(pid_n * BLOCK_SIZE_N,),
    block_shape=(BLOCK_SIZE_N,),
    order=(0,),
)
t_ptrs = tl.make_block_ptr(
    base=t_ptr + pid_h * stride_th + c_start * stride_tn,
    shape=(c_len,),
    strides=(stride_tn,),
    offsets=(pid_n * BLOCK_SIZE_N,),
    block_shape=(BLOCK_SIZE_N,),
    order=(0,),
)
idxs = tl.load(p_ptrs, boundary_check=(0,))
tl.store(t_ptrs, idxs, boundary_check=(0,))
```
- 用 `make_block_ptr` 把 source/target 包装成 1D block；
- `boundary_check=(0,)` 自动处理 block 越界；
- 索引张量本身连续存放，读取后不需要额外 gather。

**常见坑**：
- `base` 指针必须指向 block 起始位置，偏移量计算错误会导致错位；
- `shape` 是逻辑总长度（`c_len`），不是 block 数；
- `boundary_check` 仅在 block 尾部越界时生效，block 起始仍需用 `pid_n * BLOCK_SIZE_N < c_len` 早退。

### 2. 变长 cumsum 处理：cu_seqblocks / cu_topk_q_count

**问题根因**：rmpad 后，原始 batch 被拆成连续的有效 token 块。每个 block 需要知道：自己在全局有效序列中的起始位置、该 batch 的有效长度、top-k  expert 的 token 计数。

**收益明显条件**：
- 使用 varlen batch 或 MoE top-k；
- `cumsum` 已经由 host 侧预先计算；
- 需要避免按最大长度 padding 的浪费。

**实现要点**：
```python
q_block_start = tl.load(cu_seqblocks + pid_b)
q_block_end = tl.load(cu_seqblocks + pid_b + 1)
c_start = tl.load(cu_topk_q_count + pid_h * stride_ch + q_block_start * stride_cn)
c_end = tl.load(cu_topk_q_count + pid_h * stride_ch + q_block_end * stride_cn)
c_len = c_end - c_start
if c_len <= 0:
    return
```
- 用 `cu_seqblocks` 拿到当前 batch 在全局序列中的 `[start, end)`；
- 用 `cu_topk_q_count` 把序列位置映射到 top-k expert 的 token 计数；
- `c_len <= 0` 时早退。

**常见坑**：
- `cu_seqblocks` 和 `cu_topk_q_count` 的 stride 容易混淆（h vs n 维度）；
- `c_len` 计算后未检查导致空 block 仍做 load/store；
- host 侧 cumsum 必须包含最后一个累计值，否则 `q_block_end` 越界。

### 3. 右 Padding 偏移：`n_len - c_len`

**问题根因**：某些输入把有效数据放在每行右侧（如 rmpad 前原始序列右侧），而 block pointer 的 base 需要指向右侧第一个有效元素，偏移量为 `n_len - c_len`。

**收益明显条件**：
- 输入张量保留了原始 `n_len` 维度；
- 有效 token 集中在右侧；
- 不想在 host 侧做 slice 拷贝。

**实现要点**：
```python
p_ptrs = tl.make_block_ptr(
    base=p_ptr + pid_b * stride_pb + pid_h * stride_ph + (n_len - c_len) * stride_pn,
    ...
)
```
- 用 `n_len - c_len` 跳过左侧 padding；
- 保证 `c_len <= n_len`，否则说明元数据有误。

**常见坑**：
- 左右 padding 方向判断错误；
- `n_len` 和 `c_len` 数据类型不一致导致负数偏移；
- 未对 `n_len < c_len` 做防御性检查。

### 4. 无效 block 早退

**问题根因**：变长 batch 下，某些 program 对应的 block 可能完全落在 padding 区或当前 expert 无 token。如果不早退，会做无意义 load/store 并可能越界。

**收益明显条件**：
- batch 内长度差异大；
- top-k expert 的 token 数稀疏；
- grid 按最大 block 数启动。

**实现要点**：
```python
if c_len <= 0:
    return
if pid_n * BLOCK_SIZE_N >= c_len:
    return
```
- 第一层早退：`c_len <= 0`；
- 第二层早退：当前 block 起始已超出 `c_len`。

**常见坑**：
- 早退条件放太晚，已经加载了非法指针；
- 使用 `tl.where` 替代早退可能把标量分支扩散到所有 lane；
- 对 `pid_n * BLOCK_SIZE_N` 的乘法溢出未注意（应转为 int64）。

### 5. Embedding 前向大维度向量化

**问题根因**：当 `embed_dim` 较大（≥ 128）时，一次加载整行 embedding 会超出 UB；按 token 和 embed_dim 两个维度分块，可以在 Vector 单元上高效执行 gather。

**收益明显条件**：
- `embed_dim` ≥ 64/128；
- `num_tokens` 足够多，能填满 grid；
- weight 是连续存储的 2D 张量。

**实现要点**：
- 2D grid：dim0 处理 token block，dim1 处理 embedding block；
- 每个 program 加载 `BLOCK_M` 个 index，再对每个 index 加载 `BLOCK_N` 个连续 embedding 元素；
- 用 `num_stages` 做软件流水线预取。

**常见坑**：
- `BLOCK_N` 过大导致 UB spill；
- index 越界未 mask，导致读取非法 embedding 行；
- 反向 scatter-add 未使用 `tl.atomic_add`。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_idx_copy_001` | 25% | Top-k 索引拷贝：用 `tl.make_block_ptr` 实现变长索引张量的向量化 gather/copy，配合 `cu_seqblocks`/`cu_topk_q_count` 处理 ragged tensor 与右 padding 偏移。 |
| `exp_embedding_001` | 25% | Embedding 前向大维度加载：2D tile（token × embed_dim）+ 向量化加载 embedding 行，适合 embedding 维度大、token 数多的场景。 |

## 代表 Kernel

选择 `exp_idx_copy_001` 作为代表，因为它完整展示了 **block pointer、变长 cumsum、右 padding 偏移、无效 block 早退** 四个内存访问核心模式。

```python
@triton.jit
def save_topk_idx_kernel(
    p_ptr,
    t_ptr,
    cu_seqblocks,
    cu_topk_q_count,
    n_len,
    stride_pb,
    stride_ph,
    stride_pn,
    stride_th,
    stride_tn,
    stride_ch,
    stride_cn,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)

    # 通过 cumsum 获取当前 batch 在全局序列中的范围
    q_block_start = tl.load(cu_seqblocks + pid_b)
    q_block_end = tl.load(cu_seqblocks + pid_b + 1)

    # 映射到当前 head/expert 的 top-k token 计数
    c_start = tl.load(cu_topk_q_count + pid_h * stride_ch + q_block_start * stride_cn)
    c_end = tl.load(cu_topk_q_count + pid_h * stride_ch + q_block_end * stride_cn)
    c_len = c_end - c_start

    # 无效 block 早退
    if c_len <= 0:
        return
    if pid_n * BLOCK_SIZE_N >= c_len:
        return

    # 源：右 padding 偏移后的 block pointer
    p_ptrs = tl.make_block_ptr(
        base=p_ptr
        + pid_b * stride_pb
        + pid_h * stride_ph
        + (n_len - c_len) * stride_pn,
        shape=(c_len,),
        strides=(stride_pn,),
        offsets=(pid_n * BLOCK_SIZE_N,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )
    # 目标：连续写入 top-k 索引目标区
    t_ptrs = tl.make_block_ptr(
        base=t_ptr + pid_h * stride_th + c_start * stride_tn,
        shape=(c_len,),
        strides=(stride_tn,),
        offsets=(pid_n * BLOCK_SIZE_N,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )

    idxs = tl.load(p_ptrs, boundary_check=(0,))
    tl.store(t_ptrs, idxs, boundary_check=(0,))
```

**关键实现细节**：
- 3D grid `(batch, head, block_n)` 与变长元数据配合；
- `c_len` 由两层 cumsum 差值得到；
- 源指针加上 `(n_len - c_len) * stride_pn` 跳过左侧 padding；
- 目标指针用 `c_start * stride_tn` 定位到全局连续 top-k 区；
- `boundary_check=(0,)` 自动处理最后一个 block 的尾部越界。

## 已知实例

- `exp_idx_copy_001` — Memory Access（收益 25%）
  - Top-k 索引拷贝：block pointer 向量化 gather/copy，支持 `cu_seqblocks`/`cu_topk_q_count` 变长处理与右 padding 偏移。对应 kernel: `raw_experiences/extracted_kernels/024-save_topk_idx_kernel.py`。
- `exp_embedding_001` — Memory Access（收益 25%）
  - Embedding 前向大维度加载：2D tile + 向量化加载 embedding 行，适合大 embedding 维度场景。对应 kernel: `raw_experiences/extracted_kernels/090-embedding_forward_kernel.py`。

## 关联知识

- 开发指南: [Embedding / 索引拷贝算子优化](../../reference/op-class-embedding.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [Embedding / 索引拷贝算子优化](../../reference/op-class-embedding.md) — 主题相关：共享 910b2, 910b3, embedding
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](../../ops/block-sparse-attention-grad.md) — 同属优化知识，主题相关
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](../../ops/block-sparse-attention.md) — 同属优化知识，主题相关
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 同属优化知识，主题相关
- 相关主题: [RoPE paired-channel pointer addressing](../../ops/rope-pointer-addressing.md) — 同属优化知识，主题相关
- 相关主题: [FlashAttention 变体的内存访问优化](../algorithm-numerics/traj-attention-memory_access.md) — 主题相关：共享 910b2, 910b3, block_pointer, memory_access
- 相关主题: [Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion](../algorithm-numerics/traj-normalization-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [care_padding=False 消除 MTE2→Vector 同步屏障](../compute-pipeline/care-padding-false-ilp.md) — 同属优化知识，主题相关
- 相关主题: [1D grid 硬上限 65535，超出用 2D grid 或增大 BLOCK](../compute-pipeline/grid-limit-65535.md) — 同属优化知识，主题相关
- 相关主题: [多核负载均衡分配公式（每 core 处理 block 数差不超过 1）](../compute-pipeline/load-balance-formula.md) — 同属优化知识，主题相关
- 相关主题: [swizzle2d 块重排提升 L2 复用](../compute-pipeline/swizzle2d-block-reorder.md) — 同属优化知识，主题相关
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](../compute-pipeline/traj-conv-memory_access.md) — 主题相关：共享 910b2, 910b3, block_pointer, memory_access
- 相关主题: [Embedding Backward Scheduling 优化聚合](../compute-pipeline/traj-embedding-scheduling.md) — 主题相关：共享 910b2, 910b3, embedding
- 相关主题: [Triton Block Size 通用调优原则](../compute-pipeline/traj-general_perf-block_size.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [Matmul Split 切分优化：512B 行宽对齐与转置组合分块策略](../compute-pipeline/traj-matmul-split.md) — 主题相关：共享 910b2, 910b3, block_pointer
- 相关主题: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](block-ptr-vs-scalar-load.md) — 同属优化知识，主题相关
- 相关主题: [M>=256 时 BLOCK_M=256（总 block-iter 减半）](blockm-256-large-m.md) — 同属优化知识，主题相关
- 相关主题: [输入进入 kernel 前必须 contiguous](input-contiguous.md) — 主题相关：共享 memory_access
- 相关主题: [BLOCK 向上取 2 的幂并声明为 constexpr](pow2-block-constexpr.md) — 同属优化知识，主题相关
- 相关主题: [Activation Backward 内存访问优化聚合](traj-activation-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Gather/Scatter 内存访问优化聚合](traj-gather_scatter-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Ascend 大 Shape 连续分块内存访问优化](traj-general_ascend-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](traj-general_perf-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Matmul 内存访问优化：分块策略、L0 复用与算子融合](traj-matmul-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Misc 算子内存访问优化聚合](traj-misc-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Sum-and-Scale 归约内存访问优化](traj-reduction-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [RoPE 内存访问优化：cos/sin 复用、ALIGNED 向量化加载与动态特化](traj-rope-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Softmax Backward 内存访问优化聚合](traj-softmax-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Solve_Tril 内存访问优化：Preload + extract_slice 消除循环内 HBM 重读](traj-solve_tril-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [SSM 内存访问优化：转置 load 合并化、寄存器复用与分块状态矩阵](traj-ssm-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [UB 占用决定 BLOCK_ROWS/tile 大小，必须按算子特征分别自适应](ub-aware-block-adaptive.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
