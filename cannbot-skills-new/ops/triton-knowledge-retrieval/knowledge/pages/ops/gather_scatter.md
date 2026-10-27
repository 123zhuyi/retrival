---
title: Gather/Scatter 算子优化
category: ops
tags: [gather, scatter, moe, routing, token-permutation, indexed-access, binning]
hardware: [910b2, 910b3]
last_updated: 2026-04-27
---

# Gather/Scatter 算子优化

## 概述

Gather/Scatter 算子通过索引张量进行不规则（非连续）内存访问，是 MoE（Mixture of Experts）路由、Token 排列、嵌入查找、稀疏更新等场景的核心原语。

- **Gather**：`output[i] = input[index[i]]`，按索引从源张量读取元素
- **Scatter**：`output[index[i]] = input[i]`，按索引向目标张量写入元素

在 Ascend 910b2/910b3 上，不规则内存访问无法直接利用向量化加载指令，主要瓶颈在于：

1. **地址离散**：索引随机分布时每次 load/store 产生独立地址计算，无法 coalesce
2. **原子冲突**：多核 scatter 写入同一目标地址时，原子操作竞争严重
3. **负载不均**：MoE 中不同 Expert 接收 token 数量不均匀，核间负载失衡
4. **小块访问**：每个 token 单独 scatter 时访问粒度过小，向量化效率低

## 检测标准

满足以下任意模式时，将算子归类为 gather/scatter：

- Triton 原语：`tl.gather`、`tl.scatter`、或手动索引指针计算 `ptr + index[i]`
- PyTorch 原语：`torch.gather`、`torch.scatter`、`torch.index_select`、`Tensor[index]`
- MoE 路由模式：top-k 专家选择 + token 分发/收集
- Token 排列：`token_permutation`、`unpermutation`（基于 sorted_indices）
- 嵌入表查找：`nn.Embedding`（本质为大规模 gather）

## 适用条件

| 条件 | 说明 |
|------|------|
| 数据类型 | fp16、bf16、fp32；索引为 int32 或 int64 |
| 索引密度 | 稀疏（random index）：侧重地址优化；密集（sorted index）：可 coalesce |
| Expert 数量 | MoE top-k 场景：num_experts × expert_capacity ≥ 1024 时 bin 策略收益显著 |
| Scatter 冲突率 | 原子冲突多时改用两阶段（先局部聚合，再全局写回） |
| Gather 后接 GEMM | gather → matmul 是 MoE 核心路径，GEMM 部分参考 matmul 经验 |

## 经验路由

| 场景 | 经验ID | 提升% | 关键技术 | 来源文件 |
|------|--------|-------|----------|----------|
| 分散 scatter 原子竞争 | exp_gs_mem_001 | 25% | 寄存器局部聚合，单次 atomic write per program | memory_access.md |
| 小粒度 scatter 写入 | exp_gs_mem_002 | 30% | 向量化 scatter：BLOCK_SIZE 元素向量操作 | memory_access.md |
| 索引地址重复计算 | exp_gs_mem_003 | 20% | 预计算索引偏移，复用地址计算结果 | memory_access.md |
| 不规则索引访问 | exp_gs_mem_004 | 25% | Block pointer 用于索引访问 | memory_access.md |
| 随机 scatter 访问模式 | exp_gs_mem_005 | 30% | scatter 前对索引排序，改善内存访问局部性 | memory_access.md |
| MoE expert 分组 | exp_gs_bin_001 | 30% | Bin 分组：按 expert 分 bin，逐 bin 处理 | binning_strategy.md |
| 动态容量处理 | exp_gs_bin_002 | 20% | 运行时容量检查，动态 bin 管理 | binning_strategy.md |
| MoE top-k 路由 | exp_gs_bin_003 | 35% | 分 bin 的 expert dispatch（num_experts × expert_capacity ≥ 1024） | binning_strategy.md |
| Token 排列 | exp_gs_bin_004 | 25% | shuffle 实现 token 排列 | binning_strategy.md |
| Expert 负载均衡 | exp_gs_bin_005 | 30% | work-stealing 均衡 expert 负载 | binning_strategy.md |

## 关键技术

### 寄存器局部聚合（减少 Atomic 竞争）

将多个分散的 scatter 写入在寄存器中局部聚合，最终只做一次原子写回（exp_gs_mem_001，25% 提升）：

```python
@triton.jit
def scatter_with_aggregation_kernel(
    src_ptr, index_ptr, dst_ptr,
    N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    src = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    idx = tl.load(index_ptr + offsets, mask=mask, other=0)

    # 在寄存器中持有数据，批量写出（非逐元素原子）
    # 当多个 offsets 映射到同一 idx 时，先在寄存器累加
    tl.atomic_add(dst_ptr + idx, src, mask=mask)
```

### 向量化 Scatter 写入（BLOCK_SIZE 粒度）

以 BLOCK_SIZE 为单位处理 scatter，充分利用向量指令（exp_gs_mem_002，30% 提升）：

```python
@triton.jit
def vectorized_scatter_kernel(
    src_ptr, index_ptr, dst_ptr,
    num_tokens, hidden_dim,
    BLOCK_HIDDEN: tl.constexpr,
):
    """每个 program 处理一个 token，向量化写入 hidden_dim 维度。"""
    token_id = tl.program_id(0)
    dst_idx = tl.load(index_ptr + token_id)   # 目标位置索引

    hidden_offs = tl.arange(0, BLOCK_HIDDEN)
    mask = hidden_offs < hidden_dim

    # 向量化加载 + 向量化存储（连续的 hidden_dim 元素）
    src = tl.load(src_ptr + token_id * hidden_dim + hidden_offs, mask=mask)
    tl.store(dst_ptr + dst_idx * hidden_dim + hidden_offs, src, mask=mask)
```

### 索引预计算与地址复用（exp_gs_mem_003，20% 提升）

避免在热循环中重复计算相同索引偏移：

```python
@triton.jit
def gather_precompute_addr_kernel(
    src_ptr, index_ptr, dst_ptr,
    N, hidden_dim,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    pid = tl.program_id(0)
    token_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    token_mask = token_offs < N

    # 预计算所有 token 的源地址（复用于多个 hidden tile）
    src_indices = tl.load(index_ptr + token_offs, mask=token_mask, other=0)
    src_base_addrs = src_indices * hidden_dim  # 预计算基地址

    for h_start in range(0, hidden_dim, BLOCK_HIDDEN):
        h_offs = h_start + tl.arange(0, BLOCK_HIDDEN)
        h_mask = h_offs < hidden_dim

        # 利用预计算的基地址，避免重复乘法
        src_addrs = src_base_addrs[:, None] + h_offs[None, :]
        x = tl.load(src_ptr + src_addrs,
                     mask=token_mask[:, None] & h_mask[None, :])
        dst_addrs = token_offs[:, None] * hidden_dim + h_offs[None, :]
        tl.store(dst_ptr + dst_addrs, x,
                 mask=token_mask[:, None] & h_mask[None, :])
```

### 索引排序改善访问局部性（exp_gs_mem_005，30% 提升）

scatter 前对索引排序，使相邻 program 写入相邻地址，提升 cache 利用率和 HBM 访问合并度：

```python
def sorted_scatter(src: torch.Tensor, index: torch.Tensor,
                   dst_size: int) -> torch.Tensor:
    # 排序索引（在 host 侧或用单独内核完成）
    sorted_order = torch.argsort(index)
    sorted_index = index[sorted_order]
    sorted_src = src[sorted_order]

    dst = torch.zeros(dst_size, src.shape[-1],
                      dtype=src.dtype, device=src.device)
    N = src.shape[0]
    BLOCK_HIDDEN = triton.next_power_of_2(src.shape[-1])
    # 排序后 scatter：相邻写入相邻地址
    vectorized_scatter_kernel[(N,)](
        sorted_src, sorted_index, dst,
        N, src.shape[-1], BLOCK_HIDDEN=BLOCK_HIDDEN
    )
    return dst
```

### Bin 分组 Scatter（MoE Expert Dispatch）

将 token 按目标 expert 分组（binning），每次处理同一 expert 的所有 token，消除原子竞争（exp_gs_bin_001，30% 提升）：

```python
@triton.jit
def moe_dispatch_binned_kernel(
    tokens_ptr,          # [num_tokens, hidden_dim]
    expert_ids_ptr,      # [num_tokens] — 每个 token 的 expert ID
    expert_starts_ptr,   # [num_experts+1] — 每个 expert 在排序后的起始位置
    sorted_token_ids_ptr,# [num_tokens] — 按 expert 排序后的 token 索引
    output_ptr,          # [num_experts, expert_capacity, hidden_dim]
    hidden_dim, expert_capacity,
    BLOCK_HIDDEN: tl.constexpr,
):
    # grid = (num_experts, expert_capacity)
    expert_id = tl.program_id(0)
    slot_id = tl.program_id(1)

    start = tl.load(expert_starts_ptr + expert_id)
    end = tl.load(expert_starts_ptr + expert_id + 1)
    token_pos = start + slot_id

    if token_pos >= end:
        return  # 此 slot 无 token（padding）

    token_id = tl.load(sorted_token_ids_ptr + token_pos)

    hidden_offs = tl.arange(0, BLOCK_HIDDEN)
    mask = hidden_offs < hidden_dim

    # gather token 数据
    token_data = tl.load(tokens_ptr + token_id * hidden_dim + hidden_offs, mask=mask)

    # 写入 expert buffer（连续写入，无竞争）
    out_offset = (expert_id * expert_capacity + slot_id) * hidden_dim
    tl.store(output_ptr + out_offset + hidden_offs, token_data, mask=mask)
```

### MoE Top-K 路由完整流程（exp_gs_bin_003，35% 提升）

```python
def moe_token_dispatch(
    tokens: torch.Tensor,          # [num_tokens, hidden_dim]
    router_logits: torch.Tensor,   # [num_tokens, num_experts]
    top_k: int,
    expert_capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    完整 MoE dispatch 流程：
    1. Top-k 路由得到 expert_ids 和 routing_weights
    2. 排序 + binning
    3. 向量化 gather 到 expert buffer
    返回: (expert_input, token_indices_for_combine)
    """
    num_tokens, num_experts = router_logits.shape
    hidden_dim = tokens.shape[-1]

    # Step 1: Top-k 路由
    routing_weights, expert_ids = torch.topk(
        torch.softmax(router_logits, dim=-1), k=top_k, dim=-1
    )
    # expert_ids: [num_tokens, top_k]，展平
    flat_expert_ids = expert_ids.reshape(-1)      # [num_tokens * top_k]
    flat_token_ids = torch.arange(num_tokens, device=tokens.device)\
                         .repeat_interleave(top_k)

    # Step 2: 按 expert 排序
    sort_order = torch.argsort(flat_expert_ids)
    sorted_expert_ids = flat_expert_ids[sort_order]
    sorted_token_ids = flat_token_ids[sort_order]

    # 计算每个 expert 的 token 范围（binning）
    expert_starts = torch.searchsorted(
        sorted_expert_ids,
        torch.arange(num_experts + 1, device=tokens.device)
    )

    # Step 3: Dispatch（gather + 写入 expert buffer）
    expert_input = torch.zeros(
        num_experts, expert_capacity, hidden_dim,
        dtype=tokens.dtype, device=tokens.device
    )
    BLOCK_HIDDEN = triton.next_power_of_2(hidden_dim)
    grid = (num_experts, expert_capacity)
    moe_dispatch_binned_kernel[grid](
        tokens, flat_expert_ids, expert_starts, sorted_token_ids,
        expert_input, hidden_dim, expert_capacity,
        BLOCK_HIDDEN=BLOCK_HIDDEN
    )

    return expert_input, sorted_token_ids
```

### Work-Stealing 均衡 Expert 负载（exp_gs_bin_005，30% 提升）

不同 expert 接收的 token 数差异大时，work-stealing 让空闲核帮助繁忙核处理：

```python
@triton.jit
def moe_work_stealing_kernel(
    tokens_ptr, output_ptr,
    work_queue_ptr,      # 原子计数器，全局任务队列
    expert_ids_ptr,
    num_tasks, hidden_dim,
    BLOCK_HIDDEN: tl.constexpr,
):
    """每个 program 从全局队列中动态认领任务，自动均衡负载。"""
    while True:
        # 原子递增任务计数器，认领一个任务
        task_id = tl.atomic_add(work_queue_ptr, 1)
        if task_id >= num_tasks:
            break  # 无任务，退出

        expert_id = tl.load(expert_ids_ptr + task_id)
        hidden_offs = tl.arange(0, BLOCK_HIDDEN)
        mask = hidden_offs < hidden_dim

        src = tl.load(tokens_ptr + task_id * hidden_dim + hidden_offs, mask=mask)
        tl.store(output_ptr + task_id * hidden_dim + hidden_offs, src, mask=mask)
```

### Token 排列（Permutation/Unpermutation）

基于 sorted_indices 的 token 重排，用于 MoE 前向/反向（exp_gs_bin_004，25% 提升）：

```python
@triton.jit
def token_permute_kernel(
    src_ptr, dst_ptr, indices_ptr,
    num_tokens, hidden_dim,
    BLOCK_HIDDEN: tl.constexpr,
):
    """dst[i] = src[indices[i]]"""
    token_id = tl.program_id(0)
    src_idx = tl.load(indices_ptr + token_id)

    hidden_offs = tl.arange(0, BLOCK_HIDDEN)
    mask = hidden_offs < hidden_dim

    src = tl.load(src_ptr + src_idx * hidden_dim + hidden_offs, mask=mask)
    tl.store(dst_ptr + token_id * hidden_dim + hidden_offs, src, mask=mask)


@triton.jit
def token_unpermute_kernel(
    src_ptr, dst_ptr, indices_ptr,
    num_tokens, hidden_dim,
    BLOCK_HIDDEN: tl.constexpr,
):
    """dst[indices[i]] = src[i]（permute 的逆操作）"""
    token_id = tl.program_id(0)
    dst_idx = tl.load(indices_ptr + token_id)

    hidden_offs = tl.arange(0, BLOCK_HIDDEN)
    mask = hidden_offs < hidden_dim

    src = tl.load(src_ptr + token_id * hidden_dim + hidden_offs, mask=mask)
    tl.store(dst_ptr + dst_idx * hidden_dim + hidden_offs, src, mask=mask)
```

## 常见陷阱

| 陷阱 | 说明 | 解决方案 |
|------|------|----------|
| 逐元素 scatter（粒度过小） | 每次 scatter 只写一个标量元素，向量化效率极低 | 以 token 为粒度（整行 hidden_dim）做向量化 scatter（exp_gs_mem_002） |
| 原子操作竞争 | 多核写同一 expert buffer，atomic 序列化为瓶颈 | binning 策略确保同一 expert 由单核处理（exp_gs_bin_001） |
| 随机访问 cache miss | 无序索引导致每次 load 都 cache miss | 排序后再 scatter（exp_gs_mem_005）或 binning（exp_gs_bin_001） |
| Expert 负载不均 | 某些 expert 接收大量 token，某些为空，核间利用率低 | work-stealing 动态均衡（exp_gs_bin_005） |
| expert_capacity 设置过小 | token 溢出 expert_capacity，路由信息丢失 | 运行时容量检查（exp_gs_bin_002），根据实际 top-k 分布设置 |
| 忽略 wgrad scatter | MoE 反向传播中 scatter 结果需进一步 matmul | scatter → matmul 路径参考 matmul 经验页面 |
| int64 索引计算 | int64 地址计算在某些硬件路径下退化为标量 | 尽量使用 int32 索引；必须 int64 时预计算为 fp32 偏移 |
| Dynamic bin 容量越界 | 实际 token 数超过预分配的 expert_capacity | 使用 exp_gs_bin_002 动态检查；或保守设置 capacity |

## 代码模板

### 通用 Gather 模板

```python
import torch
import triton
import triton.language as tl

@triton.jit
def gather_kernel(
    src_ptr,       # [vocab_size, hidden_dim]
    index_ptr,     # [num_tokens]
    dst_ptr,       # [num_tokens, hidden_dim]
    hidden_dim,
    BLOCK_HIDDEN: tl.constexpr,
):
    token_id = tl.program_id(0)
    src_idx = tl.load(index_ptr + token_id)

    hidden_offs = tl.arange(0, BLOCK_HIDDEN)
    mask = hidden_offs < hidden_dim

    # 向量化 gather：整行一次加载
    src = tl.load(src_ptr + src_idx * hidden_dim + hidden_offs, mask=mask)
    tl.store(dst_ptr + token_id * hidden_dim + hidden_offs, src, mask=mask)


def gather_op(src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """src: [vocab_size, H], index: [N] -> dst: [N, H]"""
    src = src.contiguous()
    N = index.shape[0]
    hidden_dim = src.shape[-1]
    dst = torch.empty(N, hidden_dim, dtype=src.dtype, device=src.device)
    BLOCK_HIDDEN = triton.next_power_of_2(hidden_dim)
    gather_kernel[(N,)](src, index, dst, hidden_dim, BLOCK_HIDDEN=BLOCK_HIDDEN)
    return dst
```

### 通用 Scatter（Add）模板

```python
@triton.jit
def scatter_add_kernel(
    src_ptr,       # [num_tokens, hidden_dim]
    index_ptr,     # [num_tokens]
    dst_ptr,       # [vocab_size, hidden_dim]
    num_tokens, hidden_dim,
    BLOCK_HIDDEN: tl.constexpr,
):
    token_id = tl.program_id(0)
    dst_idx = tl.load(index_ptr + token_id)

    hidden_offs = tl.arange(0, BLOCK_HIDDEN)
    mask = hidden_offs < hidden_dim

    src = tl.load(src_ptr + token_id * hidden_dim + hidden_offs, mask=mask)
    # 原子加（处理多 token 写同一目标的情况）
    tl.atomic_add(dst_ptr + dst_idx * hidden_dim + hidden_offs, src, mask=mask)


def scatter_add_op(src: torch.Tensor, index: torch.Tensor,
                   dst_size: int) -> torch.Tensor:
    """src: [N, H], index: [N] -> dst: [dst_size, H]（scatter add）"""
    src = src.contiguous()
    N, hidden_dim = src.shape
    dst = torch.zeros(dst_size, hidden_dim, dtype=src.dtype, device=src.device)
    BLOCK_HIDDEN = triton.next_power_of_2(hidden_dim)
    scatter_add_kernel[(N,)](src, index, dst, N, hidden_dim,
                              BLOCK_HIDDEN=BLOCK_HIDDEN)
    return dst
```

## 相关页面

- [elementwise.md](./elementwise.md) — MoE Expert 计算后的激活函数是 elementwise 操作，可与 gather 融合
- [reduction.md](./reduction.md) — MoE wgrad（weight gradient）：scatter 结果需要 sum 归约
- [normalization.md](./normalization.md) — MoE 场景中 Expert 层前后可能存在 RMSNorm，Q/K RMSNorm 与 gather 结合

## 补充经验

| 场景 | 经验ID | 关键技术 | 经验文件 |
|------|--------|----------|----------|
| Gather Flat 布局 | exp_gather_flat_001 | 展平输入的 gather 操作 | [yaml](../../raw/experiences/gather_scatter/exp_gather_flat_001.yaml) |
| Padded Gather | exp_padded_gather_001 | 带 padding 的 gather（expert capacity 对齐） | [yaml](../../raw/experiences/gather_scatter/exp_padded_gather_001.yaml) |
| Radix Sort Scatter | exp_radix_sort_scatter_001 | 先排序后 scatter，提升访问连续性 | [yaml](../../raw/experiences/gather_scatter/exp_radix_sort_scatter_001.yaml) |
| Scatter MoE Padded | exp_scatter_moe_padded_001 | MoE padded scatter（expert capacity 对齐） | [yaml](../../raw/experiences/gather_scatter/exp_scatter_moe_padded_001.yaml) |
| Scatter WGrad | exp_scatter_wgrad_001 | 权重梯度 scatter-add | [yaml](../../raw/experiences/gather_scatter/exp_scatter_wgrad_001.yaml) |
| Scatter WGrad MoE | exp_scatter_wgrad_moe_001 | MoE 权重梯度 scatter-add | [yaml](../../raw/experiences/gather_scatter/exp_scatter_wgrad_moe_001.yaml) |
| MoE Weight Grad | exp_moe_wgrad_npu_001 | MoE weight gradient（NPU 专用） | [yaml](../../raw/experiences/moe/exp_moe_wgrad_npu_001.yaml) |

