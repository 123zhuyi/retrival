---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: rope
dimension: "2.3"
title: RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引
description: 针对 Ascend 910B 上 RoPE 反向的调度策略，覆盖 grid-stride 循环处理可变总块数、IS_VARLEN 变长路径、HAS_KV_LENS KV cache 偏移与 `do_not_specialize` 动态特化，收益 25%。
tags: [rope, scheduling, grid_stride, varlen, kv_cache, backward, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/rope/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引

## 概述

RoPE 反向传播需要对 Q/K 做逆向旋转（把 cos/sin 符号取反）。在 LLM 推理/训练场景中，它还面临两个调度挑战：
1. **变长序列（varlen batch）**：一个 batch 内各样本长度不同，不能简单按 `bs * seq_len` 均匀切分；
2. **KV cache 偏移**：推理阶段 query 的位置可能远小于 key/value 的已缓存长度，需要按 `kv_len` 查 cos/sin 表。

本卡唯一的 experience `exp_rope_backward_001` 通过 **grid-stride loop + 动态特化（`do_not_specialize=["seq_len"]`）+ IS_VARLEN/HAS_KV_LENS 分支** 解决这些问题，收益 25%。

**核心 trade-off**：
- Grid-stride 循环可以用固定 grid 覆盖动态总块数，减少 launch 调参，但循环内需要维护更多状态；
- `do_not_specialize` 避免 `seq_len` 变化时重复编译，但会牺牲部分静态优化机会；
- IS_VARLEN/HAS_KV_LENS 用 `tl.constexpr` 特化，编译时展开分支，避免运行时判断开销。

## 共同模式

### 1. Grid-Stride Loop：固定 grid 覆盖动态 workload

**问题根因**：变长 batch 或 KV cache 场景下，`bs * num_seq_blocks` 随输入变化。若 grid 直接等于总块数，短序列时空跑 program，长序列时 grid 过大。

**收益明显条件**：
- 序列长度或 batch size 动态变化；
- 总块数不是固定常量；
- 希望用固定核心数做负载均衡。

**实现要点**：
```python
pid = tl.program_id(axis=0)
grid_size = tl.num_programs(axis=0)
total_blocks = bs * num_seq_blocks
for block_id in range(pid, total_blocks, grid_size):
    ...
```
- 固定 grid，每个 program 处理多个 block；
- `total_blocks` 运行时计算，但循环步长固定；
- 空 block 用 mask 或早退跳过。

**常见坑**：
- `total_blocks` 为 0 时要确保没有 program 进入无效循环；
- grid-stride 循环内不要频繁做 `tl.num_programs` 同步；
- 循环变量如果是运行时值，注意类型为 int32/int64。

### 2. 变长序列路径：IS_VARLEN

**问题根因**：varlen batch 中每个 block 的真实序列位置、实际长度、cos/sin 偏移都存储在 `chunk_indices_ptr` 中，不能从 `block_id` 直接推导。

**收益明显条件**：
- 使用 `torch.nn.utils.rnn.pack_padded_sequence` 或类似 rmpad 逻辑；
- 一个 batch 内样本长度差异大；
- 需要避免按最大长度 padding 造成的空跑。

**实现要点**：
```python
if IS_VARLEN:
    chunk_idx = tl.load(chunk_indices_ptr + block_id * 5 + 1)
    seq_start = tl.load(chunk_indices_ptr + block_id * 5 + 2)
    sin_cos_offset = tl.load(chunk_indices_ptr + block_id * 5 + 3)
    actual_seq_len = tl.load(chunk_indices_ptr + block_id * 5 + 4)

    block_start_seq_idx = chunk_idx * TOKEN_BLOCK_SIZE
    seq_offsets = block_start_seq_idx + tl.arange(0, TOKEN_BLOCK_SIZE)
    seq_mask = seq_offsets < actual_seq_len
    global_seq_offsets = seq_start + seq_offsets
    sin_cos_seq_offsets = sin_cos_offset + seq_offsets
```
- 每个 block 从 `chunk_indices` 读取 4 个元数据；
- `actual_seq_len` 用于 mask，避免处理 padding；
- `sin_cos_offset` 保证 rmpad 后仍能从全局 cos/sin 表正确查表。

**常见坑**：
- `chunk_indices` 的 stride 和字段顺序要与 host 侧一致；
- 非 varlen 路径下不要误加载 `chunk_indices`；
- `batch_idx` 在 varlen 路径下通常为 0，因为样本已经 rmpad 到同一张“大表”。

### 3. KV Cache 感知索引：HAS_KV_LENS

**问题根因**：推理时 query 只处理当前 token，而 key/value 已经缓存了前面所有 token。cos/sin 查表位置应由 KV cache 长度决定，而不是当前 query 位置。

**收益明显条件**：
- 增量推理（incremental decoding）；
- KV cache 已按 `kv_len` 累积；
- 需要把当前 token 的 cos/sin 偏移到 `kv_len + local_offset`。

**实现要点**：
```python
if HAS_KV_LENS:
    kv_len = tl.load(kv_lens_ptr + batch_idx)
    sin_cos_seq_offsets = kv_len + seq_offsets
else:
    sin_cos_batch_offset = tl.where(cos_bs == 1, 0, batch_idx * seq_len)
    sin_cos_seq_offsets = sin_cos_batch_offset + seq_offsets
```
- 有 KV cache 时用 `kv_len` 作为 cos/sin 行偏移；
- 无 KV cache 时按 batch/seq 二维查表；
- `cos_bs == 1` 支持 cos/sin 在 batch 间共享。

**常见坑**：
- `kv_len` 的单位要与 `seq_offsets` 一致（token 数）；
- 多 batch 时 `kv_lens_ptr` 的索引不能和 varlen 的 `batch_idx` 混淆；
- KV cache 更新后要及时同步 `kv_len`。

### 4. 动态特化：`do_not_specialize` 避免编译风暴

**问题根因**：Triton 默认会对每个不同 `seq_len` 生成专门的 binary。推理时 `seq_len` 随请求变化，会导致大量重复编译和缓存膨胀。

**收益明显条件**：
- 推理服务中 `seq_len` 是动态的；
- 编译时间或缓存占用成为瓶颈；
- kernel 内 `seq_len` 只用于 mask，不用于 tile 形状。

**实现要点**：
```python
@triton.jit(do_not_specialize=["seq_len"])
def _rope_backward_kernel(..., seq_len, ...):
```
- 把 `seq_len` 加入 `do_not_specialize`；
- 用 mask 处理边界，而不是用 `seq_len` 做编译期常量分块；
- 若 `seq_len` 用于 constexpr 计算，则不能加 `do_not_specialize`。

**常见坑**：
- 不要把 `BLOCK_SIZE` 或 `head_dim` 加入 `do_not_specialize`；
- `do_not_specialize` 后，编译器无法基于 `seq_len` 做静态展开，需用显式 mask 保证正确性；
- 只在运行时变长且非 shape-defining 的参数上使用。

### 5. 反向旋转：sin 符号取反

**问题根因**：RoPE 反向等价于逆向旋转，公式为：
```
out[i]     = x[i]     * cos + x[i+d/2] * sin
out[i+d/2] = x[i+d/2] * cos - x[i]     * sin
```
即把正向的 `-sin` 与 `+sin` 互换。

**收益明显条件**：
- 训练反向传播需要 dQ/dK；
- 复用正向 cos/sin 加载逻辑，仅改符号。

**实现要点**：
- 复用正向的 cos/sin 加载和 head 循环；
- 用 `tl.constexpr` 的 `INVERSE` 标志在编译期选择正/反向；
- 共享 cos/sin tile，避免重复加载。

**常见坑**：
- 符号写反会导致旋转方向错误，反向传播结果偏差；
- 不要运行时判断正反向，应编译期特化；
- 反向时 `cos/sin` 表本身不变，只改变运算符号。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_rope_backward_001` | 25% | RoPE 反向调度：grid-stride 循环覆盖动态总块数，`IS_VARLEN` 支持变长序列，`HAS_KV_LENS` 处理 KV cache 偏移，并用 `do_not_specialize` 避免 seq_len 变化导致编译风暴。 |

## 代表 Kernel

选择唯一的 `exp_rope_backward_001` 作为代表。下面为合成的 RoPE 反向 kernel，展示 **grid-stride、IS_VARLEN、HAS_KV_LENS、反向旋转** 四个核心调度模式：

```python
@triton.jit(do_not_specialize=["seq_len"])
def _rope_backward_kernel(
    q_ptr,
    q_batch_stride,
    q_seq_stride,
    k_ptr,
    k_batch_stride,
    k_seq_stride,
    cos_ptr,
    cos_row_stride,
    sin_ptr,
    sin_row_stride,
    seq_len,
    num_seq_blocks,
    chunk_indices_ptr,
    kv_lens_ptr,
    bs: tl.constexpr,
    cos_bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    half_rope_dim: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_KV_LENS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    total_blocks = bs * num_seq_blocks

    half_rope_dim_offsets = tl.arange(0, half_rope_dim)
    half_rope_dim_mask = half_rope_dim_offsets < half_rope_dim
    rope_dim_offsets = tl.arange(0, rope_dim)
    rope_dim_mask = rope_dim_offsets < rope_dim

    for block_id in range(pid, total_blocks, grid_size):
        # ---------- 变长 vs 非变长路径 ----------
        if IS_VARLEN:
            chunk_idx = tl.load(chunk_indices_ptr + block_id * 5 + 1)
            seq_start = tl.load(chunk_indices_ptr + block_id * 5 + 2)
            sin_cos_offset = tl.load(chunk_indices_ptr + block_id * 5 + 3)
            actual_seq_len = tl.load(chunk_indices_ptr + block_id * 5 + 4)

            block_start_seq_idx = chunk_idx * TOKEN_BLOCK_SIZE
            seq_offsets = block_start_seq_idx + tl.arange(0, TOKEN_BLOCK_SIZE)
            seq_mask = seq_offsets < actual_seq_len
            global_seq_offsets = seq_start + seq_offsets
            sin_cos_seq_offsets = sin_cos_offset + seq_offsets
            batch_idx = 0
        else:
            batch_idx = block_id // num_seq_blocks
            seq_block_id = block_id % num_seq_blocks

            block_start_seq_idx = seq_block_id * TOKEN_BLOCK_SIZE
            seq_offsets = block_start_seq_idx + tl.arange(0, TOKEN_BLOCK_SIZE)
            seq_mask = seq_offsets < seq_len
            global_seq_offsets = seq_offsets

            if HAS_KV_LENS:
                kv_len = tl.load(kv_lens_ptr + batch_idx)
                sin_cos_seq_offsets = kv_len + seq_offsets
            else:
                sin_cos_batch_offset = tl.where(cos_bs == 1, 0, batch_idx * seq_len)
                sin_cos_seq_offsets = sin_cos_batch_offset + seq_offsets

        # ---------- 同一位置 Q/K 所有 head 复用 cos/sin ----------
        cos_token_ptr = cos_ptr + sin_cos_seq_offsets[:, None] * cos_row_stride
        sin_token_ptr = sin_ptr + sin_cos_seq_offsets[:, None] * sin_row_stride

        cos_block = tl.load(
            cos_token_ptr + half_rope_dim_offsets[None, :],
            mask=seq_mask[:, None] & half_rope_dim_mask[None, :],
            other=0.0,
        )
        sin_block = tl.load(
            sin_token_ptr + half_rope_dim_offsets[None, :],
            mask=seq_mask[:, None] & half_rope_dim_mask[None, :],
            other=0.0,
        )

        # ---------- Q heads：反向旋转 ----------
        head_q_offsets = tl.arange(0, n_qh)
        q_offsets = (
            batch_idx * q_batch_stride
            + global_seq_offsets[:, None, None] * q_seq_stride
            + head_q_offsets[None, :, None] * hd
            + nope_dim
            + rope_dim_offsets[None, None, :]
        )
        q_mask = (
            seq_mask[:, None, None]
            & (head_q_offsets[None, :, None] < n_qh)
            & rope_dim_mask[None, None, :]
        )
        q_tile = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)

        # 把 rope_dim 拆成前后两半做逆向旋转
        q1 = tl.reshape(q_tile, (TOKEN_BLOCK_SIZE, n_qh, half_rope_dim, 2), can_reorder=True)
        x_l, x_h = tl.split(q1)
        # 反向：out_l = x_l * cos + x_h * sin
        #       out_h = x_h * cos - x_l * sin
        new_l = x_l * cos_block[:, None, :] + x_h * sin_block[:, None, :]
        new_h = x_h * cos_block[:, None, :] - x_l * sin_block[:, None, :]
        new_q = tl.interleave(new_l, new_h)
        tl.store(q_ptr + q_offsets, new_q, mask=q_mask)

        # ---------- K heads：同样的反向旋转 ----------
        head_k_offsets = tl.arange(0, n_kh)
        k_offsets = (
            batch_idx * k_batch_stride
            + global_seq_offsets[:, None, None] * k_seq_stride
            + head_k_offsets[None, :, None] * hd
            + nope_dim
            + rope_dim_offsets[None, None, :]
        )
        k_mask = (
            seq_mask[:, None, None]
            & (head_k_offsets[None, :, None] < n_kh)
            & rope_dim_mask[None, None, :]
        )
        k_tile = tl.load(k_ptr + k_offsets, mask=k_mask, other=0.0)
        k1 = tl.reshape(k_tile, (TOKEN_BLOCK_SIZE, n_kh, half_rope_dim, 2), can_reorder=True)
        x_l, x_h = tl.split(k1)
        new_l = x_l * cos_block[:, None, :] + x_h * sin_block[:, None, :]
        new_h = x_h * cos_block[:, None, :] - x_l * sin_block[:, None, :]
        new_k = tl.interleave(new_l, new_h)
        tl.store(k_ptr + k_offsets, new_k, mask=k_mask)
```

**关键实现细节**：
- 固定 grid + grid-stride 循环覆盖动态 `total_blocks`；
- `IS_VARLEN` 分支从 `chunk_indices` 读取真实序列位置和 cos/sin 偏移；
- `HAS_KV_LENS` 分支用 `kv_len` 计算 cos/sin 查表位置；
- `do_not_specialize=["seq_len"]` 避免推理时不同长度触发重复编译；
- 反向旋转通过 sin 符号取反实现，cos/sin 表本身不变且被所有 head 复用。

## 已知实例

- `exp_rope_backward_001` — Scheduling（收益 25%）
  - RoPE 反向调度优化：grid-stride 循环、`IS_VARLEN` 变长支持、`HAS_KV_LENS` KV cache 索引、`do_not_specialize` 动态特化。代表 kernel 见上。

## 关联知识

- 开发指南: [RoPE（旋转位置编码）算子优化](../../reference/op-class-rope.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [Legacy 缓存读写协议（triton-knowledge-retrieval）](../../reference/SPEC-Retrieve-legacy.md) — 同属优化知识，主题相关
- 对应指南: [Ascend 910B2/910B3 Hardware Reference](../../reference/ascend-910b.md) — 主题相关：共享 910b2, 910b3
- 对应指南: [Ascend NPU 特有语法与模式](../../reference/ascend-patterns.md) — 主题相关：共享 910b2, 910b3
- 对应指南: [RoPE（旋转位置编码）算子优化](../../reference/op-class-rope.md) — 主题相关：共享 910b2, 910b3, backward, rope
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](../../ops/fused-infer-attention-score.md) — 主题相关：共享 kv_cache
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](../../ops/incre-flash-attention.md) — 主题相关：共享 kv_cache
- 相关主题: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](../../ops/kv-quant-sparse-flash-attention.md) — 主题相关：共享 kv_cache
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../../ops/mla-preprocess.md) — 主题相关：共享 kv_cache, rope
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../../ops/mla-prolog-v3.md) — 主题相关：共享 rope
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](../../ops/mla-prolog.md) — 主题相关：共享 kv_cache, rope
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](../../ops/nsa-compress-with-cache.md) — 主题相关：共享 kv_cache
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](../../ops/prompt-flash-attention.md) — 主题相关：共享 kv_cache
- 相关主题: [RoPE paired-channel pointer addressing](../../ops/rope-pointer-addressing.md) — 主题相关：共享 rope
- 相关主题: [SparseFlashAttention inference (sparse KV) on Ascend NPU](../../ops/sparse-flash-attention.md) — 主题相关：共享 kv_cache
- 相关主题: [向量 gather 随机地址在 Ascend 上极慢，二分/搜索类走 per-expert 标量+grid 并行](../memory-sram/gather-random-address-slow.md) — 同属优化知识，主题相关
- 相关主题: [输入进入 kernel 前必须 contiguous](../memory-sram/input-contiguous.md) — 同属优化知识，主题相关
- 相关主题: [pertensor scale 必须物化为连续 [1,N]，禁止 stride-0 广播 load](../memory-sram/pertensor-scale-materialize.md) — 同属优化知识，主题相关
- 相关主题: [Ascend 大 Shape 连续分块内存访问优化](../memory-sram/traj-general_ascend-memory_access.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](../memory-sram/traj-general_perf-memory_access.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [RoPE 内存访问优化：cos/sin 复用、ALIGNED 向量化加载与动态特化](../memory-sram/traj-rope-memory_access.md) — 主题相关：共享 910b2, 910b3, rope
- 相关主题: [Triton-Ascend kernel 调试排查清单](../meta-skills/debugging-checklist.md) — 同属优化知识，主题相关
- 相关主题: [避免 Stride 访问：优先连续化后一维遍历](../meta-skills/traj-general_perf-perf_pitfall.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [Llama4 RoPE 在 Ascend NPU 上的硬件适配：UB Tiling、FMA 与交错复数布局](../portability/traj-rope-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, rope
- 相关主题: [按输出元素数（而非输入）分配 grid，且钳制到 num_cores](grid-by-output-count.md) — 同属优化知识，主题相关
- 相关主题: [grid 总数不超过核数](grid-cap-to-core-num.md) — 同属优化知识，主题相关
- 相关主题: [1D grid 硬上限 65535，超出用 2D grid 或增大 BLOCK](grid-limit-65535.md) — 同属优化知识，主题相关
- 相关主题: [grid_size 必须与 num_cores 入参严格相等](grid-strict-match-num-cores.md) — 同属优化知识，主题相关
- 相关主题: [算子融合模式与约束决策](operator-fusion-patterns.md) — 主题相关：共享 rope
- 相关主题: [pad slot 单次检查提前退出，跳过无效填充序列](pad-slot-early-exit.md) — 主题相关：共享 kv_cache
- 相关主题: [persistent kernel 模式（grid-stride 摊还启动开销）](persistent-kernel-pattern.md) — 主题相关：共享 grid_stride
- 相关主题: [Activation Scheduling 优化聚合](traj-activation-scheduling.md) — 主题相关：共享 910b2, 910b3, grid_stride, scheduling
- 相关主题: [Attention Scheduling 优化聚合](traj-attention-scheduling.md) — 主题相关：共享 910b2, 910b3, grid_stride, scheduling
- 相关主题: [Causal Conv1D 调度优化：Prefill/Update 布局分野、2D Grid 与 pad_slot 早退](traj-conv-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Embedding Backward Scheduling 优化聚合](traj-embedding-scheduling.md) — 主题相关：共享 910b2, 910b3, backward, scheduling
- 相关主题: [Radix Sort Scatter 的 Lookback 调度优化](traj-gather_scatter-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [复杂算子拆分：降低编译与调度开销](traj-general_perf-op_split.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [大词表 Entropy Loss Backward 调度融合优化](traj-loss-scheduling.md) — 主题相关：共享 910b2, 910b3, grid_stride, scheduling
- 相关主题: [Matmul Scheduling 优化聚合](traj-matmul-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Misc 算子 Scheduling 优化聚合](traj-misc-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Normalization Scheduling 优化聚合](traj-normalization-scheduling.md) — 主题相关：共享 910b2, 910b3, grid_stride, scheduling
- 相关主题: [RoPE 计算优化：cos/sin 复用、向量化旋转与多模态位置编码融合](traj-rope-compute.md) — 主题相关：共享 910b2, 910b3, grid_stride, rope
- 相关主题: [Solve_Tril 调度优化：三层分级 kernel、分块递推与 Grid 并行](traj-solve_tril-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling

<!-- okf:related:end -->
