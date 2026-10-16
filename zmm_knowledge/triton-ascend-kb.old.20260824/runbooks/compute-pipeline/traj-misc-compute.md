---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: misc
dimension: "2.1"
title: Misc 算子计算优化：向量化、微内核与阶段融合
description: 针对 Ascend 910B 系列上无法归入标准算子类别的 miscellaneous 计算核（采样、排序、三角求解、分块求逆、直方图、索引融合等）的计算优化经验，核心是把串行标量循环改造为向量化/并行计算、用微内核处理小规模矩阵、并将多阶段规约融合为单遍。
tags: [misc, compute, vectorization, micro_kernel, kernel_fusion, numerical_stability, sampling, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/misc/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-11T00:00:00Z"
---

# Misc 算子计算优化：向量化、微内核与阶段融合

## 概述

本组 16 条 experience 覆盖 Ascend 910B2/910B3 上难以归入标准算子类别（attention/matmul/normalization/conv 等）的 **miscellaneous 计算核**。这些算子看似差异很大——从投机解码的拒绝采样、Top-P 采样，到三角求解、分块矩阵求逆、位onic 排序、radix 转换、直方图计数——但它们的性能瓶颈有高度共性：

1. **串行标量循环占主导**：如投机解码中对 `num_draft_tokens` 逐个验证、直方图中对每个元素串行计数；
2. **多阶段规约反复读写 HBM**：如 Top-P filter 中 softmax → cumsum → mask → renormalize 若拆开，中间概率张量多次落盘；
3. **小规模矩阵/张量操作未充分利用硬件**：如三角求解、分块求逆若用标量循环，无法饱和 Cube/Vector 单元；
4. **数值稳定性与分支发散**：如采样中的概率连乘下溢、排序中的有符号数比较、Top-P 中的条件分支。

**核心 trade-off**：

| 优化手段 | 收益 | 代价/约束 |
|---|---|---|
| **标量循环 → 向量化/并行** | 通常 30%–300% | 需要处理变长和 mask，可能增加寄存器使用 |
| **GEMM/Cube 替代标量 ALU 规约** | 直方图等场景 50%+ | 要求问题可转化为 one-hot + dot 模式 |
| **多阶段融合为单 kernel** | 减少 HBM 往返 | kernel 逻辑复杂，中间结果需驻留 SRAM/寄存器 |
| **16×16 微内核 tiling** | 小矩阵/三角求解 25%–35% | 固定块大小，需处理边界和 mask |
| **FP32 累加 / Log-space / max-shift** | 避免数值下溢/上溢 | 增加少量转换和寄存器开销 |
| **`tl.constexpr` 消除运行时分支** | 减少 warp 发散 | 仅适用于编译期已知的配置 |

## 共同模式

### 1. 把串行标量循环改造成向量化/并行

这是本组覆盖最广的优化方向，出现在投机解码验证、拒绝采样、多项式采样、直方图计数等 experience 中。

**根因**：Triton/Ascend 的 SIMD/SIMT 执行模型对向量操作高度优化，标量循环会严重低占硬件并行度，同时带来频繁的分支和地址计算。

**典型改造**：
- **投机解码验证**（`exp_spec_decode_verify_scalar`）：把 `for pos in range(num_draft_tokens)` 的逐个 token 验证改为一次性加载整个 draft token block 和对应概率，向量化计算 `target_prob/draft_prob` 的连乘/连加，再用 mask 找出接受位置。
- **拒绝采样**（`exp_reject_sampler_001`）：用 `tl.load` 一次性加载 `max_spec_len` 个 draft token/prob，向量比较 `target_prob/draft_prob >= random`，生成 boolean mask，最后通过 prefix sum 或 `tl.sum(mask)` 得到 `accept_len`。
- **多项式采样**（`exp_multinomial_001`）：把标量二分搜索改成向量二分搜索，`start/end/mid` 都是 `NBLOCK` 宽的向量，同时处理多个样本。
- **直方图计数**（`exp_hist_count_001`、`exp_digit_hist_001`）：用 `x[:, None] == bins[None, :]` 做广播比较，得到 one-hot 矩阵后再用 `tl.dot` 或 `tl.sum` 做并行计数，替代逐元素 `if x == bin then count++`。

**收益明显条件**：
- 循环迭代次数小但独立（如 `max_spec_len ≤ 16`、bins ≤ 16、vocab 大但 batch 并行）；
- 循环体内没有跨迭代的真数据依赖（可通过 prefix sum/reduction 处理累积依赖）。

**常见坑**：
- 概率连乘容易下溢，可改在 log-space 累加：`log_pi = sum(log(target_prob) - log(draft_prob))`。
- 变长序列要用 mask 跳过 padding，避免把 padding 计入结果。
- 向量比较后若需要找“第一个满足条件的位置”，可用 `tl.min(tl.where(mask, index, MAX))` 而非标量循环。

### 2. 用微内核（Micro Kernel）处理小规模矩阵运算

三角求解（`solve_tril`、`tril_solve`）和分块矩阵求逆（`block_inverse`）都采用了 **16×16 微内核 tiling**。

**根因**：这些线性代数操作的单个矩阵尺寸很小（如 64×64 或更小），无法直接用大 tile GEMM。16×16 与 Ascend Vector/Cube 的向量宽度和 tensor core 最小粒度对齐，能最大化每个 cycle 的有效计算。

**实现要点**：
- 把大矩阵拆成 16×16 块，用 `tl.make_block_ptr` 或显式指针加载；
- 在寄存器内通过 `tl.insert_slice` 拼装 32×32 等中间矩阵，避免反复读 HBM；
- 三角约束用 `tl.where` + boolean mask（`is_lower`、`on_diagonal`）实现无分支；
- 对变长序列用 `cu_seqlens` 定位每个样本的边界，避免 padding。

**收益明显条件**：
- 矩阵尺寸是 16 的倍数（如 64×64）；
- 存在批量的独立小矩阵（如 attention 中的 block-wise tril solve、SSM 中的分块求逆）。

### 3. 多阶段规约/变换融合为单一 kernel

Top-P filter、Top-P sampling、GRPO loss、fused index、mhc_post_fusion 都强调把多个阶段合并。

**典型融合**：
- **Top-P Filter**：`logits - max → exp → cumsum → mask → renormalize` 五阶段合一，中间概率张量不写 HBM。
- **Fused Index**：`GEMM → ReLU → Scale → Sum` 四阶段合一，Q 矩阵提前加载并在 N 方向复用。
- **MHC Post Fusion**：`matmul(comb_res_mix, residual) + broadcast_mul(post_layer_mix, x)` 合一。
- **GRPO Loss**：原本对 new/ref/old logits 分别做三遍 reduction，优化方向是单遍同时加载三个 tensor 并规约。

**根因**：misc 算子本身计算量不大，若阶段之间写 HBM，带宽和 launch 开销会主导延迟。

**收益明显条件**：
- 各阶段数据依赖是单向流水线（无反向依赖）；
- 中间结果不需要被其他 kernel 读取；
- 合并后的寄存器/UB 占用仍可接受。

**常见坑**：
- 不要为了融合而融合：若合并导致 register spill，反而更慢。
- softmax/cumsum 等规约操作需要 max-shift 或 LogSumExp 保证数值稳定。

### 4. 数值稳定性：max-shift、LogSumExp、FP32 累加

本组多个算子涉及概率、指数、倒数等敏感操作：

- **Top-P / Sampling**：softmax 前做 `logits - max(logits)`，防止 exp 上溢。
- **GRPO Loss**：用迭代 LogSumExp 计算 log-probabilities，分离 max 和 sum。
- **投机解码**：建议把概率连乘改到 log-space，避免 `pi` 下溢为 0。
- **分块求逆 / 状态重计算**：dot 操作 cast 到 FP32，`allow_tf32=False`，最终 RTNE 下cast。

**实现要点**：
- 规约累加（cumsum、sum、dot）用 FP32；
- 比较/搜索操作（如 binary search）对随机数做 `clamp(0.0001, 0.9999)`，避免边界越界；
- 排序 radix 转换时通过 MSB 翻转把有符号整数映射到无符号顺序。

### 5. 编译期分支消除

`tl.constexpr` 用于：
- `exp_top_p_sampling_001` 的 `strategy`（标准采样 vs noise injection）；
- `exp_radix_convert_001` 的 dtype（int8/16/32/64）；
- `exp_tril_solve_001` 的 `do_not_specialize=["T"]`，避免为每个序列长度重新编译。

**根因**：运行时 `if` 会导致 warp 发散；编译期确定分支可让编译器生成 specialized 代码并完全消除未使用路径。

### 6. 缓存与寄存器级数据复用

- **MHC Post Fusion**：`comb_res_mix` 在 N 方向外提加载，每个 weight 被 sequence 上多个 block 复用。
- **Fused Index**：Q 矩阵和 scaling factor 在 N 循环外加载，K 分块迭代复用 Q。
- **Block Inverse**：32×32 中间矩阵用 `tl.insert_slice` 在寄存器内拼装，不读 HBM。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_spec_decode_verify_scalar` | 300% | 投机解码 block verify 基线， scalar 循环遍历 `num_draft_tokens` 是瓶颈，应改为向量化验证与 mask 归约。 |
| `exp_digit_hist_001` | 50% | 数字直方图，把标量 bin 计数改为 one-hot 编码 + `tl.dot`，用 Cube 做并行计数。 |
| `exp_hist_count_001` | 50% | Top-K/稀疏注意力/MoE 路由的索引频次统计，用广播比较 + 寄存器局部累加替代全局原子加。 |
| `exp_fused_index_001` | 45% | FP8 索引/相似度搜索，融合 GEMM + ReLU + Scale + Sum，Q 矩阵在 N 方向复用。 |
| `exp_block_inverse_001` | 35% | 64×64 分块矩阵求逆，16×16 微内核 + `tl.insert_slice` 拼装 32×32 + FP32 累加 + `cu_seqlens` 变长。 |
| `exp_mhc_post_fusion_001` | 35% | MHC post fusion，合并 matmul + broadcast mul + add，HC 维需 16/32 对齐。 |
| `exp_topp_filter_001` | 35% | Top-P filter，融合 softmax、cumsum、mask、renormalize，用 `tl.where` 无分支。 |
| `exp_multinomial_001` | 35% | 多项式采样，向量化二分搜索，`do_not_specialize` 减少变长编译开销。 |
| `exp_reject_sampler_001` | 35% | 投机解码拒绝采样，向量加载 + 并行比较 + mask 归约替代标量 early-exit 循环。 |
| `exp_tril_solve_001` | 35% | 下三角求解，16×16 tiling + 向量化 reduction + `cu_seqlens` 变长处理。 |
| `exp_solve_tril_001` | 25% | 三角求解，16×16 block + UB 模拟 reshape + `tl.where` 无分支三角约束。 |
| `exp_recompute_w_u_001` | 25% | Attention 状态重计算，FP32 累加 + `allow_tf32=False` + 融合 gating scale + 变长路径。 |
| `exp_sort_bitonic_001` | 25% | Bitonic sort，用 `iinfo` 极值做 padding，batch 并行处理多个小数组。 |
| `exp_top_p_sampling_001` | 30% | Top-P sampling，向量 softmax/cumsum + 向量搜索 + `strategy` constexpr。 |
| `exp_grpo_loss_001` | 30% | GRPO loss，LogSumExp 稳定计算，优化方向是把 new/ref/old 三个 reduction loop 融合为单遍。 |
| `exp_radix_convert_001` | 20% | Radix sort 预处理，用 bit shift/mask 翻转 MSB，把有符号整数映射为可排序的无符号顺序。 |

## 代表 Kernel

选取 `exp_spec_decode_verify_scalar`（`171-rejection_random_sample_block_verify_kernel.py`）作为代表，原因有二：

1. **它是本组收益最高的经验（300%）**，典型展示了 misc 算子中最常见的瓶颈——**标量循环**；
2. 其优化路径（向量化加载 draft token 和概率、向量比较、mask 归约）对本组中 `exp_reject_sampler_001`、`exp_top_p_sampling_001`、`exp_multinomial_001` 等采样类经验具有直接参考价值。

下面展示的是**基线实现**，核心瓶颈即第 104 行起的 `for pos in range(num_draft_tokens)` 标量循环：

```python
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_block_verify_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,
    vec_len,
    NO_DRAFT_PROBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < vec_len
    is_greedy = tl.load(is_greedy_ptr + offsets, mask, other=1)
    not_greedy_mask = is_greedy == 0
    start_idxs = tl.where(offsets == 0, 0, tl.load(cu_num_draft_tokens_ptr + offsets - 1, not_greedy_mask))
    end_idxs = tl.load(cu_num_draft_tokens_ptr + offsets, not_greedy_mask)
    n_num_draft_tokens = end_idxs - start_idxs
    for req_i in range(BLOCK_SIZE):
        not_greedy = tl.get_element(not_greedy_mask, (req_i,))
        if not_greedy:
            rejected = False
            pi = 1.0
            uniform_prob = 1.0
            last_accepted_token_pos = -1
            start_idx = tl.get_element(start_idxs, (req_i,))
            req_idx = block_idx * BLOCK_SIZE + req_i
            num_draft_tokens = tl.get_element(n_num_draft_tokens, (req_i,))

            for pos in range(num_draft_tokens):
                draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
                target_prob = tl.load(target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id)
                tmp_uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
                uniform_prob = uniform_prob * tmp_uniform_prob

                if NO_DRAFT_PROBS:
                    draft_prob = 1
                else:
                    draft_prob = tl.load(draft_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id)

                pi = min(pi * target_prob / draft_prob, 1.0)
                if draft_prob > 0 and pi >= uniform_prob:
                    last_accepted_token_pos = pos
                    rejected = False
                else:
                    rejected = True

            if last_accepted_token_pos > -1:
                for pos in range(last_accepted_token_pos + 1):
                    token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
                    tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id)

            if rejected:
                recovered_token_id = tl.load(recovered_token_ids_ptr + start_idx + last_accepted_token_pos + 1)
                tl.store(
                    output_token_ids_ptr + req_idx * (max_spec_len + 1) + last_accepted_token_pos + 1,
                    recovered_token_id,
                )
            else:
                bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
                tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens, bonus_token_id)
```

**优化方向**：
- 把内层 `for pos in range(num_draft_tokens)` 改为一次性加载 `num_draft_tokens` 长度的 `draft_token_ids` 和概率向量；
- 用向量运算计算 `log(target_prob) - log(draft_prob)` 并做 prefix sum / cumulative max，避免概率连乘下溢；
- 用 `tl.where` 生成接受 mask，再用 `tl.sum` 或 prefix scan 得到 `accept_len` 和 `last_accepted_token_pos`；
- 输出写入同样用向量 store + mask，消除第 122–135 行的串行循环。

## 已知实例

- `exp_solve_tril_001` — Compute （收益 25%）
  - 三角求解，16×16 block + UB 模拟 reshape + `tl.where` 无分支三角约束。
  - 对应 kernel: `raw_experiences/extracted_kernels/solve_tril_16x16_kernel_paral_v3`
- `exp_block_inverse_001` — Compute （收益 35%）
  - 64×64 分块矩阵求逆，16×16 微内核 + `tl.insert_slice` 拼装 32×32 + FP32/RTNE + `cu_seqlens` 变长。
  - 对应 kernel: `raw_experiences/extracted_kernels/185-merge_16x16_to_64x64_inverse_kernel_reorder_all_masked.py`
- `exp_digit_hist_001` — Compute （收益 50%）
  - 数字直方图，one-hot 编码 + `tl.dot` 用 Cube 替代标量计数循环。
  - 对应 kernel: `raw_experiences/extracted_kernels/136-digit_hist_kernel.py`
- `exp_spec_decode_verify_scalar` — Compute （收益 300%）
  - 投机解码 block verify 基线，标量循环是瓶颈，优化方向为向量化验证与 mask 归约，代表 kernel。
  - 对应 kernel: `raw_experiences/extracted_kernels/171-rejection_random_sample_block_verify_kernel.py`
- `exp_topp_filter_001` — Compute （收益 35%）
  - Top-P filter，融合 softmax、cumsum、mask、renormalize，logits 先减 max 防上溢。
  - 对应 kernel: `raw_experiences/extracted_kernels/223-_top_p_filter_kernel.py`
- `exp_top_p_sampling_001` — Compute （收益 30%）
  - Top-P sampling，向量 softmax/cumsum + 向量搜索 + `strategy` constexpr。
  - 对应 kernel: `raw_experiences/extracted_kernels/222-_top_p_sample_kernel.py`
- `exp_mhc_post_fusion_001` — Compute （收益 35%）
  - MHC post fusion，合并 matmul + broadcast mul + add，HC 需 16/32 对齐。
  - 对应 kernel: `raw_experiences/extracted_kernels/075-_mhc_post_kernel.py`
- `exp_tril_solve_001` — Compute （收益 35%）
  - 下三角求解，16×16 tiling + 向量化 reduction + `cu_seqlens` 变长。
  - 对应 kernel: `raw_experiences/extracted_kernels/153-solve_tril_16x16_kernel.py`
- `exp_multinomial_001` — Compute （收益 35%）
  - 多项式采样，向量化二分搜索，`do_not_specialize` 减少变长编译。
  - 对应 kernel: `raw_experiences/extracted_kernels/139-multinomial_with_replacement.py`
- `exp_reject_sampler_001` — Compute （收益 35%）
  - 投机解码拒绝采样，向量加载 + 并行比较 + mask 归约替代 early-exit 标量循环。
  - 对应 kernel: `raw_experiences/extracted_kernels/224-_reject_sampler_kernel.py`
- `exp_grpo_loss_001` — Compute （收益 30%）
  - GRPO loss，LogSumExp 稳定，优化方向是把 new/ref/old 三个 reduction loop 融合为单遍。
  - 对应 kernel: `raw_experiences/extracted_kernels/209-grpo_kernel.py`
- `exp_fused_index_001` — Compute （收益 45%）
  - FP8 索引/相似度搜索，融合 GEMM + ReLU + Scale + Sum，Q 在 N 方向复用。
  - 对应 kernel: `raw_experiences/extracted_kernels/197-triton_index_kernel.py`
- `exp_sort_bitonic_001` — Compute （收益 25%）
  - Bitonic sort，用极值 padding 处理尾块，batch 并行处理多个小数组。
  - 对应 kernel: `raw_experiences/extracted_kernels/134-bitonic_sortbykey_kernel.py`
- `exp_radix_convert_001` — Compute （收益 20%）
  - Radix sort 预处理，bitwise MSB 翻转把有符号整数映射为可排序无符号顺序。
  - 对应 kernel: `raw_experiences/extracted_kernels/135-radix_type_convert.py`
- `exp_recompute_w_u_001` — Compute （收益 25%）
  - Attention 状态重计算，FP32 累加 + `allow_tf32=False` + 融合 gating scale + 变长路径。
  - 对应 kernel: `raw_experiences/extracted_kernels/156-recompute_w_u_fwd_kernel.py`
- `exp_hist_count_001` — Compute （收益 50%）
  - 索引频次统计，广播比较 + 寄存器局部累加替代全局原子加。
  - 对应 kernel: `raw_experiences/extracted_kernels/022-count_kernel.py`

## 关联知识

- 开发指南: [Misc 算子优化](../../reference/op-class-fused_ops.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [Elementwise 算子优化](../../reference/op-class-elementwise.md) — 主题相关：共享 910b2, 910b3, vectorization
- 对应指南: [归一化算子优化](../../reference/op-class-normalization.md) — 主题相关：共享 910b2, 910b3, numerical_stability
- 对应指南: [归约算子优化](../../reference/op-class-reduction.md) — 主题相关：共享 910b2, 910b3, numerical_stability
- 对应指南: [Softmax / LogSoftmax 算子优化](../../reference/op-class-softmax.md) — 主题相关：共享 910b2, 910b3, numerical_stability
- 相关主题: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 主题相关：共享 numerical_stability
- 相关主题: [统计归约必须用 fp32 累加器，禁止 fp16/bf16 直接累加](../algorithm-numerics/fp32-reduction-accumulator.md) — 主题相关：共享 numerical_stability
- 相关主题: [Online Softmax / Flash Attention 单遍稳定计算](../algorithm-numerics/online-softmax-flash-attention.md) — 主题相关：共享 numerical_stability
- 相关主题: [Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度](../algorithm-numerics/traj-attention-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [低精度数值稳定性：exp 最大偏移与 sqrt 非负保护](../algorithm-numerics/traj-general_stability-numerical_stability.md) — 主题相关：共享 910b2, 910b3, numerical_stability
- 相关主题: [Normalization Compute 优化聚合](../algorithm-numerics/traj-normalization-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion](../algorithm-numerics/traj-normalization-memory_access.md) — 主题相关：共享 910b2, 910b3, kernel_fusion
- 相关主题: [Gather/Scatter 内存访问优化聚合](../memory-sram/traj-gather_scatter-memory_access.md) — 主题相关：共享 910b2, 910b3, vectorization
- 相关主题: [Misc 算子内存访问优化聚合](../memory-sram/traj-misc-memory_access.md) — 主题相关：共享 910b2, 910b3, misc
- 相关主题: [Sum-and-Scale 归约内存访问优化](../memory-sram/traj-reduction-memory_access.md) — 主题相关：共享 910b2, 910b3, vectorization
- 相关主题: [动态序列长度场景使用 do_not_specialize 降低编译开销](../meta-skills/traj-misc-compilation_opt.md) — 主题相关：共享 910b2, 910b3, misc
- 相关主题: [SDA（Scaled Dot Attention）Ascend 硬件适配：HiVM 分块、多缓冲预取与 Cube/Vector 均衡](../portability/traj-misc-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, misc
- 相关主题: [SwiGLU 计算优化：门控融合、展平布局与激活量化](traj-activation-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Activation Scheduling 优化聚合](traj-activation-scheduling.md) — 主题相关：共享 910b2, 910b3, vectorization
- 相关主题: [Causal Conv1D Compute 优化聚合](traj-conv-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Scatter Weight Gradient 向量化计算优化](traj-gather_scatter-compute.md) — 主题相关：共享 910b2, 910b3, compute, vectorization
- 相关主题: [复杂算子拆分：降低编译与调度开销](traj-general_perf-op_split.md) — 主题相关：共享 910b2, 910b3, kernel_fusion
- 相关主题: [Loss 计算优化聚合](traj-loss-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Matmul Compute 优化聚合](traj-matmul-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Misc 算子 Scheduling 优化聚合](traj-misc-scheduling.md) — 主题相关：共享 910b2, 910b3, misc
- 相关主题: [MoE 稀疏 Weight Gradient 计算优化](traj-moe-compute.md) — 主题相关：共享 910b2, 910b3, compute, vectorization
- 相关主题: [Reduction Compute 优化聚合](traj-reduction-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [RoPE 计算优化：cos/sin 复用、向量化旋转与多模态位置编码融合](traj-rope-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Softmax 计算优化：Online Softmax、Two-Pass 向量化与数值稳定](traj-softmax-compute.md) — 主题相关：共享 910b2, 910b3, compute, numerical_stability
- 相关主题: [三角矩阵求解（solve_tril）计算优化：结构零元掩码消除与 constexpr 双路径](traj-solve_tril-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [SSM Compute 优化聚合](traj-ssm-compute.md) — 主题相关：共享 910b2, 910b3, compute

<!-- okf:related:end -->
