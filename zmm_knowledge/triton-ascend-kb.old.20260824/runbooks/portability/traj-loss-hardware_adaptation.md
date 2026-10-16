---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: loss
dimension: "4.1"
title: Entropy Backward 在 Ascend NPU 上的硬件适配优化
description: 针对 Ascend NPU 的 entropy backward split-N 实现，通过 int32 比较避免标量退化、grouped scheduling 提升 L2 命中率、大 K tiling 打满 Cube，提升 25%。
tags: [loss, hardware_adaptation, ascend_npu, int32_cast, grouped_scheduling, large_k_tiling, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/loss/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# Entropy Backward 在 Ascend NPU 上的硬件适配优化

## 概述

在大词表语言模型训练中，entropy loss backward 常常按 vocab 维度切分到多卡（split-N），每张卡只计算自己负责的 vocab 片段的 `d_logits`。Ascend NPU 在某些操作上与 GPU 有显著差异：例如 `int64` 相等比较会退化为标量执行，严重拖慢性能。该经验针对 NPU 微架构做了三项关键适配：

- **int32 标签比较**：将 `labels` 与 `offs_bn + rank * vocab_size` 的比较强制转为 `int32`，避免 NPU int64 标量退化；
- **grouped scheduling**：`GROUP_SIZE_M=4` 让连续 program 复用同一批 token 的 `hidden`、`maximum`、`accu`，提升 L2 命中率；
- **大 K tiling**：`BLOCK_SIZE_K=512` 使 Cube 单元有足够的数据量做流水线，最大化矩阵乘吞吐。

代价是 `d_logits` 需要按 split 写回，且 `BLOCK_SIZE_K` 增大会提高寄存器压力；当 `hidden_size < 512` 时需要调整 tile 大小。

## 共同模式

### 1. int32 相等比较规避 NPU 标量路径

在构建 one-hot mask 时，代码需要判断 `offs_bn + rank * vocab_size` 是否等于 `labels`。NPU 上对 `int64` 张量做相等比较会退化为逐元素标量比较。通过 `.to(tl.int32)` 把两边都转成 32 位整数，编译器可以生成向量比较指令，性能显著提升。

### 2. grouped scheduling 提高 L2 复用

`GROUP_SIZE_M=4` 表示同一组 program 先处理 4 个连续的 M（token）tile，再依次遍历 N（vocab）tile。这样 `hidden`、`maximum`、`accu`、`d_entropy` 等 M 维数据被多个 N tile 共享，减少了对 HBM 的重复访问。

### 3. 大 K tiling 匹配 Cube 流水线

`BLOCK_SIZE_K=512` 远大于常见的 128/256。对 entropy backward 中的 `tl.dot(hidden, weight.trans())` 来说，更大的 K tile 意味着 Cube 单元在每个 dot 操作中有更多的数据可供流水，更容易接近峰值算力。需要保证 `hidden_size ≥ 512` 才能发挥效果。

### 4. split-N 边界处理

通过 `split_idx` 和 `vocab_per_split` 计算当前 tile 在全局 vocab 中的位置，并用 `vocab_right_bound = min((split_idx + 1) * vocab_per_split, vocab_size)` 限制 weight load 的边界，避免跨 split 读取无效数据。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_entropy_backward_npu_001` | 25% | Entropy backward split-N 在 Ascend NPU 上的适配：int32 比较 + grouped scheduling + 大 K tiling，提升 25%。 |

## 代表 Kernel

以下为 `exp_entropy_backward_npu_001` 对应的 kernel：

```python
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 512,
                "GROUP_SIZE_M": 4,
            },
        ),
    ],
    key=["num_tokens", "hidden_size", "vocab_size"],
)
@triton.jit
def efficient_entropy_backward_kernel_general_d_logits_split_N(
    split_idx: int,
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    vocab_per_split: int,
    rank: int,
    hidden_ptr,
    stride_hidden_m: tl.int64,
    stride_hidden_k: tl.int64,
    weight_ptr,
    stride_weight_n: tl.int64,
    stride_weight_k: tl.int64,
    labels_ptr,
    stride_labels: tl.int64,
    maximum_ptr,
    stride_maximum: tl.int64,
    accu_ptr,
    stride_accu: tl.int64,
    d_entropy_ptr,
    stride_d_entropy: tl.int64,
    d_logprobs_ptr,
    stride_d_logprobs: tl.int64,
    reduction: int,
    entropy_b_ptr,
    stride_entropy_b,
    d_logits_ptr,
    stride_d_logits_m: tl.int64,
    stride_d_logits_n: tl.int64,
    rcp_temperature: tl.float32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(vocab_per_split, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = (
        split_idx * vocab_per_split + pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    )
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    maximum = tl.load(
        maximum_ptr + offs_am * stride_maximum, mask=offs_am < num_tokens, other=0.0
    )
    accu = tl.load(
        accu_ptr + offs_am * stride_accu, mask=offs_am < num_tokens, other=1e-6
    )
    accu_rcp = tl.fdiv(1.0, accu)
    d_entropy = tl.load(
        d_entropy_ptr + offs_am * stride_d_entropy, mask=offs_am < num_tokens, other=0.0
    )
    if reduction == 0:
        d_logprobs = tl.load(
            d_logprobs_ptr + offs_am * stride_d_logprobs,
            mask=offs_am < num_tokens,
            other=0.0,
        )
    elif reduction == 1:
        d_logprobs = tl.load(d_logprobs_ptr)
        d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
    else:
        d_logprobs = tl.fdiv(tl.load(d_logprobs_ptr), num_tokens.to(tl.float32))
        d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
    d_logprobs = -1 * d_logprobs
    entropy_b = tl.load(
        entropy_b_ptr + offs_am * stride_entropy_b, mask=offs_am < num_tokens, other=0.0
    )
    labels = tl.load(
        labels_ptr + offs_am * stride_labels, mask=offs_am < num_tokens, other=0
    ).to(tl.int32)

    hidden_ptrs = hidden_ptr + (
        offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k
    )
    weight_ptrs = weight_ptr + (
        offs_bn[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k
    )

    vocab_right_bound = min((split_idx + 1) * vocab_per_split, vocab_size)
    logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
        _hidden = tl.load(
            hidden_ptrs,
            mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K)
            & (offs_am[:, None] < num_tokens),
            other=0.0,
        )
        _weight = tl.load(
            weight_ptrs,
            mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K)
            & (offs_bn[:, None] < vocab_right_bound),
            other=0.0,
        )
        logits = tl.dot(_hidden, _weight.trans(), logits)

        hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
        weight_ptrs += BLOCK_SIZE_K * stride_weight_k

    logits *= rcp_temperature
    exp_logits = tl.exp(logits - maximum[:, None])

    # In NPU, when performing equality checks, if the operands are of type int64, the operation degrades to
    # a scalar operation. Therefore, during equality comparison, if the data can be converted to int32 without overflow,
    # it can be converted to int32 to improve performance.
    mask = ((offs_bn + rank * vocab_size)[None, :]).to(tl.int32) == labels[:, None]
    d_logits = d_logprobs[:, None] * (exp_logits * accu_rcp[:, None] - mask)
    d_logits += (
        d_entropy[:, None]
        * (-exp_logits * accu_rcp[:, None])
        * (logits - entropy_b[:, None])
    )

    d_logits *= rcp_temperature

    # filter d_logits with mask
    result_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_am[:, None] < num_tokens) & (result_offs_n[None, :] < vocab_per_split)

    tl.store(
        d_logits_ptr
        + offs_am[:, None] * stride_d_logits_m
        + result_offs_n[None, :] * stride_d_logits_n,
        d_logits,
        mask,
    )
```

## 已知实例

- `exp_entropy_backward_npu_001` — Hardware Adaptation Optimization（收益 25%）
  - 适用于 Ascend 910B 系列上的 entropy backward split-N 场景，关键参数 `num_tokens ≥ 128`、`vocab_per_split ≥ 64`、`hidden_size ≥ 512`。
  - 对应 kernel: `raw_experiences/extracted_kernels/205-efficient_entropy_backward_kernel_general_d_logits_split_N.py`

## 关联知识

- 开发指南: [Loss 算子优化](../../reference/op-class-loss.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [Loss 函数算子优化](../../reference/op-class-loss.md) — 主题相关：共享 910b2, 910b3, loss
- 相关主题: [Loss 计算优化聚合](../compute-pipeline/traj-loss-compute.md) — 主题相关：共享 910b2, 910b3, loss
- 相关主题: [大词表 Entropy Loss Backward 调度融合优化](../compute-pipeline/traj-loss-scheduling.md) — 主题相关：共享 910b2, 910b3, loss
- 相关主题: [Attention 硬件适配优化聚合](traj-attention-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, hardware_adaptation
- 相关主题: [Ascend 硬件资源适配规则聚合](traj-general_ascend-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, hardware_adaptation
- 相关主题: [Matmul 硬件适配优化聚合](traj-matmul-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, hardware_adaptation
- 相关主题: [SDA（Scaled Dot Attention）Ascend 硬件适配：HiVM 分块、多缓冲预取与 Cube/Vector 均衡](traj-misc-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, ascend_npu, hardware_adaptation
- 相关主题: [Llama4 RoPE 在 Ascend NPU 上的硬件适配：UB Tiling、FMA 与交错复数布局](traj-rope-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, ascend_npu, hardware_adaptation

<!-- okf:related:end -->
