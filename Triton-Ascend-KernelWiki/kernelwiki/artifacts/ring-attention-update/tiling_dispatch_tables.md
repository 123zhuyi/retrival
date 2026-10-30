# ring_attention_update - tiling 与 dispatch 分析

## A. 架构分发

`def.cpp` L81-82：`ascend910b` + `ascend910_93`（arch32 kernel `ring_attention_update.cpp`）。L84-92：`ascend950`（arch35 kernel `ring_attention_update_apt.cpp`，opFile=ring_attention_update_apt）。L94-96：`kirinx90` + `kirin9030`（arch32 kernel，仅 FP16/FLOAT）。

## B. tiling key

arch32 kernel（`ring_attention_update.cpp` L32-84）：

| tiling key | dtype | layout | 类 |
|---|---|---|---|
| 0 | FP16 | SBH | `KernelRingAttentionUpdate<half>` |
| 1 | BF16 | SBH | `KernelRingAttentionUpdate<bfloat16_t>`（310P 排除） |
| 2 | FLOAT | SBH | `KernelRingAttentionUpdate<float>` |
| 10 | FP16 | TND | `KernelRingAttentionUpdateTND<half>` |
| 11 | BF16 | TND | `KernelRingAttentionUpdateTND<bfloat16_t>`（310P 排除） |
| 12 | FLOAT | TND | `KernelRingAttentionUpdateTND<float>` |

arch35 kernel（`ring_attention_update_apt.cpp` L22-24）：

| tiling key | layout | 类 |
|---|---|---|
| 100 | SBH | `KernelRingAttentionUpdateRegbaseSBH` |
| 110 | SoftmaxTND | `KernelRingAttentionUpdateRegbaseSoftmaxTND` |
| 120 | TND | `KernelRingAttentionUpdateRegbaseTND` |

## C. 核心计算（在线 softmax rescale）

公式（README L18-28）：
```
softmax_max = max(prev_max, cur_max)
softmax_sum = prev_sum * exp(prev_max - max) + exp(cur_max - max)
attn_out = prev_out * exp(prev_max - max) / sum + cur_out * exp(cur_max - max) / sum
```

## D. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| tiling key (0/1/2/10/11/12) | `if constexpr` / launcher | dtype×layout 分支 |
| rescale 公式 | `tl.where` + `tl.exp` + 标量乘 | 在线 softmax 合并 |
| prev_out * exp(prev_max-max) / sum | 元素级乘加 | 可迁移 |
| cur_out * exp(cur_max-max) / sum | 元素级乘加 | 可迁移 |
| SBH/TND layout | `tl.load` stride 参数 | layout 抽象 |
| arch32 vs arch35 | launcher 选 kernel | 架构分支 |