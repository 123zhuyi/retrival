# ring_attention_update - 流水编排分析

## 1. 核心公式（在线 softmax 合并）

```
softmax_max = max(prev_max, cur_max)
alpha_prev = exp(prev_max - softmax_max)
alpha_cur = exp(cur_max - softmax_max)
softmax_sum = prev_sum * alpha_prev + alpha_cur
attn_out = prev_out * alpha_prev / softmax_sum + cur_out * alpha_cur / softmax_sum
```
（README L18-28）

这与 FlashAttention 的 FlashSoftmax rescale 逻辑完全一致（`exp[i]=e^(max[i-1]-max[i])`），只是把"上一步 vs 当前步"改为"第一次 vs 第二次 FA 输出"。**本算子只做 rescale 合并，不做 QK/PV。**

## 2. 流水阶段

| 阶段 | 操作 | 说明 |
|---|---|---|
| 1. 读 prev/cur | 加载 prev_attn_out/prev_max/prev_sum + cur_attn_out/cur_max/cur_sum | 按 layout stride |
| 2. 新 max | `max = max(prev_max, cur_max)` | 元素级 |
| 3. alpha | `exp(prev_max-max)` + `exp(cur_max-max)` | 元素级 |
| 4. 新 sum | `prev_sum * alpha_prev + alpha_cur` | 元素级 |
| 5. 输出 | `prev_out * alpha_prev / sum + cur_out * alpha_cur / sum` | 元素级 |

## 3. 阶段 -> Triton 抽象

| 阶段 | CANN 机制 | Triton 抽象 | 可迁移 |
|---|---|---|---|
| rescale 公式 | 元素级 FP32 计算 | `tl.where`+`tl.exp`+标量乘除 | 是（数学） |
| SBH/TND layout | 按 stride 寻址 | `tl.load` stride 参数 | 是 |
| 6 tiling key 分支 | TILING_KEY_IS | `if constexpr`/launcher | 是（分支逻辑） |
| 950 regbase | 独立 kernel | launcher 选 kernel | 是（架构分支） |