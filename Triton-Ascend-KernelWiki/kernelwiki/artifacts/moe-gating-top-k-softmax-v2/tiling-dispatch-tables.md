# MoeGatingTopKSoftmaxV2 - Tiling / 核切分 / 模板 dispatch (S4 + S5)

## A. 算子语义

MoE gating 的 topk + softmax（README.md）：

- **renorm=0**：`softmaxResult = softmax(x, axis=-1)`；`yOut, expertIdxOut = topk(softmaxResult, k)`
- **renorm=1**：`topkOut, expertIdxOut = topk(x, k)`；`yOut = softmax(topkOut, axis=-1)`

若某行 finished=True，expertIdxOut 直接填 num_expert（x 最后一轴大小）。

## B. 输入输出（def.cpp）

- **输入**：x（FP16/FLOAT/BF16）、finished（BOOL，可选）
- **输出**：y（FP16/FLOAT/BF16）、expert_idx（INT32）、softmax_result（FLOAT，可选）
- **属性**：k（必需）、renorm(0)、output_softmax_result_flag(false)

## C. 架构

ascend910b/910_93（membase）+ ascend950（regbase）

## D. 模板 dispatch（kernel L68-176）

`KERNEL_TYPE_MIX_AIV_1_0`（AIV-only，g_coreType==AIC return）。~36 tiling key：

- **EKFullLoad**（101xxx）：expert+K 全加载，dtype(3) × bufferNum(1/2) × renorm(0/1)
- **KFullLoad**（102xxx，renorm=0）：K 全加载
- **KRenorm**（102xxx，renorm=1）：K renorm
- **Perf**（103xxx）：高性能，dtype(3) × renorm(0/1) × colRange(3: `<8`/`8-64`/`>64`)

## E. 约束（README.md）

- k <= 1024
- renorm 只支持 0/1
- x 和 finished 每一维 <= INT32_MAX

## F. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 |
|---|---|
| AIV-only | 纯 softmax + topk |
| softmax | `tl.softmax` 跨 expert 轴 |
| topk | `tl.topk(x, k)` |
| finished 标记 | mask 替换 |
| colRange 分档 | 按 expert 数选 BLOCK |