# MoeFinalizeRoutingV2 — Tiling / 核切分 / 模板 dispatch (S4 + S5)

## A. 算子语义

MoE 计算中最后合并 FFN 输出结果（README.md L18-27）：

```
expertid = expertIdx[i,k]
out(i,j) = x1(i,j) + x2Optional(i,j) + Σ_k (scales(i,k) * (expandedX[expandedRowIdx(i+k*num_rows), j] + bias(expertid, j)))
```

## B. Tiling 参数

TilingData 字段（`tiling.h` L77-103）：
- `totalCoreNum/usedCoreNum` — 总核/使用核数
- `H` — expandedX 列数（head dim），`normalH/unnormalH/hSliceNum` — H 切分
- `K` — scales 列数（topk 数），`normalK/unnormalK/kSliceNum` — K 切分
- `skip2IsNull/biasRowNum/totalRowNum/skip1IsNull/scalesIsNull/dropPadMode` — 可选输入标记
- `normalCoreHandleNum/tailCoreHandleNum/...` — 核循环控制

## C. 模板 dispatch（kernel L92-298）

~50 个 tiling key，按 dtype × 算法策略 × 是否有 bias 区分：
- **dtype**：DT_FLOAT/DT_FLOAT16/DT_BF16（+ mix_precision）
- **算法策略**：BIG_K_V2（cutK）、DB_V2（double buffer）、CUTH_K2_V2（cut H K=2）、CUTH_K4_V2（cut H K=4）、CUTH_V2（cut H）、DB_ALL_BIAS_V2（double buffer all bias）
- **bias**：有/无（WITHOUT_BIAS 后缀）
- `KERNEL_TYPE_AIV_ONLY`（纯 Vector，无 Cube）

## D. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 |
|---|---|
| AIV-only | 纯 elementwise gather + reduce |
| expandedRowIdx gather | `tl.load(expandedX + expandedRowIdx * stride)` |
| scales * (gathered + bias) | `* scales + tl.load(bias)` |
| 跨 K 累加 | `tl.sum` 或循环累加 |
| H 轴切分 | grid 1D 映射 |