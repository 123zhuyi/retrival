# rain_fusion_attention - tiling 与 dispatch 分析

## A. 架构分发

`def.cpp` L76-77：仅 `ascend910b` + `ascend910_93`（A2/A3）。README L193：编译 `--soc=Ascend910B3`。不支持 950/Kirin/310P。

## B. tiling key 编码（tilingkey.h L17-26）

十进制组装（与 FA arch32 tilingkey 风格一致），base=`9000000000000000`：

| 位域 | 位置 | 取值 |
|---|---|---|
| Q Layout | [0-1] | 2=TND, 3=BNSD |
| Mask Type | [2-4] | 0=NoMask, 3=Causal |
| Softmax Precision | [5-7] | 0=Float, 1=Half |
| PagedCache | [8-10] | 0=NoCache, 1=WithCache |
| KV Layout | [11-13] | 00=TND, 20=BNSD |
| Data Type | [14-15] | 00=FP16, 22=BF16 |
| Operator Category | [16-18] | 900=RFA |

6 个预定义 key：

| tiling key | dtype | Q layout | KV layout | SM precision | 源码 |
|---|---|---|---|---|---|
| `9000000030000002` | FP16 | TND | TND | Float | L31 |
| `9000000030100002` | FP16 | TND | TND | Half | L34 |
| `9000000030022222` | BF16 | TND | TND | Float | L37 |
| `9000000050000003` | FP16 | BNSD | BNSD | Float | L40 |
| `9000000050100003` | FP16 | BNSD | BNSD | Half | L43 |
| `9000000050022223` | BF16 | BNSD | BNSD | Float | L46 |

## C. 任务类型

`KERNEL_TYPE_MIX_AIC_1_2`（kernel 入口 L34）：AIC:AIV=1:2。

## D. 核心计算（selectIdx 驱动稀疏注意力）

```
for each Q block (按 blockShapeX 切分):
  for each selected KV block (selectIdx/selectNumIdx 驱动):
    Q_block · K_block^T × scale -> online softmax -> PV -> rescale -> 写回
  softmaxLse 可选输出
```

与 [[kernel-sparse-flash-attention]] 的稀疏 S2 遍历思路一致，但 SFA 的 sparse_indices 是 block 级索引，RFA 的 selectIdx 是显式 Q×KV 块选择矩阵（[T, headNum, maxKvBlockNum]），更灵活。

## E. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| selectIdx / selectNumIdx | `tl.load(select_idx)` 驱动循环 | 块稀疏索引 |
| blockShape [X, Y] | `tl.constexpr BLOCK_M/N` | 稀疏块大小 |
| 6 tiling key 分支 | `if constexpr` / launcher | dtype×layout×precision |
| online softmax | running max/sum + rescale | 可迁移 |
| softmaxLse 输出 | `log(l_i) + m_i` | 推理 LSE |
| MIX_AIC_1_2 | 不迁移（CANN 配比） | CANN-specific |
| CATLASS 模板库 | 不迁移（CANN 框架） | CANN-specific |