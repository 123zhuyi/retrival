# fused_floyd_attention_grad - tiling 与 dispatch 分析

来源：`attention/fused_floyd_attention_grad/`（commit `d28d8f9…`）。Floyd Attention Grad 是 Floyd Attention 的反向，双流（K1/V1+K2/V2），6 个 matmul 反向，仅 A2/A3 训练。

## A. 架构分发

`def.cpp` L164-165 注册 AICore config：

| 架构 | SocVersion | config | kernel 入口 |
|---|---|---|---|
| Atlas A2 训练 | `ascend910b` L164 | `fused_floyd_attention_grad` | `fused_floyd_attention_grad.cpp` |
| Atlas A3 训练 | `ascend910_93` L165 | 同上 | 同上 |

> 仅训练场景（A2/A3 训练），无 950，无 310P。支持 FP16/BF16/FLOAT。

## B. TilingKey 与 dispatch（kernel.cpp L79-122）

kernel 按 `ORIG_DTYPE_QUERY` 和 `TILING_KEY_IS` 宏分发：

| dtype | TilingKey | 特性 |
|---|---|---|
| BF16 | 10000000000000022434UL | mask=0, pse=0, drop=0, mm_out=nd |
| BF16 | 10000000000100022434UL | mask=1, pse=0, drop=0, mm_out=nd |
| FP16 | 10000000000000023434UL | mask=0, pse=0, drop=0, mm_out=nd |
| FP16 | 10000000000100023434UL | mask=1, pse=0, drop=0, mm_out=nd |

## C. 核间切分

| 项 | 结论 | 来源 |
|---|---|---|
| 逻辑轴 | B、N2、G、S1、S2（Floyd 双流用 seq 作为 batch 轴） | README L17 |
| 切分方式 | S1S2 BN2GS1S2 模板 | `tiling_s1s2_bn2gs1s2.cpp` |
| 三段式 | Pre → SyncAll → S1S2 → SyncAll → Post | kernel.cpp L41-76 |

## D. 核内处理（6 个 matmul 反向）

| 阶段 | 操作 | 说明 |
|---|---|---|
| Pre | dS = softmax * dY | 下游梯度 * softmax 输出 |
| S1S2 Bmm1 | dV1 = P^T · dY | Value1 梯度 |
| S1S2 Bmm2 | dV2 = P^T · dY | Value2 梯度 |
| S1S2 MM1 | dQ = (dS·K1 + dS·K2) / √d | Query 梯度（双流相加） |
| S1S2 MM2 | dK1 = (dS^T · Q) / √d | Key1 梯度 |
| S1S2 MM3 | dK2 = (dS^T · Q) / √d | Key2 梯度 |
| Post | 梯度搬运/格式转换 | GM 输出 |

## E. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| 6 matmul 反向 | `tl.dot` × 6 | 标准 FA 反向梯度 |
| Pre（dS = softmax * dY） | `dS = softmax * dY` | elemwise |
| dV1/dV2 | `tl.dot(P.T, dY)` | Value 梯度 |
| dQ | `(tl.dot(dS,K1) + tl.dot(dS,K2)) / sqrt(d)` | Query 梯度（双流） |
| dK1/dK2 | `tl.dot(dS.T, Q) / sqrt(d)` | Key 梯度 |
| 三段式 | 单 kernel 或 3 kernel launch | 不迁移手动 SyncAll |
| TilingKey | `if constexpr` | 不迁移 key 本身 |