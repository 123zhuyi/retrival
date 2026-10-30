---
id: ops-transformer-rain-fusion-attention
title: CANN ops-transformer RainFusionAttention source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: attention/rain_fusion_attention/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-29
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer RainFusionAttention 来源摘要

固定提交 `d28d8f9...`。基于 CATLASS 模板库的块级稀疏推理注意力。仅 A2/A3（Ascend910B）。

## 核心源文件 SHA-256

| 源文件 | SHA-256 |
|---|---|
| `README.md` | `e698daceeddf158e0471ced2ee199751fb8526f6d782392d7c7ac3a317d3d624` |
| `op_host/rain_fusion_attention_def.cpp` | `d8e916c4b897ac138b79d0c63a4f4857399da12843ec3db51e022483cd0c3751` |
| `op_kernel/rain_fusion_attention.cpp` | `e692caebee2877b08818958c4b9adb87764cca124b35a4c9e62a1463810deedf` |
| `op_kernel/rain_fusion_attention_tilingkey.h` | `cad683fd297a2bffc24f14d390346429d16a6f846778ab157b8108d8471bedec` |
| `op_kernel/rain_fusion_attention_kernel.h` | `be6c4e8c3a4249124d8f9b457eca7434232736f130853772ce7370c286b38dbd` |

## 关键代码区域

- **功能**（README L3-14）：基于 CATLASS 的块级稀疏注意力，`selectIdx`/`selectNumIdx` 驱动。
- **算子定义**（def.cpp L19-77）：输入 query/key/value/selectIdx/selectNumIdx/blockShape（REQUIRED）+ attenMask/actualSeqLengths/actualSeqLengthsKv/blockTable（OPTIONAL）。输出 attentionOut + softmaxLse(可选)。属性 qInputLayout/kvInputLayout("TND")/numKeyValueHeads(1)/maskType(0)/scaleValue(0.0)/innerPrecise(1)/blockSize(128)。AICore config：ascend910b/910_93。
- **tiling key**（tilingkey.h L17-46）：十进制组装(base=9000000000000000)，6 个 key（TND/BNSD × FP16/BF16 × FloatSM/HalfSM）。位域编码 layout+dtype+precision+mask+cache。
- **kernel dispatch**（`rain_fusion_attention.cpp` L32-75）：MIX_AIC_1_2。6 tiling key 分支，模板参数 `<dtype, softmax_dtype, LseMode, qLayoutFlag, kvLayoutFlag>`。
- **约束**（README L133-181）：仅 TND（但源码有 BNSD 分支）；selectIdx [T,headNum,maxKvBlockNum]；blockShapeX/Y；非对齐场景自动处理。

## 衍生引用

- `wiki/kernels/rain-fusion-attention.md`、`artifacts/rain-fusion-attention/`