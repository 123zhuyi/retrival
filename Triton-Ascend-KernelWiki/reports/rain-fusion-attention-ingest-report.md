# rain_fusion_attention CANN->Triton-Ascend ingestion report

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| OPERATOR_NAME | `rain_fusion_attention` | source-confirmed |
| OPERATOR_TYPE | `attention` | inferred |
| commit | `d28d8f99a8858b71caa27a535da32bb5ba68ce2a` | source-confirmed |
| mode | static_ingest | - |

## 2. Source file inventory

~17 文件，无 arch 子目录（基于 CATLASS 模板库）。核心：`README.md`(193L)、`op_host/rain_fusion_attention_def.cpp`(84L)、`op_host/rain_fusion_attention_tiling.cpp`(1014L)、`op_kernel/rain_fusion_attention.cpp`(78L,6 tiling key)、`op_kernel/rain_fusion_attention_tilingkey.h`(48L)、`op_kernel/rain_fusion_attention_kernel.h`(589L)。完整清单见 `artifacts/rain-fusion-attention/source_file_inventory.md`。

## 3. Operator semantics

- **功能**：CATLASS 风格块级稀疏推理注意力。selectIdx 驱动显式 Q×KV 块选择矩阵（READEME L3-14）。
- **公式**：标准 FlashAttention 在线 softmax + PV rescale，但 KV 块由 selectIdx 驱动而非连续遍历。
- **输入**：query/key/value/selectIdx/selectNumIdx/blockShape（REQUIRED）+ attenMask/actualSeqLengths/_kv/blockTable（OPTIONAL）。
- **输出**：attentionOut + softmaxLse(可选)。
- **属性**：qInputLayout/kvInputLayout("TND")/numKeyValueHeads(1)/maskType(0)/scaleValue(0.0)/innerPrecise(1)/blockSize(128)。
- **架构**：仅 Ascend910B1/910_93（A2/A3）。MIX_AIC_1_2。
- **约束**：仅 TND（但源码有 BNSD 分支，以源码为准）。

## 4. Pipeline

标准 FlashAttention 4 阶段（QK→Vec1→PV→Vec2），KV 块由 selectIdx 驱动。

## 5. Tiling

6 个十进制 tiling key（base=9000000000000000），编码 dtype×layout×softmax_precision。MIX_AIC_1_2。

## 6. Migration

transferable：selectIdx 驱动稀疏遍历、online softmax、LSE 输出、FP32/Half SM 精度选择。rewrite：tiling key→if constexpr。CANN-specific：CATLASS 模板库、MIX_AIC_1_2。

## 7. Validation

```
validate.py        -> OK
query rain fusion attention -> #1 hit kernel ✓
query catlass                -> 2 hits (source+kernel) ✓
```

## 8. Created files

`sources/ops-transformer/rain-fusion-attention.md`、`wiki/kernels/rain-fusion-attention.md`、`artifacts/rain-fusion-attention/`(6 文件)、`reports/rain-fusion-attention-ingest-report.md`。