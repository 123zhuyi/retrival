# moe_gating_top_k_softmax CANN->Triton-Ascend ingestion report

## 1. Task parameters
| Parameter | Value | Status |
|---|---|---|
| OPERATOR_NAME | `moe_gating_top_k_softmax` | source-confirmed |
| OPERATOR_TYPE | `moe` | source-confirmed |
| commit | `d28d8f9...` | source-confirmed |
| mode | static_ingest | - |

## 2. Source file inventory
~20 文件，arch32+arch35 双架构。核心：`README.md`(102L)、`op_host/moe_gating_top_k_softmax_def.cpp`(63L)、`op_kernel/moe_gating_top_k_softmax.cpp`(113L,18 tiling key)+`_apt.cpp`(61L,950)、`op_kernel/moe_gating_top_k_softmax_{e_k_fullload,k_fullload,perf}.h`(473/463/827L)、`op_kernel/arch35/moe_gating_top_k_softmax_fullload_generalized_regbase.h`(533L)。完整清单见 `artifacts/moe-gating-top-k-softmax/source_file_inventory.md`。

## 3. Operator semantics
- **功能**：MoE gating softmax + topk。`softmax(x)` -> `topK` -> 输出 y/expertIdx/rowIdx。
- **公式**：`softmaxOut=softmax(x,axis=-1)` -> `yOut,expertIdxOut=topK(softmaxOut,k)` -> `rowIdxOut=arange(N*K).reshape(K,N).T`。finished=True 时 expertIdx 填 num_expert。
- **输入**：x(FP16/FP32/BF16)/finished(OPTIONAL,BOOL)。输出：y/expert_idx/row_idx(INT32)。属性 k。
- **架构**：A2/A3 + 950(arch35)。AIV-only。
- **tiling key**：18 个，3 模板族（E_K_FullLoad 0-5 / K_FullLoad 6-8 / Perf 9-17 按 ColRange <8/8-64/>64）。

## 4. Pipeline
4 阶段：Softmax(axis=-1) -> TopK -> RowIdx(arange+reshape+transpose) -> Finished 条件填充。

## 5. MoE pipeline 位置
入口算子：gating_top_k_softmax -> init_routing_quant -> FFN -> finalize_routing。

## 6. Validation
```
validate.py -> OK
query moe gating topk softmax -> #2 hit kernel ✓
```

## 7. Created files
`sources/ops-transformer/moe-gating-top-k-softmax.md`、`wiki/kernels/moe-gating-top-k-softmax.md`、`artifacts/moe-gating-top-k-softmax/`(6 文件)、`reports/moe-gating-top-k-softmax-ingest-report.md`。

## 8. Note
moe 类 kernel 页达 **12 个**。gating 系列 V1+V2 配对。