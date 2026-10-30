# moe_init_routing_quant CANN->Triton-Ascend ingestion report

## 1. Task parameters
| Parameter | Value | Status |
|---|---|---|
| OPERATOR_NAME | `moe_init_routing_quant` | source-confirmed |
| OPERATOR_TYPE | `moe` | source-confirmed |
| commit | `d28d8f9...` | source-confirmed |
| mode | static_ingest | - |

## 2. Source file inventory
~22 文件，无 arch 子目录。核心：`README.md`(117L)、`op_host/moe_init_routing_quant_def.cpp`(65L)、`op_kernel/moe_init_routing_quant.cpp`(81L)、`op_kernel/moe_quant_{common,sort_multi_core,sort_one_core,mrgsort,mrgsort_out,src_to_dst_op,gather_out,gather_out_small_activate_row}.h`。

## 3. Operator semantics
- **功能**：MoE routing + 量化。3 阶段：Sort -> SrcToDst -> GatherOut+Quant（INT8）。
- **输入**：x(FP16/BF16/FP32)/row_idx/expert_idx(INT32)。输出：expanded_x(INT8)/expanded_row_idx/expanded_expert_idx(INT32)。属性 active_num/scale/offset。
- **架构**：仅 A2/A3。AIV-only（`g_coreType==AIC return`）。
- **tiling key**：4 个（1/2 normal + 3/4 small_active_row × 1core/multicore）。

## 4. Pipeline
Sort(keyValueSort) -> SrcToDst(反向映射 expandedRowIdx[sortedRowIdx[i]]=i) -> GatherOut+Quant(gather x + round(x·scale+offset) cast INT8)。

## 5. Validation
```
validate.py -> OK
query moe init routing quant -> #1 hit kernel ✓
```

## 6. Created files
`sources/ops-transformer/moe-init-routing-quant.md`、`wiki/kernels/moe-init-routing-quant.md`、`artifacts/moe-init-routing-quant/`(6 文件)、`reports/moe-init-routing-quant-ingest-report.md`。