# moe_compute_expert_tokens CANN->Triton-Ascend ingestion report

## 1. Task parameters
| Parameter | Value | Status |
|---|---|---|
| OPERATOR_NAME | `moe_compute_expert_tokens` | source-confirmed |
| OPERATOR_TYPE | `moe` | source-confirmed |
| commit | `d28d8f9...` | source-confirmed |
| mode | static_ingest | - |

## 2. Source file inventory
~12 文件，无 arch 子目录。核心：`README.md`(72L)、`op_host/moe_compute_expert_tokens_def.cpp`(53L)、`op_kernel/moe_compute_expert_tokens.cpp`(44L,3 tiling key)、`op_kernel/moe_compute_expert_tokens_int32_{ss,m,l}.h`(261/452/625L)。完整清单见 `artifacts/moe-compute-expert-tokens/source_file_inventory.md`。

## 3. Operator semantics
- **功能**：MoE 通过二分查找找每个专家处理的最后一行位置，输出累积 token 数。
- **公式**：`out[i]=BinarySearch(sortedExperts, numExperts)`。
- **输入**：sorted_experts(INT32)。输出：total_rows_before_expert(INT32)。属性 num_experts。
- **架构**：A2/A3+950+Kirin（全架构统一 config）。AIV-only（MIX_AIV_1_0）。
- **tiling key**：3 个（1001/1002/1003）= SS/M/L 按数据量。

## 4. Pipeline
遍历专家 -> 二分查找边界 -> 输出累积 token 数。

## 5. Validation
```
validate.py -> OK
query moe compute expert tokens -> #1 hit kernel ✓
```

## 6. Created files
`sources/ops-transformer/moe-compute-expert-tokens.md`、`wiki/kernels/moe-compute-expert-tokens.md`、`artifacts/moe-compute-expert-tokens/`(6 文件)、`reports/moe-compute-expert-tokens-ingest-report.md`。

## 7. Note
moe 类 kernel 页达 **13 个**。MoE pipeline 辅助算子（统计专家 token 数）。