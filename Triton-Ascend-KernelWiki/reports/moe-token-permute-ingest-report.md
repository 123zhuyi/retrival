# moe_token_permute CANN->Triton-Ascend ingestion report

## 1. Task parameters
| Parameter | Value | Status |
|---|---|---|
| OPERATOR_NAME | `moe_token_permute` | source-confirmed |
| OPERATOR_TYPE | `moe` | source-confirmed |
| commit | `d28d8f9...` | source-confirmed |
| mode | static_ingest | - |

## 2. Source file inventory
~17 文件，无 arch 子目录。核心：`README.md`(118L)、`op_host/moe_token_permute_def.cpp`(57L)、`op_kernel/moe_token_permute.cpp`(82L)、`op_kernel/moe_{sort_base,sort_one_core,sort_multi_core,mrgsort,mrgsort_out,index_copy,index_copy_spilt_d}.h`。

## 3. Operator semantics
- **功能**：MoE token 重排，根据 indices 将 tokens 广播并排序。
- **公式**（paddedMode=false）：`sortedIndicesFirst=argSort(indices)` -> `sortedIndicesOut=argSort(sortedIndicesFirst)` -> `permuteTokens[sortedIndicesFirst[i]]=tokens[i//topK]`。
- **输入**：tokens(FP16/BF16/FP32/INT8)/indices(INT32/INT64)。输出：permute_tokens/sorted_indices(INT32)。属性 num_out_tokens/padded_mode。
- **架构**：仅 A2/A3。AIV-only。950 转调 moe_init_routing_v2。
- **tiling key**：8 个（1-8）= one/multicore × normal/spilt_d × padded_false/true。

## 4. Pipeline
3 阶段：Sort1(argSort indices) -> Sort2(argSort sortedFirst) -> IndexCopy(permuteTokens[sortedFirst[i]]=tokens[i//topK])。

## 5. Validation
```
validate.py -> OK
query moe token permute -> #1 hit kernel ✓
```

## 6. Created files
`sources/ops-transformer/moe-token-permute.md`、`wiki/kernels/moe-token-permute.md`、`artifacts/moe-token-permute/`(6 文件)、`reports/moe-token-permute-ingest-report.md`。

## 7. Note
moe 类 kernel 页已达 **8 个**。token_permute 与 init_routing_quant 配对（init 排序 expert，permute 排序 token）。