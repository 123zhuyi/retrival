# moe_token_unpermute CANN->Triton-Ascend ingestion report

## 1. Task parameters
| Parameter | Value | Status |
|---|---|---|
| OPERATOR_NAME | `moe_token_unpermute` | source-confirmed |
| OPERATOR_TYPE | `moe` | source-confirmed |
| commit | `d28d8f9...` | source-confirmed |
| mode | static_ingest | - |

## 2. Source file inventory
~11 文件，无 arch 子目录。核心：`README.md`(125L)、`op_host/moe_token_unpermute_def.cpp`(123L)、`op_kernel/moe_token_unpermute.cpp`(88L,~16 tiling key)、`op_kernel/moe_token_unpermute.h`(376L)、`op_kernel/moe_token_unpermute_common.h`(85L)。完整清单见 `artifacts/moe-token-unpermute/source_file_inventory.md`。

## 3. Operator semantics
- **功能**：MoE token 反向重排，按 sortedIndices gather + probs 乘 + topK 累加还原。
- **公式**（有 probs）：`T[k]=permutedTokens[sortedIndices[k]]` -> `T[k]*=probs[k]` -> `O[i]=Σ_{k=i*topK}^{(i+1)*topK-1} T[k]`。无 probs（topK=1）：`O[i]=permutedTokens[sortedIndices[i]]`。
- **输入**：permuted_tokens(FP16/BF16/FP32)/sorted_indices(INT32)/probs(OPTIONAL)。输出：unpermuted_tokens。属性 padded_mode/restore_shape。
- **架构**：A2/A3 + 310P/Kirin（仅 FP16/FP32）。950 转调 finalize_routing_v2。AIV-only。
- **tiling key**：~16 个，dtype×haveProbs×mix mode（probs dtype ≠ token dtype）。

## 4. Pipeline
3 阶段：Gather(sortedIndices) -> ProbsMul(可选) -> ReduceSum(topK 累加还原)。

## 5. 与 token_permute 配对
- permute：tokens -> permuteTokens（按 indices 排序展开）
- unpermute：permutedTokens -> unpermutedTokens（gather + probs + 累加还原）

## 6. Validation
```
validate.py -> OK
query moe token unpermute -> #1 hit kernel ✓
```

## 7. Created files
`sources/ops-transformer/moe-token-unpermute.md`、`wiki/kernels/moe-token-unpermute.md`、`artifacts/moe-token-unpermute/`(6 文件)、`reports/moe-token-unpermute-ingest-report.md`。

## 8. Note
moe 类 kernel 页达 **10 个**。token permute/unpermute 配对完整。