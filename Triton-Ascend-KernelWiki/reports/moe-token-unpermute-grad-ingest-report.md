# moe_token_unpermute_grad CANN->Triton-Ascend ingestion report

## 1. Task parameters
| Parameter | Value | Status |
|---|---|---|
| OPERATOR_NAME | `moe_token_unpermute_grad` | source-confirmed |
| OPERATOR_TYPE | `moe` | source-confirmed |
| commit | `d28d8f9...` | source-confirmed |
| mode | static_ingest | - |

## 2. Source file inventory
~11 文件，无 arch 子目录。核心：`README.md`(135L)、`op_host/moe_token_unpermute_grad_def.cpp`(80L)、`op_kernel/moe_token_unpermute_grad.cpp`(46L,4 tiling key)、`op_kernel/moe_token_unpermute_grad_{base,prob_none,prob_not_none}.h`(197/125/373L)。完整清单见 `artifacts/moe-token-unpermute-grad/source_file_inventory.md`。

## 3. Operator semantics
- **功能**：`aclnnMoeTokenUnpermute` 的反向传播。计算 permutedTokensGrad（scatter 加权）+ probsGrad（topK 求和）。
- **公式**（有 probs）：`probsGrad=Σ(grad*permutedTokens[sortedIndices])` + `permutedTokensGrad[sortedIndices]=grad*probs`。无 probs：`permutedTokensGrad[sortedIndices]=grad`（纯 scatter）。
- **输入**：permuted_tokens/unpermuted_tokens_grad/sorted_indices/probs(OPTIONAL)。输出：permuted_tokens_grad/probs_grad。属性 padded_mode/restore_shape。
- **架构**：仅 A2/A3。AIV-only。950 转调 finalize_routing_v2_grad。
- **tiling key**：4 个（0/1/10/11）= padded_mode × haveProbs。2 模板：ProbNone/ProbNotNone。

## 4. Pipeline
有 probs：Gather -> ProbsGrad(Σ grad*gathered) -> Scatter(grad*probs)。无 probs：Scatter(grad)。

## 5. 与 unpermute 配对
正向 unpermute：gather + probs mul + reduce_sum。反向 grad：scatter 加权 + probsGrad reduce_sum。

## 6. Validation
```
validate.py -> OK
query moe token unpermute grad -> #1 hit kernel ✓
```

## 7. Created files
`sources/ops-transformer/moe-token-unpermute-grad.md`、`wiki/kernels/moe-token-unpermute-grad.md`、`artifacts/moe-token-unpermute-grad/`(6 文件)、`reports/moe-token-unpermute-grad-ingest-report.md`。

## 8. Note
moe 类 kernel 页达 **11 个**。token permute/unpermute 正反向配对完整（permute/permute_grad/permute_with_ep/permute_with_ep_grad/unpermute/unpermute_grad）。