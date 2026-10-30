# moe_token_permute_with_ep CANN->Triton-Ascend ingestion report

## 1. Task parameters
| Parameter | Value | Status |
|---|---|---|
| OPERATOR_NAME | `moe_token_permute_with_ep` | source-confirmed |
| OPERATOR_TYPE | `moe` | source-confirmed |
| commit | `d28d8f9...` | source-confirmed |
| mode | static_ingest | - |

## 2. Source file inventory
~17 文件，无 arch 子目录。核心：`README.md`(140L)、`op_host/moe_token_permute_with_ep_def.cpp`(72L)、`op_kernel/moe_token_permute_with_ep.cpp`(86L)、`op_kernel/moe_{sort_base,sort_one_core,sort_multi_core,mrgsort,mrgsort_out,index_copy,index_copy_spilt_d}.h`（与 token_permute 共享）。完整清单见 `artifacts/moe-token-permute-with-ep/source_file_inventory.md`。

## 3. Operator semantics
- **功能**：MoE permute with EP，根据 indices 将 tokens 和可选 probs 广播排序并按 range 切片。
- **公式**（paddedMode=false）：`sortedIndicesFirst=argSort(indices)` -> `sortedIndicesOut=argSort(sortedIndicesFirst)` -> 当 `range[0]<=sortedIndices[i]<range[1]`：`permuteTokens[idx-range[0]]=tokens[i//topK]` + `permuteProbs[idx-range[0]]=probs[i]`。
- **输入**：tokens(FP16/BF16/FP32)/indices(INT32/INT64)/probs(OPTIONAL)。输出：permute_tokens/sorted_indices(INT32)/permute_probs。属性 range/num_out_tokens/padded_mode。
- **架构**：仅 A2/A3。AIV-only。
- **tiling key**：8 个（1-8）= one/multicore × normal/spilt_d × padded_false/true。

## 4. 与 token_permute 差异
| 差异 | token_permute | token_permute_with_ep |
|---|---|---|
| probs 输入/输出 | 无 | 有 |
| range 属性 | 无 | 有（EP 切分范围） |
| index_copy | 单 tokens | tokens + probs 双输出 |
| 切片逻辑 | numOutTokens | range[0]<=idx<range[1] |
| 回退 | - | range 空时回退到 token_permute |

## 5. Pipeline
3 阶段：Sort1(argSort indices) -> Sort2(argSort sortedFirst) -> IndexCopy（tokens + probs 双输出，按 range 切片）。

## 6. Validation
```
validate.py -> OK
query moe token permute with ep -> #1 hit kernel ✓
```
PROVENANCE 5 文件 sha256 已冻结。

## 7. Created files
`sources/ops-transformer/moe-token-permute-with-ep.md`、`wiki/kernels/moe-token-permute-with-ep.md`、`artifacts/moe-token-permute-with-ep/`(6 文件)、`reports/moe-token-permute-with-ep-ingest-report.md`。

## 8. Note
moe 类 kernel 页达 **9 个**。token_permute 系列（permute / with_ep / grad）完整配对。