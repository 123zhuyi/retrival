# moe_init_routing_quant_v2 CANN->Triton-Ascend ingestion report

## 1. Task parameters
| Parameter | Value | Status |
|---|---|---|
| OPERATOR_NAME | `moe_init_routing_quant_v2` | source-confirmed |
| OPERATOR_TYPE | `moe` | source-confirmed |
| commit | `d28d8f9...` | source-confirmed |
| mode | static_ingest | - |

## 2. Source file inventory
~40 文件，arch32 + arch35 双架构。核心：`README.md`(222L)、`op_host/moe_init_routing_quant_v2_def.cpp`(96L)、`op_kernel/moe_init_routing_quant_v2.cpp`(147L,arch32)+`_apt.cpp`(160L,arch35)、`op_kernel/moe_v2_{fullload_quant,fullload_dynamic_quant,gather_quant,gather_dynamic_quant,src_to_dst_and_gather,fullload_quant_base}.h`、`op_kernel/inner/*.h`(13) + `op_kernel/arch35/*.h`(13)。

## 3. Operator semantics
- **功能**：MoE routing V2。新增 drop/dropless 模式、动态 quant、删除 rowIdx、expertTokens 统计、INT4 输出。
- **输入**：x(FP16/BF16/FP32)/expert_idx(INT32)/scale/offset(OPTIONAL)。
- **输出**：expanded_x(INT8/INT4)/expanded_row_idx/expert_tokens_count_or_cumsum/expert_tokens_before_capacity/dynamic_quant_scale(OPTIONAL)。
- **属性**：active_num/expert_capacity/expert_num/drop_pad_mode/expert_tokens_count_or_cumsum_flag/expert_tokens_before_capacity_flag/quant_mode。
- **架构**：A2/A3 + 950(arch35)。AIV-only（MIX_AIV_1_0）。
- **tiling key**：3 档（20000 静态fullload / 21000 动态fullload / 10000-11110 多阶段）。

## 4. V2 vs V1
| 差异 | V1 | V2 |
|---|---|---|
| rowIdx | 有 | 删除 |
| drop/dropless | 无 | 有 |
| 动态 quant | 无 | 有 |
| expertTokens 统计 | 无 | 有 |
| 架构 | A2/A3 | A2/A3 + 950 |
| 输出 dtype | INT8 | INT8/INT4 |
| fullload 快速路径 | 无 | 有 |

## 5. Pipeline
6 步：Sort -> SrcToDst -> ExpertToken(Histogram+Cumsum) -> Quant(静态/动态) -> Gather -> Drop容量处理。fullload 路径单阶段融合。

## 6. Validation
```
validate.py -> OK
query moe init routing quant v2 -> #1 hit kernel ✓
query moe init routing quant -> V1+V2 配对命中 ✓
```

## 7. Created files
`sources/ops-transformer/moe-init-routing-quant-v2.md`、`wiki/kernels/moe-init-routing-quant-v2.md`、`artifacts/moe-init-routing-quant-v2/`(6 文件)、`reports/moe-init-routing-quant-v2-ingest-report.md`。