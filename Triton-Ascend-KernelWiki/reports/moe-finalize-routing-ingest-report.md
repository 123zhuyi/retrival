# moe_finalize_routing CANN->Triton-Ascend ingestion report

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| OPERATOR_NAME | `moe_finalize_routing` | source-confirmed |
| OPERATOR_TYPE | `moe` | source-confirmed（tags.yaml#kernel_types 有 moe） |
| commit | `d28d8f99a8858b71caa27a535da32bb5ba68ce2a` | source-confirmed |
| mode | static_ingest | - |

## 2. Source file inventory

~25 文件，无 arch 子目录。BF16 和 FP 双谱系（6+8 模板）。核心：`README.md`(109L)、`op_host/moe_finalize_routing_def.cpp`(148L)、`op_kernel/moe_finalize_routing.cpp`(143L,19 tiling key)、`op_kernel/moe_finalize_routing_common.h`(39L)。完整清单见 `artifacts/moe-finalize-routing/source_file_inventory.md`。

## 3. Operator semantics

- **功能**：MoE FFN 输出路由合并（gather+scale+add）（README L18-19）。
- **公式**：`out = x1 + x2 + Σ scales·(expandedX[idx] + bias[expertId])`（README L21-27）。
- **输入**：expanded_x/x1/x2(opt)/bias/scales/expanded_row_idx/expanded_expert_idx（FP16/FP32/BF16，INT32）。
- **输出**：y。
- **架构**：A2/A3 + KirinX90/Kirin9030（仅 FP16/FP32）。
- **模板**：19 tiling key(20000-20018)，dtype×变体（CutK/Cuth/K2/K4/Db/DbAllBias/AllBias/CuthK2Optimized）。

## 4. Pipeline

AIV-only（无 Cube）：索引 gather → expandedX gather → bias add → scale multiply → Σ累加 → x1+x2 add → 写回。

## 5. Tiling

19 tiling key 编码 dtype×模板变体。`BUFFER_NUM=1`，`PARALLEL_NUM=2`。

## 6. Migration

全部可迁移（纯 elementwise gather+scale+add）。CANN-specific：BUFFER_NUM/PARALLEL_NUM 优化参数、310P BF16 排除。

## 7. Validation

```
validate.py        -> OK
query moe finalize routing -> #3 hit kernel（moe 系列 5 个 kernel 页完整）✓
query moe                  -> 10 hits ✓
```

## 8. Created files

`sources/ops-transformer/moe-finalize-routing.md`、`wiki/kernels/moe-finalize-routing.md`、`artifacts/moe-finalize-routing/`(6 文件)、`reports/moe-finalize-routing-ingest-report.md`。

## 9. Note

这是 KB 中首个 `kernel_types: [moe]` 的 kernel 页。moe 类别还有 `moe_init_routing`、`moe_init_routing_v2`、`moe_finalize_routing_v2`、`moe_finalize_routing_v2_grad` 等已有页。