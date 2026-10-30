# ring_attention_update CANN->Triton-Ascend ingestion report

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| OPERATOR_NAME | `ring_attention_update` | source-confirmed |
| OPERATOR_TYPE | `attention` | inferred |
| commit | `d28d8f99a8858b71caa27a535da32bb5ba68ce2a` | source-confirmed |
| mode | static_ingest | - |

## 2. Source file inventory

~25 文件，arch32+arch35 双架构 + Kirin。核心：`README.md`(137L)、`op_host/ring_attention_update_def.cpp`(169L)、`op_host/ring_attention_update_tiling.cpp`(634L)+`_arch35.cpp`(873L)、`op_kernel/ring_attention_update.cpp`(85L,6 tiling key)、`op_kernel/ring_attention_update_apt.cpp`(60L,950 3 tiling key)、`op_kernel/arch35/`(4 模板)。完整清单见 `artifacts/ring-attention-update/source_file_inventory.md`。

## 3. Operator semantics

- **功能**：Ring Attention 输出合并。将两次 FA 的输出根据 softmax max/sum 做在线 rescale（README L14-15）。
- **公式**：`new_max=max(prev_max,cur_max)`、`alpha=exp(prev_max-new_max)`、`sum=prev_sum*alpha+exp(cur_max-new_max)`、`out=prev_out*alpha/sum + cur_out*exp(cur_max-new_max)/sum`（README L18-28）。
- **输入**：prev_attn_out/prev_softmax_max/prev_softmax_sum + cur 同族（REQUIRED）+ actual_seq_qlen（OPTIONAL）。
- **输出**：attn_out/softmax_max/softmax_sum。
- **属性**：input_layout("SBH")/input_softmax_layout("")。
- **架构**：A2/A3 + 950 + KirinX90/Kirin9030。FP16/BF16/FLOAT（Kirin 仅 FP16/FLOAT）。

## 4. Pipeline

纯 elementwise（AIV-only，无 Cube）：读 prev/cur → `max` → `exp` → `sum` → `out` → 写回。6 tiling key(arch32) + 3 tiling key(arch35 regbase)。

## 5. Tiling

纯 elementwise 分核。tiling key 编码 dtype×layout(0/1/2 SBH + 10/11/12 TND × FP16/BF16/FLOAT)。

## 6. Migration

transferable：rescale 公式（与 FlashSoftmax 同逻辑）、FP32 中间、SBH/TND layout 寻址。rewrite：tiling key→if constexpr、950 regbase 模板→launcher 分支。

## 7. Validation

```
validate.py        -> OK
query ring attention update -> #1 hit kernel ✓
query ring attention        -> #1 hit kernel ✓
```

## 8. Created files

`sources/ops-transformer/ring-attention-update.md`、`wiki/kernels/ring-attention-update.md`、`artifacts/ring-attention-update/`(6 文件)、`reports/ring-attention-update-ingest-report.md`。