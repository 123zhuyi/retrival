# recurrent_gated_delta_rule CANN->Triton-Ascend ingestion report

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| OPERATOR_NAME | `recurrent_gated_delta_rule` | source-confirmed |
| OPERATOR_TYPE | `attention` | inferred |
| commit | `d28d8f99a8858b71caa27a535da32bb5ba68ce2a` | source-confirmed |
| mode | static_ingest | - |

## 2. Source file inventory

~18 文件，无 arch 子目录。核心：`README.md`(119L)、`docs/aclnnRecurrentGatedDeltaRule.md`(557L)、`op_host/recurrent_gated_delta_rule_def.cpp`(97L)、`op_host/recurrent_gated_delta_rule_tiling.cpp`(445L)、`op_kernel/recurrent_gated_delta_rule.cpp`(38L)、`op_kernel/recurrent_gated_delta_rule.h`(429L)、`op_kernel/recurrent_gated_delta_rule_tiling_data.h`(42L)。完整清单（含 SHA-256）见 `artifacts/recurrent-gated-delta-rule/source_file_inventory.md`。

## 3. Operator semantics

- **功能**：线性注意力（SSM 风格）RGDR。`S_t = α·Diag(αk)·S_{t-1} + β·(v - α·Diag(αk)·S_{t-1}·k)·k^T`，`o_t = S_t·q/√d`（README L24-31）。
- **输入**：query/key/value/beta/state/actual_seq_lengths/ssm_state_indices（REQUIRED，BF16）+ g/gk/num_accepted_tokens（OPTIONAL）。
- **输出**：out/state（BF16，原地更新）。
- **属性**：scale_value(1.0)。AICore：ascend910b/910_93。
- **约束**：仅 BF16，仅 A2/A3，**AIV-only**（`KERNEL_TYPE_AIV_ONLY`，无 Cube）。`BUFFER_NUM=1`。

## 4. Pipeline

AIV-only，串行 token 遍历（无 CV 流水）：门控 exp → 记忆衰减 α·αk·S → 误差 v - decayed·k → 外积更新 S → 输出 o = S·q·scale。

## 5. Tiling

AIV 分核（vectorCoreNum），vStep/sBlockNum 控制 V 维分块。BF16 FP32 中间。BUFFER_NUM=1。

## 6. Core partition

沿 batch×head 分核，每核串行遍历 token（cu_seqlens 前缀和驱动）。

## 7. Migration

transferable：核心公式、门控 exp、AIV-only 语义、变长 cu_seqlens。rewrite：分核→program_id、constexpr。CANN-specific：BUFFER_NUM=1。

## 8. Validation

```
validate.py        -> OK
query recurrent gated delta rule -> #1 hit kernel ✓
query rgdr                        -> 2 hits (source+kernel) ✓
```

## 9. Created files

`sources/ops-transformer/recurrent-gated-delta-rule.md`、`wiki/kernels/recurrent-gated-delta-rule.md`、`artifacts/recurrent-gated-delta-rule/`(6 文件)、`reports/recurrent-gated-delta-rule-ingest-report.md`。