---
id: ops-transformer-recurrent-gated-delta-rule
title: CANN ops-transformer RecurrentGatedDeltaRule source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: attention/recurrent_gated_delta_rule/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-29
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer RecurrentGatedDeltaRule 来源摘要

固定提交 `d28d8f9...`。`fixture: true`。仅 A2/A3，BF16，AIV-only（无 Cube）。

## 1. 仓库与算子定位

- 仓库根：`kernelwiki/artifacts/ops-transformer/`
- 算子目录：`attention/recurrent_gated_delta_rule/`
- 固定提交：`d28d8f99a8858b71caa27a535da32bb5ba68ce2a`

## 2. 核心源文件 SHA-256

| 源文件 | SHA-256 |
|---|---|
| `README.md` | `bb2964d80a709b79a6d8d795aaf90f27a1a8cd524882fc464cc53ef24f3a447e` |
| `docs/aclnnRecurrentGatedDeltaRule.md` | `63b27050a78c55293990da2ec0fa4a605aa9198bb86d1b0152f87e3ff62966ae` |
| `op_host/recurrent_gated_delta_rule_def.cpp` | `05d617b61b70be600fcb0db523490931adea88ee8360dcb6e690443dec5a6e3b` |
| `op_host/recurrent_gated_delta_rule_tiling.h` | `4babd14f336791d700b58bdea3719bb3961cf678a5a95ec716be151842d10033` |
| `op_kernel/recurrent_gated_delta_rule.cpp` | `5ea0cecbe238a02e5d14f4dcc26af6c9e31a06b6a7138eb0b25a61d4c957b66f` |
| `op_kernel/recurrent_gated_delta_rule.h` | `3681292c16a289e6a56e48708241d7a40a6086f00e0d92e0de442d70538dd0c2` |
| `op_kernel/recurrent_gated_delta_rule_tiling_data.h` | `b55f5cf11d5b3d947ca65bec068acbc4d6158bafd29f4f4be76c099b1e731bec` |

## 3. 关键代码区域摘要

### 3.1 功能与公式（README L16-32）
RGDR：线性注意力（SSM 风格）。`S_t = α·Diag(αk)·S_{t-1} + β·(v_t - α·Diag(αk)·S_{t-1}·k_t)·k^T_t`，`o_t = S_t·q_t / √d`。

### 3.2 算子定义（def.cpp L22-92）
- 输入：query/key/value/beta/state/actual_seq_lengths/ssm_state_indices（REQUIRED，BF16）+ g/gk/num_accepted_tokens（OPTIONAL，FLOAT/INT32）。
- 输出：out/state（REQUIRED，BF16）。
- 属性：scale_value(1.0)。AICore config：ascend910b/910_93。

### 3.3 tiling 与 dispatch（tiling_data.h L23-38）
AIV-only。`vectorCoreNum`/`ubCalSize`/`sBlockNum`/`vStep`/`nk`/`dk`/`nv`/`dv`。7 步骤 tiling 类（tiling.h L37-63）。

### 3.4 约束
仅 BF16，仅 A2/A3，AIV-only。`BUFFER_NUM=1`（无 double buffer）。`MAX_REPEAT_TIME=255`。

## 4. 衍生引用

- `wiki/kernels/recurrent-gated-delta-rule.md`
- `artifacts/recurrent-gated-delta-rule/`

## 5. 复现性说明

本页仅记录源文件路径、SHA-256 与代码区域摘要。tiling 参数数值、dk/dv 约束、MTP 场景适用性等需结合硬件验证，相关结论在引用 wiki 页中标记 `confidence: inferred`。