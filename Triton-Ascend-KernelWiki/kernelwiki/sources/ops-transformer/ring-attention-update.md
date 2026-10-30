---
id: ops-transformer-ring-attention-update
title: CANN ops-transformer RingAttentionUpdate source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: attention/ring_attention_update/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-29
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer RingAttentionUpdate 来源摘要

固定提交 `d28d8f9...`。支持 A2/A3 + 950 + KirinX90/Kirin9030。

## 核心源文件 SHA-256

| 源文件 | SHA-256 |
|---|---|
| `README.md` | `6927ed510afb97bc063ca1f38027b220507c3b400f006667f42764edd89cf1a3` |
| `op_host/ring_attention_update_def.cpp` | `30ce24c1e357480840e26cf4b411446f596a6bcf75428ae0cdc1c7917a80cd4f` |
| `op_kernel/ring_attention_update.cpp` | `52f7bc0669ed78b082cb07c3acb994dbe1538f171103cee7388ec8c8a5f32e37` |
| `op_kernel/ring_attention_update_apt.cpp` | `3d0b0480e733bbebc81e4f45a792d5f73f89c61c7eda471b252ff9c3c6dca49d` |

## 关键代码区域

- **公式**（README L18-28）：在线 softmax rescale 合并：`max=max(prev_max,cur_max)`、`sum=prev_sum·exp(prev_max-max)+exp(cur_max-max)`、`out=prev_out·exp(prev_max-max)/sum + cur_out·exp(cur_max-max)/sum`。
- **算子定义**（def.cpp L22-96）：输入 prev_attn_out/prev_softmax_max/prev_softmax_sum + cur 同族（REQUIRED，FLOAT/FP16/BF16）+ actual_seq_qlen（OPTIONAL，INT64）。输出 attn_out/softmax_max/softmax_sum。属性 input_layout("SBH")/input_softmax_layout。AICore config：910b/910_93 + 950(opFile=ring_attention_update_apt) + kirinX90/9030(仅 FP16/FLOAT)。
- **kernel dispatch**（`ring_attention_update.cpp` L32-84）：6 tiling key（0=FP16_SBH, 1=BF16_SBH, 2=FLOAT_SBH, 10/11/12=TND）。950 独立入口（`_apt.cpp`，tiling key 100/110/120 SBH/SoftmaxTND/TND）。
- **约束**（README L127-131）：TND 时 D 为 64 倍数，TND 时 actual_seq_qlen 必填。

## 衍生引用

- `wiki/kernels/ring-attention-update.md`、`artifacts/ring-attention-update/`