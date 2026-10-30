---
id: ops-transformer-moe-gating-top-k-softmax-v2
title: CANN ops-transformer MoeGatingTopKSoftmaxV2 source snapshot
type: source
repo: cann/ops-transformer
path: moe/moe_gating_top_k_softmax_v2/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-31
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer MoeGatingTopKSoftmaxV2 来源摘要

仓库快照固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`（2026-07-09）。`MoeGatingTopKSoftmaxV2` 是 MoE gating 的 topk + softmax 算子，按 renorm 标记区分 softmax 与 topk 的先后顺序。

## 1. 仓库与算子定位

- 仓库根: `kernelwiki/artifacts/ops-transformer/`
- 算子目录: `moe/moe_gating_top_k_softmax_v2/`

## 2. 核心源文件清单与 SHA-256

| 源文件 | SHA-256 | 作用 |
|------|------|------|
| `README.md` | `fe472161d5764dae4d867c42ea2a4729142f8ae11de93a05a33edabb444aa706` | 算子总览、renorm 0/1 公式 |
| `op_host/moe_gating_top_k_softmax_v2_def.cpp` | `288847da6fc97c2c9a5464cfbcdad93e7a67a9664a69ef92cce672b2751fcd02` | OpDef（2 套 AICore 配置）|
| `op_host/moe_gating_top_k_softmax_v2_tiling.h` | `be8381a7d6a859c909c61811d248d56775cffea60474e1e88877c4b249c6cac4` | 4 套 TilingData |
| `op_kernel/moe_gating_top_k_softmax_v2.cpp` | `8178a7b9f3debaa7e594ffe66a0fc902f1e3ffbeecd5ece4c30f5f9ef9954dfb` | kernel 入口（~36 tiling key）|
| `op_kernel/moe_gating_top_k_softmax_v2_perf.h` | `bc34ce5dccbd462a1e9ac8d8de73247071d6ad7def01f5e98957e83b1b6abfa5` | Perf 高性能模板 |

## 3. 关键代码区域摘要

### 公式（README.md）
- **renorm=0**：`softmaxResult = softmax(x, axis=-1)`；`yOut, expertIdxOut = topk(softmaxResult, k)`
- **renorm=1**：`topkOut, expertIdxOut = topk(x, k)`；`yOut = softmax(topkOut, axis=-1)`
- finished=True 时 expertIdxOut = num_expert

### 输入输出（def.cpp）
- 输入：x（FP16/FLOAT/BF16）、finished（BOOL，可选）
- 输出：y（FP16/FLOAT/BF16）、expert_idx（INT32）、softmax_result（FLOAT，可选）
- 属性：k（必需）、renorm(0)、output_softmax_result_flag(false)

### 架构（def.cpp L53-65）
ascend910b/910_93（membase）+ ascend950（regbase）

### Kernel dispatch（kernel L68-176）
`KERNEL_TYPE_MIX_AIV_1_0`（AIV-only）。~36 tiling key：4 模板策略（EKFullLoad 101xxx / KFullLoad 102xxx / KRenorm 102xxx / Perf 103xxx），按 dtype(3) × renorm(0/1) × bufferNum/colRange 区分。

### 约束（README.md）
k <= 1024；renorm 只支持 0/1；x 和 finished 每一维 <= INT32_MAX

## 4. 衍生引用
- `wiki/kernels/moe-gating-top-k-softmax-v2.md`
- `artifacts/moe-gating-top-k-softmax-v2/`