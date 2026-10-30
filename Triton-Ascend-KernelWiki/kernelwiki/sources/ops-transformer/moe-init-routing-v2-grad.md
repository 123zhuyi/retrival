---
id: ops-transformer-moe-init-routing-v2-grad
title: CANN ops-transformer MoeInitRoutingV2Grad source snapshot
type: source
repo: cann/ops-transformer
path: moe/moe_init_routing_v2_grad/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-31
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer MoeInitRoutingV2Grad 来源摘要

仓库快照固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`（2026-07-09）。`MoeInitRoutingV2Grad` 是 `MoeInitRoutingV2` 的反向传播算子。

## 1. 仓库与算子定位

- 仓库根: `kernelwiki/artifacts/ops-transformer/`
- 算子目录: `moe/moe_init_routing_v2_grad/`

## 2. 核心源文件清单与 SHA-256

| 源文件 | SHA-256 | 作用 |
|------|------|------|
| `README.md` | `332f3f5083c62980c87d17fbee67bbce8e30866c11835d292017112037dea345` | 算子总览、反向公式 |
| `op_host/moe_init_routing_v2_grad_def.cpp` | `45afc299889c3b9efba3a4759211fe08ad73a47bacbe872f64ad1a41d07881ab` | OpDef（grad_expanded_x/expanded_row_idx -> grad_x）|
| `op_host/moe_init_routing_v2_grad_tiling.h` | `17adc65558b7535cc94001aa156d8e463c02842af2e193acab9bdf09493f63b7` | TilingData、3 套 regbase TilingData |
| `op_kernel/moe_init_routing_v2_grad.cpp` | `5d839347769cabbde5e69f0998ab3f483a4c4e6a1b7683bb66819894771ac450` | kernel 入口（9 tiling key）|
| `op_kernel/moe_init_routing_v2_grad_base.h` | `a49d75799be1838105bb6d319ed3080a03745889f263a500f067913911e0fd0a` | 基础模板 |

## 3. 关键代码区域摘要

### 核心公式（README.md）
```
gradX_i = Σ_{t=0}^{topK} gradExpandedX[expandedRowIdx[i*topK + t]]
```

### 输入输出（def.cpp L22-38）
- 输入：grad_expanded_x（FP16/BF16/FLOAT）、expanded_row_idx（INT32）
- 输出：grad_x（FP16/BF16/FLOAT）
- 属性：top_k（必需）、drop_pad_mode(0)、active_num(0)

### 架构（def.cpp L42-54）
ascend910b/910_93（membase, opFile=moe_init_routing_v2_grad）、ascend950（regbase, opFile=moe_init_routing_v2_grad_apt）

### Kernel dispatch（kernel L22-83）
`KERNEL_TYPE_AIV_ONLY`（g_coreType==AIC return）。9 tiling key：3 模式（Dropless/Activate/PositionPad0）× 3 dtype（FLOAT/FLOAT16/BF16）。3 模板类。

## 4. 衍生引用
- `wiki/kernels/moe-init-routing-v2-grad.md`
- `artifacts/moe-init-routing-v2-grad/`