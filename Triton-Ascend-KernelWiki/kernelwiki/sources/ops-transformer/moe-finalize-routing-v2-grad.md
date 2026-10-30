---
id: ops-transformer-moe-finalize-routing-v2-grad
title: CANN ops-transformer MoeFinalizeRoutingV2Grad source snapshot
type: source
repo: cann/ops-transformer
path: moe/moe_finalize_routing_v2_grad/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-31
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer MoeFinalizeRoutingV2Grad 来源摘要

仓库快照固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`（2026-07-09）。`MoeFinalizeRoutingV2Grad` 是 `MoeFinalizeRoutingV2` 的反向传播算子。

## 1. 仓库与算子定位

- 仓库根: `kernelwiki/artifacts/ops-transformer/`
- 算子目录: `moe/moe_finalize_routing_v2_grad/`
- 固定提交: `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`

## 2. 核心源文件清单与 SHA-256

| 源文件 | SHA-256 | 作用 |
|------|------|------|
| `README.md` | `f2d2b90405b73422c97d6dc456a180126345b261307b0b17ecb99a1e57180813` | 算子总览、3 场景反向公式 |
| `op_host/moe_finalize_routing_v2_grad_def.cpp` | `34a18de7957d72ec67cffd16750e6e3e8c1cdbb10d16f6586c58ed287f5c07b8` | OpDef（2 套 AICore 配置）|
| `op_host/moe_finalize_routing_v2_grad_tiling.h` | `a80ba9c2b870ed6d4fd9028e0e19aa6e8a0ec4892b146fc245cd6cec8dce982f` | TilingData 结构体 |
| `op_kernel/moe_finalize_routing_v2_grad.cpp` | `9c41ae72af5f88c053cdd8a0ecaba30029f8c77a82758a25fcbfe3f83bdae710` | kernel 入口（4 tiling key）|
| `op_kernel/moe_finalize_routing_v2_grad_base.h` | `7af8024901a1fd74c7d69c8d45d7fee758367f3ed5781768a35da57c8bf6c672` | 基础模板 |

## 3. 关键代码区域摘要

### 3 场景反向公式（README.md L40-64）
1. 无 scales：`gradExpandedXOut[expandedRowIdx[i]][j] = gradY[i/K][j]`（scatter）
2. 有 scales 无 bias：`gradExpandedXOut[idx[i]][j] = gradY[i/K][j]*scales[i]`；`gradScalesOut[i] = sum(expandedX[idx[i]][j]*gradY[i/K][j])`
3. 有 scales 有 bias：同 2，gradScalesOut 含 `+bias[expertIdx[i]][j]`

### 输入输出（def.cpp）
- 输入：grad_y（FP16/BF16/FLOAT）、expanded_row_idx（INT32）；可选 expanded_x/scales/expert_idx/bias
- 输出：grad_expanded_x、grad_scales（FP16/BF16/FLOAT）
- 属性：drop_pad_mode、active_num、expert_num、expert_capacity

### 架构（def.cpp L66-149）
ascend910b/910_93（membase）、ascend950（regbase）

### Kernel dispatch（kernel L20-58）
`KERNEL_TYPE_MIX_AIV_1_0`（AIV-only）。4 tiling key：WithoutScale/WithScale × CutH/NotCutH（10001/10002/20001/20002）。4 模板类。

## 4. 衍生引用
- `wiki/kernels/moe-finalize-routing-v2-grad.md`
- `artifacts/moe-finalize-routing-v2-grad/`

## 5. 复现性说明

本来源页仅记录源文件路径、SHA-256 和关键代码区域摘要，不包含完整源码。要复现分析，需 checkout 到固定提交并按上述路径读取原文件。tiling 参数数值、atomic_add 多核精度边界需结合源码/硬件进一步验证。本页为 `static_ingest` 静态分析，未运行/未验证/未 benchmark 算子。