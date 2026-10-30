# moe_finalize_routing_v2_grad - CANN -> Triton-Ascend 迁移分类

四分类（directly transferable / rewrite to Triton-Ascend / CANN-specific do-not-transfer / needs later verification）。MoE Finalize Routing V2 Grad 是路由最终化的反向传播，核心是 scatter-add + 可选 scale 梯度。

| 知识点 | CANN 来源（path Lxx） | Triton-Ascend 处理方式 | 分类 | 可信度 | 需后续验证 |
|---|---|---|---|---|---|
| gradExpandedX[expandedRowIdx[i]] += gradY[i/K] | README L42-44 | `tl.atomic_add(gradX + expandedRowIdx[i], gradY[i/K])` | transferable | inferred | no |
| gradExpandedX[expandedRowIdx[i]] += gradY[i/K] * scales[i] | README L48-49 | `tl.atomic_add(gradX + expandedRowIdx[i], gradY[i/K] * scales[i])` | transferable | inferred | no |
| gradScales[i] = sum(expandedX[expandedRowIdx[i]] * gradY[i/K]) | README L52-53 | `tl.sum(expandedX[idx] * gradY[i/K])` | transferable | inferred | no |
| gradScales with bias: sum((expandedX + bias[expertIdx]) * gradY) | README L62-63 | `tl.sum((expandedX + bias[expertIdx]) * gradY)` | transferable | inferred | no |
| expandedRowIdx 驱动的 scatter-add | README L40-64 | `tl.atomic_add`（多行可能写同一位置） | transferable | inferred | yes（atomic_add 冲突） |
| 3 种场景（无 scale / 有 scale / 有 scale+bias） | README L40-63 | `if constexpr` 分支 | rewrite | inferred | no |
| 4 种 TilingKey（cut_h/not_cut_h × with/without_scale） | kernel.cpp L20-23 | `if constexpr` / 独立 kernel | rewrite | inferred | no |
| cut_h（切 H 轴） | `with_scale_cut_h.h` | `tl.program_id` 沿 H 维切分 | rewrite | inferred | no |
| not_cut_h（不切 H） | `with_scale_not_cut_h.h` | 单点 | rewrite | inferred | no |
| dropPadMode / activeNum / expertNum / expertCapacity | def.cpp 属性 | `tl.constexpr` | rewrite | inferred | no |
| 核间分核 + 原子写 | `tiling.cpp` | `tl.atomic_add` + mask | rewrite | inferred | yes（atomic_add 精度） |
| arch32 vs arch35 模板 | def.cpp L104/149 | 按 arch 选 kernel 分支 | rewrite | inferred | no |
| MTE DataCopy 搬运 | kernel 模板 | Triton load/store | CANN-specific | inferred | no |
| KERNEL_TYPE_MIX_AIV_1_0 | kernel.cpp L31 | 不迁移 | CANN-specific | inferred | no |
| tiling 参数具体数值 | `tiling.cpp` | constexpr 值 source-reported | needs verification | inferred | yes |
| UB 字节预算 | `tiling.h` | 不声称；需硬件实测 | needs verification | inferred | yes |