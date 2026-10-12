# flash_attention_score_grad — CANN 9.1.0 arch22 device 参考提取

> 本目录是 **ScaledDotProductAttentionBackward** 的 CANN 官方 device kernel 参考，取自本机 CANN 安装包，
> 与既有 `src/flash_attention_score_grad/`（gitcode ops-transformer @ d28d8f99a 克隆，arch32/arch35）**互为补充**。

## 出处与同源关系

| 项 | 值 |
|---|---|
| 服务 level4 算子 | `ScaledDotProductAttentionBackward`（`ascendc-kernelgen-data/npu_benchmark/level4/70_ScaledDotProductAttentionBackward.{json,py}`） |
| 源目录 | `/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/flash_attention_score_grad/` |
| 硬件代 | **arch22**（与既有 clone 的 arch32/arch35 不同代） |
| 文件 | 35：入口 `flash_attention_score_grad.cpp` + `flash_attention_score_grad_apt.cpp` + `arch22/`（33 个 kernel/tiling .h） |
| 体积 | 1.6M |
| 提取时间 | 2026-09-21 |

## 为什么要单列（gap 说明）

- 既有 `src/flash_attention_score_grad/op_kernel/` 只含 **arch32 / arch35**；本机 CANN 9.1.0 另带 **arch22** device kernel。
- arch22 的两个入口 `.cpp` 与 clone 对应文件 **内容不同**（`diff` 不一致），且 `arch22/` 整棵（33 文件）此前未曾入库。
- 因此本目录 = 补 arch22 硬件代的官方参考，二者不重叠、不覆盖。

## 目录结构

```
src/flash_attention_score_grad_cann921/
├── README_SOURCE.md
├── flash_attention_score_grad.cpp          # arch22 主入口（含 KFC_L1_RESERVER_SIZE=0 的 Gm in/Gm out 配置）
├── flash_attention_score_grad_apt.cpp
└── arch22/                                  # arch22 kernel + tiling .h
    ├── flash_attention_score_grad_tiling.h            # tiling 分发
    ├── flash_attention_score_grad_template_tiling_key.h # ASCENDC_TPL tilingkey 位段
    ├── flash_attention_score_grad_{pre,post,sfmg}.h   # 三段软max统计/主计算/写回
    ├── flash_attention_score_grad_s1s2_bn2gs1s2*.h    # 主计算变体（basic/sab/det）
    ├── flash_attention_score_grad_{constant_propagation,empty_tensor}.h
    ├── basic_modules/                                  # cube1/2/3 + vec_pre/sfmg/post 融合模块
    │   ├── cube_modules/{cube1_op,cube2_op,cube3_op,cube23_d64_op,cuber_op,cube_addr}.h
    │   └── vec_modules/{vec_pre,vec_post,vec_sfmg,_det}.h
    └── flash_attention_score_grad_ngs1s2_bn.h / bngs1s2_b.h / s1s2_bn2.h
```

## P1 主线建议

- arch22 与既有 clone 的 arch32/35 在调度/模块组织上同名，P1 抽取时按 **arch22 为主线**或**与 arch32/35 做三路并采**，差别点显式记录（尤其 `sfmg` 软max梯度前向换算、`cube23_d64_op` 的 D=64 专用 cube 档）。
- 需与 `output/flash_attention_score_grad/extraction.md`（arch32/35 既有抽取）合并视角。