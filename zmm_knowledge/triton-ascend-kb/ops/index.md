---
kind: index
type: section_index
title: ops/ 算子参考卡
description: 从 Triton-Ascend-KernelWiki 迁移的 48 张算子语义 / tiling / dispatch 参考卡（operator_spec）。当前为静态分析级知识（inferred），后续挑高频算子改写成成功案例。
---

# ops/ 算子参考卡

本目录收录 48 张算子级参考卡，来源为 A 的 `kernelwiki/wiki/kernels/`。每张卡记录算子语义、输入输出、tiling key / dispatch、Agent actions 与 Triton 骨架，置信度 `inferred`（源码静态分析，未硬件实测）。

案例卡与 `runbooks/` 条目经「实例化 ⇄ 已知实例」双向回链，后续由 `triton-knowledge-*-ingest` skill 按需增补为真正的成功案例。

## 新增案例卡（2026-08-27）

- [GDN 线性注意力三阶段优化案例](gdn-chunked-linear-attention.md) — FLA 移植→Triton 体系化→AscendC 重写完整轨迹，chunk 递推+CV 融合类算子方法论样板

## 新增案例卡（2026-09-14，triton-knowledge-reflux 回流试点）

- [Gdn2Fwd（varlen Gated DeltaNet 2 前向）双骨架 host dispatch 实测案例](gdn2-fwd-chunked-wy-dispatch.md) — serial persistent + chunked WY 双骨架、host 成本模型 dispatch，50 case 几何平均 22.25（vs PyTorch eager 实测），大 T case 2.8-4.1×；；二次验证 31.28（2026-09-16）；confidence: verified（硬件实测）

## 新增算子卡（2026-09-04，AscendC→OKF 工作流产出）

- [Add（broadcast 逐元素加 + alpha）](add-broadcast-alpha.md) — GEIR Add 算子语义卡：广播 + alpha 融合、dtype 组合约束、fp32 中间精度策略；status: draft，P4 回放待做

## 新增算子卡（2026-09-07，AscendC→OKF 工作流产出）

- [FusedBiasLeakyRelu（偏置+LeakyReLU+缩放融合）](fused-bias-leaky-relu.md) — ops-nn GEIR 算子语义卡：等形无广播 activation 融合、fp32 计算域 + 末段单次 RTNE、BUFFER_MODE tilingkey；最简转化实例（0 个不可达项）；status: draft，P4 回放待做

## 新增算子卡（2026-09-08，AscendC→OKF 工作流产出）

- [ApplyRotaryPosEmb（RoPE 旋转位置编码 q/k 双路融合）](apply-rotary-pos-emb.md) — ops-transformer aclnn/GEIR 算子语义卡：三旋转模式 × 四布局 × cos/sin 广播 × partial rope 全规格；arch35 4 tilingkey/3 kernel 类计算数学相同 → 单 kernel + constexpr 合并论证；0 个不可达项；status: draft，P4 回放待做

## 新增算子卡（2026-09-09，AscendC→OKF 工作流产出）

- [ChunkGatedDeltaRuleFwdH（GDN/KDA chunk 间状态递推前向）](chunk-gated-delta-rule-fwd-h.md) — fla_npu（非 CANN 仓）CV 融合算子语义卡：v_new = u − w@h_prev、h_next = h_prev·exp(g_last)·exp2(gk_last) + kg^T@(v_new·decay)；关键决策 (b,hv,v_block) 状态条带常驻并行、V 列条带独立消解全部 GM workspace、禁 tl.trans 用转置寻址 load；Catlass mmad/Reg VF 按黑盒降级；status: draft，P4 回放待做
