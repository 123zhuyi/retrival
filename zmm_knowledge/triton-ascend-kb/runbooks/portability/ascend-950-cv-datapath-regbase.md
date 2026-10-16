---
schema_version: okf.v1
kind: guide
type: programming_guide
source_family: runbooks
category: portability
dimension: "4.3"
title: 昇腾 950 平台特性：CV 直通数据通路与 RegBase（含 Triton 可达性边界）
description: 950PR/DT 相对 910B/C 新增 Cube↔Vector 直通数据通路（L0C 经 Fixpipe 直送 UB、UB 经 MTE3 直送 L1，免 GM 中转）与 RegBase 寄存器编程接口（中间结果不落 UB）；CV 融合算子在 950 上额外获益 12%~37%，Triton 由编译器自动利用通路，RegBase 不可达、是 Triton→AscendC 升级判据之一。
tags: [ascend_950, cv_datapath, fixpipe, regbase, l0c2ub, ub2l1, platform_boundary, triton_ceiling]
confidence: draft
status: draft
created_at: "2026-08-27T00:00:00Z"
updated_at: "2026-09-14T00:00:00Z"
---

## 昇腾 950 平台特性与 Triton 可达性边界（来源：AscendC GDN 硬件特性优化节）

【优化维度】：平台｜硬件特性利用
【泛化层级】：平台事实（适用：950PR/950DT；不适用：910B/C、A2/A3。本卡为平台知识卡，供优化决策与跨平台预期管理使用）

**1. CV 直通数据通路（950 新增）**

| 通路 | 910B/C | 950PR/DT |
|---|---|---|
| Cube→Vector | L0C→GM→UB（GM 中转） | L0C --Fixpipe--> UB 直达 |
| Vector→Cube | UB→GM→L1 | UB --MTE3--> L1 直达 |

意义：CV 融合算子（matmul 与 elementwise 交替链，GDN Fwd_h/Fwd_o 型）的中间结果不再绕行 GM，搬运延迟与带宽占用双降。AscendC 消融：叠加 CV 通路后 Fwd_h 在 950PR +18.7%、950DT +29.4%。

**2. RegBase 寄存器编程（950 新增）**

MemBase 流程中 A→B→C→D 的链式 Vector 计算，中间结果 B/C 需读写 UB；RegBase 下操作数取自寄存器，A0 分块可不落 UB 直接算到 D0，省掉中间变量的 UB 读写。面向 **UB Bound** 程序。消融：再叠加 RegBase 后 950PR +24.0%、950DT +37.5%（相对无 950 特性版本）。

**Triton 可达性分级**：

- ✅ **CV 通路：编译器自动利用**。Triton-Ascend 在 950 上生成代码走 Fixpipe/MTE3 通路无需用户干预；但同一份 Triton kernel 在 950 vs 910B 上的 CV 融合收益差异主要来源于此，跨平台对比性能时须归因到平台而非代码。
- ❌ **RegBase：Triton 不可达**。Triton 编程模型无寄存器级操作数表达。
- **Triton→AscendC 升级判据**：kernel 已确认 UB Bound（Vector 链中间变量读写占大头）、访存与流水手段用尽、且目标平台为 950 时，转 AscendC RegBase 仍有约 15%~25% 的头部空间（GDN 消融外推）。

**3. 平台选型经验（访存受限算子）**

绝对性能排序 950DT > 950PR > 910B1：950DT 显存带宽高于 PR，GDN 实测 DT 相对 PR +12%~22%；950 相对 910B1 稳定 +38%~64%（核数 + CV 通路 + RegBase 共同贡献）。chunkSize=128 在 950 上需更多片上缓存支撑双缓冲，UB 预算更紧。

**注意事项**：

- 950PR 上优化良好的 Triton 基线可基本持平 950DT 的弱优化实现——平台红利不能替代优化本身。
- 引用本卡收益数字时必须带平台标签，禁止把 950DT 数字外推到 910B。

**已知实例（CV 低级 API 平台可达性）**：

- 910b1 + triton 3.2.0：`triton.language.extra.ascend` 下仅 libdevice，**无低级 CV API（al./bl. 子模块）**，CV 融合类优化（Batch 流水线、T0-T5 逐拍交错）import 即失败，连 910B 系编译崩溃风险的前置警告都触发不到。CV 融合优化点（latency-optimizer 点 28 类）在 910B 系 + triton 3.2.0 直接判「平台不可用」，勿再尝试（2026-09-14，Gdn2Fwd；证据：op_94_94_Gdn2Fwd_20260914_1219_8124/output/opt_iter_1/log.md）。

<!-- okf:related:start -->

# 相关

- 讲解接口: [计算优化: fixpipe 随路量化 + K 维循环展开 + 预取](../compute-pipeline/exp_matmul_002.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
