---
schema_version: okf.v1
kind: runbook
type: hardware_spec
source_family: reference
title: 昇腾平台硬件规格速查（UB/L0/L1/核数/对齐）
description: 从 AscendNPU-IR target spec 源码抽取的各平台硬件常数，是卡片「片上内存预算」「分核策略」节的权威数据源。替代此前卡片中手写经验值（如 UB 一律按 192KB 的写法对 950 是错的）。
tags: [hardware, ub, l1, l0c, alignment, aicore, ascend910b, ascend950, target_spec]
confidence: verified
status: draft
created_at: "2026-09-11T00:00:00Z"
updated_at: "2026-09-15T00:00:00Z"
---

# 昇腾平台硬件规格速查

> [SRC] 唯一数据源：`AscendNPU-IR/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td`
> （生成工具 `bishengir-target-spec-tblgen`）。引用时标注平台代号；文档（zh）常数多为 A2，勿跨平台默认。

## 1. 平台级常数（BaseSpec）

| 平台 | Arch | UB | L1 | L0a/L0b | L0c | UB 对齐 | L1 对齐 | L0c 对齐 | td 锚点 |
|---|---|---|---|---|---|---|---|---|---|
| Ascend910B（A2） | dav-c220 | **192KB**（1572864b） | 512KB | 64KB/64KB | 128KB | 32B | 32B | 512B | :73-83 |
| Ascend310B | dav-m300 | 256KB（2097152b） | 1024KB | 64KB/64KB | 128KB | 32B | 32B | 512B | :125-135 |
| Ascend910_93（A3） | dav-c220 | 192KB（1572864b） | 512KB | 64KB/64KB | 128KB | 32B | 32B | 512B | :165-175 |
| **Ascend950（PR/DT）** | **dav-c310** | **248KB 可用**（2031616b；物理 256KB，编译器保留 8KB） | 512KB | 64KB/64KB | **256KB** | 32B | 32B | 512B | :223-235 |

950 特有：SIMT 数据缓存 MinimalDCacheSize=32KB / MaximumDCacheSize=120KB（:225-226）。

⚠️ **关键修正**：950 的 UB 可用预算是 **248KB** 而非 A2 的 192KB——此前 mhc 等卡片按
「UB 192KB」校核（如 mhc_sinkhorn 卡 §6），对 950 平台偏保守但方向安全；
新卡片必须按本表平台值校核。UB OVERFLOW 报错中的 `1572864 bits available` 即 A2 值。

## 2. 核数（AiCoreCount = Cube 核数；VectorCoreCount = 2×Cube）

| 系列 | 型号（Spec 名） | AIC（Cube） | AIV（Vector） |
|---|---|---|---|
| 910B | B1/B2/B2C | 24 | 48 |
| 910B | B3/B4/B4_1 | 20 | 40 |
| 910_93 | 9362/9363/9372 | 20 | 40 |
| 910_93 | 9381/9382/9391/9392 | 24 | 48 |
| 950 | 950z / 950PR_950z | 4 | 8 |
| 950 | 9579/957b/957d；950PR_9579/957a~957d；950DT_9571~9578 | 28 | 56 |
| 950 | 9581/9589/958a/958b；950PR_9589~958d；950DT_9581~9588 | 32 | 64 |
| 950 | 9599；950PR_9599/959a/959b；950DT_9591~9596/95A1/95A2 | 36 | 72 |
| 950DT | 950x/950y | 8 | 16 |
| 310B | 全系 | 1 | 1 |

规律确认：VectorCoreCount = 2 × AiCoreCount（与 zh 文档「1 Cube : 2 Vector」一致 [DOC]）。

## 3. 卡片写作规则

1. **分核**：grid 取物理核数时写「按运行时 `get_device_properties` 取值」而非写死核数
   （同 950 内部从 4 核到 36 核跨度 9 倍，写死必错）；本表仅用于预算估算与 sanity check。
2. **UB 预算**：校核公式 `Σ(tile 字节数 × buffer 份数) ≤ UbSize`，buffer 份数含 multibuffer
   翻倍（A2 默认开 double buffer 后可用≈96KB [DOC]）。
   **⚠️ 校准警告（[EXP] 2026-09-15 chunk_gated_delta_rule_fwd_h 生成实验）**：
   解析估计只覆盖命名 tile，未计入 dot 操作数 staging、mask/cast 临时量、exp 中间
   向量等，实测占用系统性高于估计——BT=128/BK=128/BV=64 @910b 解析判「贴顶可行」
   （~128-190KB vs 192KB），实测不可行，最终 BVT=16/32 才落地。规则：
   ① 解析预算达到容量的 ~2/3 即视为高风险，不得写「可行」结论；
   ② 卡片给配置建议时按估计值至少 1.5 倍冗余折算（单数据点校准，待更多实例修正）；
   ③ 「贴顶可行」类论断一律改写为「⚠️ 贴顶，须 P4 实测」。
3. **对齐**：tile 尾轴按 UB 对齐 32B（VV 类）/ L0c 对齐 512B（CV 类）设计——与
   backend_contracts.md C-010 的 zh 文档事实一致，本表给出源码级出处。
4. **L0c 差异**：950 的 L0c 翻倍至 256KB，Cube 类算子的 K 轴切分上限判断需按平台区分。

## 维护记录

- 2026-09-11 初版：全量抽取 NPUTargetSpec.td（910B×6 / 310B×4 / 910_93×6 / 950×34 个 Spec）。
- 2026-09-15 §3.2 增补 UB 解析预算校准警告（[EXP] chunk_gated_delta_rule_fwd_h 生成实验：
  解析「贴顶可行」实测不可行，BV=64@910b → 实际 BVT=16/32），含 2/3 容量高风险线、
  1.5 倍冗余折算、「贴顶须实测」写法规则。
- 待办：dtype 支持矩阵（向量指令级）仍需从 AVE intrinsic 定义抽取，本表不含。
