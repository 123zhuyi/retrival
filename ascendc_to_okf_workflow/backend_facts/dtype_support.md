---
schema_version: okf.v1
kind: runbook
type: dtype_support
source_family: curated
title: Triton-Ascend 数据类型支持与标量化陷阱
description: 昇腾向量运算单元的 dtype 支持事实与「静默标量化」性能陷阱清单。核心原则：dialect 接受某 dtype ≠ 向量指令原生支持；无原生支持时经 ImplByScalarOpInterface 退化为逐元素标量运算，性能悬崖。标量化判定规则已落实到编译器源码（shouldLowerToScalarLoops），分 A2/A3（mem-based）与 950/A5（reg-based）两架构给出。供 P2 骨架选型与 preflight 检查使用。
tags: [triton_ascend, dtype, scalarization, vector, int64, int32, mask, performance, regbase, v300]
confidence: verified
status: draft
created_at: "2026-09-11T00:00:00Z"
updated_at: "2026-09-14T00:00:00Z"
---

# Triton-Ascend 数据类型支持与标量化陷阱

> 仲裁顺序：[SRC] AscendNPU-IR 源码 > [DOC] model/zh 文档 > [EXP] 回放实测。
> 相关卡：`backend_contracts.md`（C-001/C-008/C-009 的详细版即本卡）。

## 1. 核心机制：为什么"能编译"不等于"是向量指令"

HIVM dialect 的 op 类型约束（`OperElemTypeConstraints`）决定**能否通过编译**；
而是否**标量化**由编译器在 `hivm-lower-to-loops` pass 中逐 op 判定：

```
hivm.vadd(i64) ──dialect 校验──▶ 通过（接受 I64）
        │
        ▼ HIVMLowerToLoopsPass（每个 ImplByScalarOpInterface op）
        op.shouldLowerToScalarLoops() == true ?
        ├─ 是 → 展开为 scf 标量循环（结果正确，性能悬崖）
        └─ 否 → 保留向量 op，走向量指令/模板库路径
```

[SRC] 证据锚点：
- 判定入口：`bishengir/lib/Dialect/HIVM/Transforms/HIVMLowerToLoops.cpp:42-66`
- **判定规则（权威表）**：
  `bishengir/lib/Dialect/HIVM/IR/ImplByScalarOpInterface/ShouldLowerToScalarLoops.cpp`
  （主树，内含 `isRegBasedArch` 分支区分 A2/A3 与 950；hivmc 子树有同名胜简版）
- dialect 接受集（宽于指令支持集）：
  `bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td`（如 vadd :646 接受 I64）

## 2. 标量化判定权威表 [SRC]

来源：主树 `ShouldLowerToScalarLoops.cpp`（2026-09-14 抽取，快照 @9c51a351）。
**所有规则前提**：op 为 pure buffer semantics；SIMT/VF 路径一律标量化（`util::isSIMTVF`）；
含硬件不支持的标量操作数（`hasHWUnsupportedScalarOperand`）一律标量化。

### 2.1 两架构共通规则

| op | 标量化条件 | 向量安全的 dtype |
|---|---|---|
| vadd / vsub / vmul / vmin / vmax / vabs / vshl / vdiv / vinterleave / vdeinterleave | **i64** | i8/i16/i32/u8/u16/u32/f16/f32（bf16 见 §3 注） |
| **vmod** | **i64 或 i32**（:147 走 mod 专用规则 :61-72） | i8/i16/f16/f32 |
| vmulext / vmulextui | i32 或 i64（dialect 又只接受 i32 ⇒ 实际总标量化） | 无安全整型 |
| vcmp（整型输入） | 非 i32，**或** i32 但模式 ∉ {EQ, NE}（:262-264） | **仅 i32 的 EQ/NE 保留向量**；i32 的 LT/LE/GT/GE 标量化；i8/i16/i64 全模式标量化 |
| vcmp（浮点输入） | 不标量化（:258 提前返回） | f16/f32/bf16 全模式安全 |
| vcumsum / vcumprod / vcummax / vcummin | dst 为 **i64** | 非 i64 |
| vreduce（min/max/sum/prod/xori） | A2/A3：i64；**950：不按 dtype 标量化**（:453-463 regbase 分支仅查 index 变体几何） | 950 上 i64 reduce 不走标量兜底（走模板库/其他路径，实测前勿外推性能） |
| vreduce *_with_index（left/right） | i16/i32/i64 一律标量化；f16/f32/bf16 当 flatten 后 rank>2 标量化（right 变体在 1 维或尾轴归约时也标量化）；950 上当模板库几何/对齐非法时标量化 | f16/f32/bf16 且 rank≤2 的规整几何 |

### 2.2 架构差异点

| 项 | A2/A3（mem-based） | 950/A5（reg-based, V300 ISA） |
|---|---|---|
| vcum* 非 i64 | cum 维为 flatten 后最内维时也标量化（:46-54） | 不标量化（走 vcadd/vcmax 指令或模板库） |
| vcumsum i64 特例 | 标量化 | 1 维、dim0、且 (i64→i64) 或 (i32→i64) 时**走 SIMT Sklansky 库调用**而非标量循环（:184-201）；其他 i64 组合无 bc 符号、仍标量化 |
| vshr | i64 标量化；移位量为张量/广播时也标量化（A3 无 `vshr_*` 库符号，:130-139） | 仅 i64 标量化 |
| vreduce i64 | min/max/sum/prod/xori 标量化 | 不标量化（见上表注） |

### 2.3 950 指令层（V300/dav-c310）dtype 支持 [SRC]

来源：`hivmc/bishengir/lib/Dialect/HIVMRegbaseIntrins/Utils/RegbaseUtils.cpp`
与 `hivmc/.../Dialect/HIVMRegbaseIntrins/IR/HIVMRegbaseIntrins.td`。

- **`isHIVMSupportedType`**（RegbaseUtils.cpp:634-643）：regbase 仅支持
  **i8/i16/i32（含无符号）+ f16/f32**——**不含 i64、不含 bf16**。
- 向量算术指令选择器 `buildRegbaseBinaryVectorOp`（:649-677，vadd/vsub/vmax/vmin/
  vcadd/vcmax 共用）：s8/s16/s32/f16/f32 + u8/u16/u32；i64/bf16 命中
  `llvm_unreachable`（即前端必须在此之前消除这些类型）。
- `buildVselOp`（:690-723）与 `buildVdupOp`（:611-632）：i8/i16/i32/u + f16/f32/**bf16**；
  无 i64。
- 向量寄存器长度（`createVLVectorType` :907-925，V300 ISA）：8bit→256 元素，
  16bit→128，32bit→64，64bit→32。谓词寄存器统一 vector<256xi1>。
- intrinsic dialect 类型约束 `ValidVector`（HIVMRegbaseIntrins.td:64-84）形式上
  接受 v32xi64，但选择器层 i64 不可达——**dialect 接受 ≠ 指令支持**在指令层再次成立。

## 3. mask 运算规范范式（官方示例 + 源码互证）[DOC+SRC]

`tl.arange` 默认产出 int32，直接 `<` 比较 = i32 LT 模式的 vcmp = **§2.1 确认的标量化路径**。
官方推荐模式（layernorm 示例）与源码规则完全互证：

```python
cols = tl.arange(0, BLOCK_N)                    # int32 —— 直接 < 会标量化
# ✅ where 内比较：转 fp32 再比（浮点 vcmp 全模式向量化，§2.1）
xbar = tl.where(cols.to(tl.float32) < N, x - mean, 0.0)
# ✅ store 的 mask：用整型比较（配合 backend_contracts C-002：避免 fp32 导出 mask）
tl.store(Out + cols, out, mask=cols < N)
```

规则归纳：
1. **参与 `tl.where`/算术的掩码比较** → 操作数转 fp32（源码级依据：浮点 vcmp 不标量化）；
2. **`tl.store` 的 mask** → 保持整型比较（fp32 导出 mask 有非确定 store 污染风险，
   见 backend_contracts.md C-002）；i32 等值比较（==/!=）恰好是 vcmp 向量安全模式，
   不等值比较此处属"正确性优先、性能让路"的已知例外；
3. 两者矛盾时以「store mask 用整型、中间计算用 fp32」分离写法为准。

## 4. 累加精度与 dtype 提升（工作流既有经验，同源记录）

- 中间累加用 fp32（bf16/fp16 输入在 Compute 中提升），回写时再降精度——
  P1 抽取时必须记录 Ascend C kernel 的 cast 提升点（详设「精度策略」节），
  P2 骨架用 `.to(tl.float32)` 对应。
- `TRITON_DEFAULT_FP_FUSION`（默认 1）控制 mul+add→fma 融合，精度比对异常时
  可设 0 排查 [DOC: zh 环境变量表]。

## 5. preflight 增补检查项

并入 `backend_contracts.md` preflight 清单第 4 条的细化：

| # | grep 模式 | 判定 | 处置 |
|---|---|---|---|
| 4a | `int64` 且同表达式含 `+`/`-`/`*`/`<<`/`>>`/`/` | **确认** ADD/SUB/MUL/SHL/SHR/DIV 标量化（§2.1 [SRC]） | 降 int32 或论证必要性 |
| 4a'' | **i32 向量除法（`//`/`/` on int32 tensor）在热路径** | **性能灾难实测 44x**（SwiGLU 回放：offs//HA decode 13.4ms → 消除后 0.3ms，2026-09-18）——vdiv-i32 指令存在 ≠ 可用；地址 decode 用代数化简/查表/结构消除，勿用向量除法 | 热路径禁用；冷路径（一次性）可豁免 |
| 4a' | `int64` 的 min/max/abs/reduce/cumsum | **确认**标量化（§2.1 [SRC]；950 reduce 除外） | 同上 |
| 4b | `tl.arange`/int32 结果直接 `<`/`>`/`<=`/`>=` | **确认** vcmp 标量化（§2.1 [SRC]，i32 仅 EQ/NE 向量安全） | `.to(tl.float32)` 后比较 |
| 4b' | i8/i16/int64 的任何整型比较 | **确认** vcmp 标量化 | 比较前转 fp32（或 i32+EQ/NE） |
| 4c | `mask=` 右值为 fp32 比较 | C-002 风险 | 改整型比较（接受 4b 的代价，正确性优先） |
| 4d | kernel 内 `%` 且操作数 i32/i64 | **确认** vmod 标量化（§2.1 [SRC]）+ C-001 禁用 | 用 `a-(a//b)*b` 且操作数保 i16/f32 |
| 4e | `offs_m[:, None] * K` 类纯 int32 行寻址 | **numel ≥ 2³¹ 时静默溢出错值**（正确性，非性能） | int64 仅限**标量块基址**（`blk.to(tl.int64) * BLOCK_M * K`），块内偏移保 int32；int64 不进向量 lane（不受 4a 标量化影响） |

> **4e 来源**（2026-09-14 mhc_pre 重提取）：旧骨架 `offs_m[:, None] * K` 在 T·K ≥ 2³¹ 时
> 溢出——benchmark 13 case 中 6 个触发（T·K 最大 4.3e9）。小 shape 调试不复现，
> 属最难排查的静默错值类缺陷。判定需 shape 上限信息（半自动：grep 出行寻址模式后
> 人工核算 `T_max × K_max`）。大 K/大 T 算子（mhc 全族等）通用。

## 附录 A：全量向量 op dtype 约束矩阵 [SRC]

从 `HIVMVectorOps.td` 机械抽取（56 个 op，2026-09-11）；标量化列 2026-09-14 按
`ShouldLowerToScalarLoops.cpp` 回填（**该列现为 [SRC] 确认**，非推测）。列含义：
**dialect 接受类型** = 编译放行；**标量化 dtype** = shouldLowerToScalarLoops 确认的标量兜底
触发类型（前提：pure buffer semantics，两架构共通规则；架构差异见 §2.2）。

| op | mnemonic | dialect 接受类型（节选） | 标量化 dtype（确认） |
|---|---|---|---|
| VAbsOp | vabs | F16/F32/I8/I16/I32/I64 | **i64** |
| VAddOp | vadd | I8/I16/I32/F16/F32/I64 | **i64** |
| VSubOp / VMulOp | vsub/vmul | 同上族 | **i64** |
| VMaxOp / VMinOp | vmax/vmin | I8~I64/UI/F16/BF16/F32/F8/I1 | **i64** |
| VDivOp | vdiv | （主树默认组） | **i64** |
| VCmpOp | vcmp | I1/F16/BF16/F32/I8~I64/UI/F8（dst 限 I1/I8） | **整型：非 i32 全标量；i32 仅 EQ/NE 安全** |
| VModOp | vmod | I1/I8/I16/I32/I64/F16/F32 | **i64 和 i32** |
| VModUIOp | vmodui | I1/I8/I16/I32/I64 | 未挂接口（无标量路径，按 td 约束使用） |
| VShLOp | vshl | I16/I32/I64 | **i64** |
| VShROp | vshr | I8/I16/I32/I64 | **i64**；A2/A3 移位量为张量时也标量化 |
| VMulExtOp / VMulExtUiOp | vmulext/vmulextui | 仅 I32 | **i32/i64（实际总标量化）** |
| VOrOp / VAndOp / VNotOp | vor/vand/vnot | I1/I8~I64/UI/F16/BF16/F32 | 未挂接口 |
| VXorOp | vxor | 仅整型 I1~I64/UI | 未挂接口 |
| VPowOp | vpow | I8/I16/I32/F16/F32 | 未挂接口 |
| VExp~VLog 族（vexp/vexp2/vln/vlog2/vlog10/vlog1p/vexpm1/vilogb） | — | 仅 F16/F32 | 未挂接口 |
| VReluOp | vrelu | F16/F32/I32 | 未挂接口 |
| VSqrtOp / VRsqrtOp / VRecOp | vsqrt/vrsqrt/vrec | 仅 F16/F32 | 未挂接口 |
| VTanh/VSin/VCos/VAtan/VTan/VErf | — | （td 未列类型约束） | 未挂接口 |
| VLdexpOp | vldexp | F16/F32 | 未挂接口 |
| VSelOp | vsel | 仅 I1/I8 | 未挂接口 |
| VBrcOp | vbrc | I8~I64/UI/F16/BF16/F32/F8/I1 | 未挂接口 |
| VReduceOp | vreduce | I1/I8~I64/UI/F16/F32（无 BF16） | A2/A3：min/max/sum/prod/xori 的 **i64**；with_index 的 **i16/i32/i64** + 高 rank 浮点；950：仅 index 变体几何非法时 |
| VCumsum/VCumprod/VCummax/VCummin | vcum* | I1/I8/I16/I32/I64/F16/F32/BF16 | **i64**（950 cumsum 1D dim0 i64/i32→i64 走 SIMT 库例外）；A2/A3 非 i64 且 cum 维最内时也标量化 |
| VTransposeOp | vtranspose | I1/I8/I16/I32/F16/BF16/F32/I64/UI64/F8 | 未挂接口 |
| VInterleave/VDeinterleave | — | I8~I64/UI/F16/BF16/F32/F8/I1 | **i64** |
| VFlipOp | vflip | I8~I64/UI/F16/F32/BF16 | 未挂接口 |
| VGatherOp | vgather | I1/I8/I16/UI16/I32/UI32/F16/BF16/F32/F8（**无 I64**） | 未挂接口 |
| VGatherMaskOp | vgathermask | I16/UI16/I32/UI32/F16/BF16/F32（**无 I64/I8**） | 未挂接口 |
| VSortOp | vsort | F16/F32/I8/I16/I32/I64 | 未挂接口 |
| VIsInfOp / VIsNanOp | visinf/visnan | F16/F32 | 未挂接口 |
| VCastOp | vcast | 按转换表（td :558-600，如 i64→f16 为 trunc） | 未挂接口 |
| VMulextendedOp | vmulextended | 仅 I16 | 未挂接口 |
| VPadOp / VConcatOp | vpad/vconcat | （td 未列类型约束） | 未挂接口 |

**使用要点**：
1. **i64 是一刀切红线**：算术/比较/归约/累积全族标量化——P2 骨架中索引、长度、
   偏移一律 int32（`tl.arange` 默认 int32，显式 `.to(tl.int32)` 保险）；
2. **整型不等值比较（</>/<=/>=）只有浮点化一条路**：i32 LT/GT 同样标量化，
   这是 §3 官方 fp32 比较范式的源码级依据；
3. gather 族**不接受 I64**——索引张量用 int64 会在更前面报错而非标量化；
4. 浮点数学函数族（exp/log/sqrt/rec/rsqrt）仅 F16/F32——BF16 输入会先 vcast；
5. vsel 仅 I1/I8：where 的 condition 经 vcmp 转成 I1/I8，与 §3 范式一致；
6. "未挂接口"= 该 op 无 ImplByScalarOpInterface 标量兜底，遇到不支持类型通常
   在更前端（HFusion/类型转换）就已处理或报错，而非静默降速。

## 维护记录

- 2026-09-11 初版：[DOC] dtype 表 + [SRC] HIVMVectorOps.td 互证，确认「dialect 接受 ≠ 指令支持」机制。
- 2026-09-11 附录 A：56 个向量 op 全量机械抽取（dialect 层）。
- 2026-09-14 **指令级闭环**：从 `ShouldLowerToScalarLoops.cpp`（主树，含 isRegBasedArch
  分支）抽取全部标量化判定规则，附录 A 标量化列由 [UNVERIFIED] 升级为 [SRC]；
  新增 §2.3（950 V300 指令层：isHIVMSupportedType 仅 i8/16/32+f16/f32，无 i64/bf16；
  向量长度 256/128/64/32 元素）。关键新事实：**i32 不等值比较标量化**（zh 文档只笼统说
  "int32 比较标量化"，源码细化为仅 EQ/NE 安全）；**vmod i32 即标量化**（为 C-001 禁 `%`
  提供源码依据）；950 与 A2/A3 的 vreduce/vcum/vshr 差异点。
- 待办：① "未挂接口"各 op（vor/vand/vpow/vbrc/vgather/vsort 等）在 950 上的指令覆盖度
  未逐一确认（它们无标量兜底，但不等于全 dtype 都有指令）；② 950 上 i64 vreduce 的实际
  路径（模板库 or 前端拦截）待 P4 实测或 HIVMToLLVM 模板符号表确认；
  ③ 注意：zh 官方示例代码中仍出现 kernel 内 `%`（performance_guidelines 示例 2），
  与 backend_contracts C-001（EXP 证据）冲突，按仲裁顺序以 EXP 为准，待源码确认触发条件后修订。
