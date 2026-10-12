---
schema_version: okf.v1
kind: runbook
type: tl_op_lowering
source_family: curated
title: tl op → 后端 lowering 路径与代价分级
description: Triton language op 在 Triton-Ascend SIMD 编译链（triton-to-structured → triton-to-unstructured → triton-to-linalg → triton-to-other）中的 lowering 去向与代价分级。P2 骨架选型时逐 op 核对：优先 T1 原生 op，T3/T4 需显式权衡并在卡片中标注。
tags: [triton_ascend, lowering, tl_op, linalg, scalarization, gather, scan, cost_model]
confidence: inferred
status: draft
created_at: "2026-09-11T00:00:00Z"
updated_at: "2026-09-18T00:00:00Z"
---

# tl op → 后端 lowering 路径与代价分级

> 数据源：[DOC] zh `architecture_design_and_core_features.md` §3.2.2（SIMD compiler 各 pass converter 全表）；
> [SRC] `AscendNPU-IR/bishengir/triton/lib/Conversion/`（TritonToTritonGPU/TritonGPUToLLVM，SIMT 路径）。
> 代价分级为本卡归纳（DOC 未给分级），有 [SRC]/[EXP] 支撑处单独标注。

## 1. 代价分级定义

| 级 | 含义 | 生成建议 |
|---|---|---|
| **T1 原生** | 1:1 映射为 linalg/memref/tensor 结构化 op，后续进 HIVM 向量/cube 指令 | 放心用 |
| **T2 结构化** | 映射为 linalg named/generic op（reduce/broadcast/matmul 等），后端有成熟 lowering | 放心用，注意 dtype（见 dtype_support.md） |
| **T3 循环展开** | 展开为 scf 循环或多 op 序列，有额外搬运/标量化风险 | 可用但需在卡片标注性能不确定性 |
| **T4 func-call / 慢路径** | 降到 `func::CallOp`/`func::FuncOp`（黑盒函数调用），脱离结构化优化流 | **避免**；必须用时给替代方案 |

## 2. triton-to-linalg converter 分级表 [DOC]

### T1/T2（骨架主力 op）

| tl op | lowering 去向 | 级 |
|---|---|---|
| `tl.load` | `memref.copy` + `bufferization.to_tensor` | T1 |
| `tl.store` | `memref.copy` | T1 |
| `tl.dot` | `linalg.matmul` | T2（Cube） |
| `tl.sum` / `tl.max` / `tl.min` / reduce 族 | `linalg.reduce` | T2 |
| `tl.argmax` / `tl.argmin` | `linalg.reduce` | T2 |
| `tl.broadcast_to` | `linalg.broadcast` | T2 |
| `tl.trans` | `linalg.transpose` | T2（⚠️ 分态见下注） |
| `tl.reshape` | `tensor.reshape` | T1 |
| `tl.expand_dims` | `tensor.expand_shape` | T1 |
| `tl.split` / `tl.join` / `tl.cat` | `tensor.extract_slice` / `insert_slice` | T1 |
| `tl.arange`（make_range） | `linalg.generic` | T2 |
| `tl.splat` / 常量 | `linalg.fill` | T2 |
| `extern_elementwise`（libdevice 数学函数） | `linalg.map` | T2 |
| `tl.atomic_add` 等 RMW | `linalg.generic` | T2 |
| `tl.atomic_cas` | `linalg.generic` | T2 |
| `scf.for` / `scf.while` / `yield` | 透传 | T1 |
| 指针算术（`addptr`/`advance`） | `memref.reinterpret_cast` | T1 |
| `precise_div/sqrt`、`mulhiui`、`clampf`、`bitcast` | arith/math 直译 | T1 |
| `tl.rsqrt` | arith/math 直译→vrsqrt（硬件 F16/F32 存在） | T1（本栈实测可译可执行，见维护记录 2026-09-18） |

### T3（可用但标注）

| tl op | lowering 去向 | 风险 |
|---|---|---|
| `tl.gather` 的 load 变体（GatherLoad） | `scf.for` 循环 | 逐元素 gather，吞吐随规模线性退化 |

> **`tl.trans` 分态注**（2026-09-15 [EXP]）：小 tile（≤64×128 量级）寄存器级 trans 实测
> **优于**等效标量跨步访存 2.4-2.8x（chunk_gated_delta_rule_fwd_h A2 实验，svf 路径）；
> 外部 runbook 的「trans 标量化 ~15x」证据来自**大 tile**。按 tile 大小两态实测，
> 勿一刀切禁用（详见 `operator_class_patterns.md` §4.2 svf 两法）。

### T4（避免）

| tl op | lowering 去向 | 替代 |
|---|---|---|
| `tl.scan`（cumsum 等） | **`func::CallOp`** | 手写循环 + 规约重写，或换算法（如分块前缀和） |
| `tl.gather` | **`func::FuncOp`** | 离散索引改 `tl.load` + 索引张量；或参考扩展 API `index_select_simd`（非尾轴并行 gather，GM→UB 零拷贝 [DOC]） |
| `tl.device_assert` | `func::FuncOp` | 调试用，交付代码删除 |
| `tl.device_print` | `func::FuncOp` | 调试用（需 `TRITON_DEVICE_PRINT=1`，每线程 GM 缓冲 16KB 上限 [DOC]），交付删除 |

## 3. 前置 pass 对骨架写法的硬约束 [DOC]

骨架写法若踩以下限制，op 会在 triton-to-structured / triton-to-unstructured 阶段被改写或退化：

1. **指针/mask 表达式含 `//`、`%`**：triton-to-structured 通过升维消除，但要求
   ① 原始迭代轴能被分裂轴整除；② 外层 BLOCK 大小是 divisor 的整数倍或约数。
   不满足 = 改写失败（叠加 backend_contracts C-001 的 `%` 禁令，骨架应尽量无整除取余）。
2. **离散 mask（非连续索引访问）**：经 discrete-mask-access-conversion 改写为
   load+select+store（多一次往返），再经 triton-to-unstructured **展开为标量循环**——
   性能悬崖。离散访问场景（scatter/gather 类）必须在卡片显式预警并给 tile 化替代。
3. `while` 循环体内指针叠加：不支持循环体内含条件分支（`if`）的复杂指针路径变换。

## 4. SIMT 路径备注 [SRC]

`bishengir/triton`（TritonToTritonGPU → TritonGPUToLLVM）是 SIMT 编译路径
（`TRITON_ENABLE_LIBDEVICE_SIMT=1`，见 zh libdevice 手册），与本卡默认的 SIMD 路径
（triton-to-linalg）是**两套 lowering**。P2 默认按 SIMD 路径分级；若算子明确走 SIMT，
本卡分级不适用，需另行核对。

## 5. 使用方式（接入 P2/P3）

- P2 骨架写完，逐 op 查本表标级；凡 T3/T4 在 mapping.md「lowering 路径」列写明理由与替代；
- P3 卡片「Triton 骨架」节末尾附一句分级摘要，如「全 T1/T2，无慢路径」或「含 T4：tl.gather（已给 index_select_simd 替代）」；
- 新 op 类型的分级不在表内时，先查 zh converter 表，仍无则标「未经后端确认」。

## 维护记录

- 2026-09-11 初版：按 zh §3.2.2 converter 表全量录入并分级。
- 2026-09-15 `tl.trans` 加分态注（小 tile 寄存器级实测可用 [EXP A2]，大 tile 才有
  标量化风险——外部 runbook 的一刀切禁令按 tile 大小收窄）。
- 2026-09-18 `tl.rsqrt` 实测 [EXP-skeleton]（layer_norm ①门，910_9372+CANN 9.1.0）：
  编译+triton 执行通过，`tl.rsqrt(v+eps)` 对照 `1/torch.sqrt` maxerr=1.19e-7 正确。
  主路径 layer_norm 骨架用 `1/tl.sqrt`（sqrt T1 更保守），`tl.rsqrt` 为经证可用的等语义备选。
- 待办：T3/T4 的量化代价（吞吐退化倍数）需 P4 实测补充；libdevice 函数清单（SIMT 可用数学函数）未录入。
