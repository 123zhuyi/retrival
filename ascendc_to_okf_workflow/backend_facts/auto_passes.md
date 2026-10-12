---
schema_version: okf.v1
kind: runbook
type: auto_passes
source_family: reference
title: 编译器自动行为清单（HIVM pipeline）——"不要手写什么"
description: 从 AscendNPU-IR HIVMPipelines.cpp 实际 pass 序列抽取的编译器自动行为。用途：P2 可达性判定的权威依据——凡编译器自动完成的事，Triton 骨架不应手写，卡片 ⚠️ 项直接引用本表 pass 名。
tags: [hivm, pipeline, passes, auto_optimization, sync, multibuffer, plan_memory, layout]
confidence: verified
status: draft
created_at: "2026-09-11T00:00:00Z"
updated_at: "2026-09-11T00:00:00Z"
---

# 编译器自动行为清单（HIVM pipeline）

> [SRC] 数据源：`AscendNPU-IR/bishengir/lib/Dialect/HIVM/Pipelines/HIVMPipelines.cpp`
> （寄存器基址变体 `Pipelines/regbase/HIVMRegbasePipelines.cpp`）。
> 下表按功能分组，行号为主 pipeline 锚点。凡本表覆盖的行为，卡片可达性判定写
> 「编译器自动（pass 名）」，不再写"惯例推断"。

## 1. 同步 —— 不要手写任何同步

| 自动行为 | pass（锚点） | 相关编译选项 |
|---|---|---|
| 图级同步求解（自动插同步） | `GraphSyncSolverPass`（:61）、`CrossCoreGSSPass`（:90） | `sync_solver` |
| block 同步注入 | `InjectBlockSyncPass`（:99） | `inject_block_all` / `disable_auto_inject_block_sync` |
| 细粒度同步注入 | `InjectSyncPass`（:70） | `inject_barrier_all` |
| sync block 提升/锁参数绑定/锁 lowering/释放 | `SyncBlockHoisting`（:391）、`BindSyncBlockLockArg`（:392）、`SyncBlockLockLowering`（:396）、`InsertFreeLockVarBeforeReturn`（:398） | — |
| FFTS 同步块 | `AddFFTSToSyncBlockSetOp`（:510）；可用 `TRITON_DISABLE_FFTS=1` 关闭 | — |

→ Ascend C 的 `SetFlag/WaitFlag` 事件链、手工 barrier 在 Triton 一律 ❌ 不需要
（mhc_sinkhorn 卡 §7 表已按此判定，现获 pass 级佐证）。

## 2. 多缓冲与流水 —— 不要手写乒乓

| 自动行为 | pass（锚点） | 相关编译选项 |
|---|---|---|
| 自动多缓冲标记 | `MarkMultiBufferPass`（regbase :404/:419/:611；选项接线 HIVMPipelines :275-282） | `multibuffer`、`limit_auto_multi_buffer_*` |
| 预取/预加载生成 | `CreatePreloadPass`（:505；regbase :638） | — |
| CV 流水编排 | `CVPipelining`（pipeline 内接线） | CV 类选项 |

→ Ascend C 的队列 double buffer 在 Triton 对应编译选项 `multibuffer`，不可手写
（UB 不足或存在阻塞依赖时编译器无法使能，见 zh performance_guidelines 三条件 [DOC]）。

## 3. 内存规划与 buffer 尺寸 —— 不要假设地址/尺寸

| 自动行为 | pass（锚点） |
|---|---|
| buffer 尺寸自动推导/常量化/设置 | `AutoInferBufferSize`（:111）、`ConstantizeBufferSize`（:115）、`SetBufferSize`（:116） |
| 内存规划（UB 分配） | `PlanMemoryPass`（:333、:496，两处） |
| 额外 buffer 分配 | `AllocExtraBuffer`（:473） |
| 对齐与 stride 优化 | `AlignAllocSize`（:379）、`MarkStrideAlign`（:381）、`EnableStrideAlign`（:385） |
| workspace 尺寸推导 | `InsertInferWorkSpaceSizeFunc`（:340）、`InsertWorkSpaceForMixCV`（:261）+ `BindWorkSpaceArg`（:262） |

→ 卡片的 UB 预算只做「校核」（对 hardware_platforms.md 平台值），分配本身全自动。

## 4. 布局转换 —— ND/NZ 由编译器管

| 自动行为 | pass（锚点） | 相关编译选项 |
|---|---|---|
| layout 转换插入/传播/合并 | `InsertConvertLayout`（:200）、`PropagateConvertLayout`（:204）、`CombineOptimizedConvertLayout`（:210）、`ConvertLayoutToTranspose`（:211） | — |
| ND→NZ 调试/转换 | `InsertNZ2NDForDebug`（:250）、`InsertFixpipe`（:231/:251）、`InlineFixpipe`（:236/:256） | `enable_nd2nz_on_vector` |

## 5. 分核/tiling/循环 —— CV 融合与混合 kernel 自动处理

| 自动行为 | pass（锚点） | 相关编译选项 |
|---|---|---|
| 子块划分与绑定 | `PartitionAndBindSubBlock`（:311）、`TileAndBindSubBlock`（:361） | `enable_auto_bind_sub_block` |
| Cube/Vector loop tiling | `TileCubeVectorLoop`（:320） | `tile_mix_cube_loop` / `tile_mix_vector_loop` |
| 混合 kernel 拆分（CV/SIMT） | `SplitMixKernel`（:356）、`SplitSimtModule` | — |
| 混合 CV load/store 插入 | `InsertLoadStoreForMixCV`（:190/:194）、`InsertLoadStoreForScalar`（:195） | — |
| forall→block 映射 | `MapForToForall`（:405）+ `HIVMMapForallToBlocks`（:406） | — |
| LICM/子集提升 | `LoopInvariantCodeMotion`/`LoopInvariantSubsetHoisting`（:368-369） | — |
| 标量化兜底 | 各 op 的 `ImplByScalarOpInterface`（见 dtype_support.md） | — |

## 6. 规范化与收尾（背景知识，勿依赖其挽救坏骨架）

`NormalizeMatmul`（:225/:249）、`NormalizeConvOps`（:226）、`NormalizeBitwiseSelect`（:227）、
`FlattenOps`（:462）、`HIVMDecomposeOp`（:408/:416/:501）、`HIVMLowerToLoops`（:499）、
`InferHIVMMemScope`（:414/:475/:503）、`InferHIVMDataLayout`（:449）、多轮
`ExtendedCanonicalizer` + `CSE`（:43-49, :460）。

→ 规范化 pass 能清理冗余，但**不能**把 T4 慢路径（tl_op_lowering.md）变好；
骨架层面的坏写法（离散 mask、`%`、T4 op）必须在 P2 避免。

## 7. 卡片可达性判定引用格式（推荐）

```
| Ascend C 手段 | 可达性 | 依据 |
| 队列 double buffer | ⚠️ 编译器自动 | auto_passes §2 MarkMultiBuffer；选项 multibuffer |
| SetFlag/WaitFlag | ❌ 不需要 | auto_passes §1 GraphSyncSolver/InjectSync |
| UB buffer 声明 | ⚠️ 仅做预算校核 | auto_passes §3 PlanMemory；hardware_platforms 平台 UB 值 |
```

## 维护记录

- 2026-09-11 初版：从 HIVMPipelines.cpp 全量 addPass 序列（:41-:510）分组抽取，关键 pass 与 17 个编译选项（compiler_options.yaml）交叉挂接。
- 待办：regbase pipeline（HIVMRegbasePipelines.cpp）与主 pipeline 的 pass 差异未逐条比对；各 pass 的触发条件（option 默认值下的实际开关状态）需结合 PassPipeline.cpp 默认值再核。
