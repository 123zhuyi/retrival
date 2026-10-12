---
schema_version: okf.v1
kind: runbook
type: extension_apis
source_family: curated
title: Triton-Ascend 扩展 API（al/bl）速查——"❌不可达"的官方降级路径
description: 昇腾扩展编程接口 al（triton.language.extra.cann.extension）与 bl（triton.extension.buffer.language）全 16 个 API 的用途、约束与适用场景。P2 可达性判定时：凡 Ascend C 手段判 ❌，先查本卡是否有对应扩展 API，有则降级路径写它而非"放弃"。
tags: [triton_ascend, extension_api, al, bl, buffer, sync_block, fixpipe, scope, address_space]
confidence: inferred
status: draft
created_at: "2026-09-11T00:00:00Z"
updated_at: "2026-09-11T00:00:00Z"
---

# Triton-Ascend 扩展 API（al/bl）速查

> [DOC] 数据源：zh `triton_api_extention/al/`（11 篇）、`bl/`（5 篇）。
> 导入：`import triton.language.extra.cann.extension as al`、
> `import triton.extension.buffer.language as bl`。
> ⚠️ 这些是**专家级细粒度控制接口**，普通算子不需要；仅当标准 Triton 语义不够时使用。
> 标注 [A5] 的为 A5（950 代）硬件特性，A2/A3 不可用。

## 1. 内存空间与 buffer（bl 族 + address_space）

| API | 作用 | 关键约束 |
|---|---|---|
| `al.ascend_address_space.{UB,L1,L0A,L0B,L0C}` | 地址空间枚举（对接 `hivm::AddressSpace`） | 配合 `bl.alloc` 使用 |
| `bl.alloc(etype, shape, address_space, is_mem_unique)` | 手动在指定地址空间分配 buffer（对接 `memref.alloc`） | 返回 `bl.buffer`，与 `tl.tensor` 语义隔离，不可互赋值 |
| `bl.to_buffer(tensor, space, bind_buffer)` | tensor → buffer 转换；`bind_buffer` 复用已有 buffer | 传了 bind_buffer 则返回其本身 |
| `bl.to_tensor(buffer, writable=True)` | buffer → tensor；`writable=True` 允许 bufferization 原地修改 | 约束同 `bl.alloc` |
| `al.subview(src, offsets, sizes, strides)` [A5] | buffer 上定义新视图（偏移/大小/步幅），**零拷贝** | src 为 bl.buffer |

**典型降级对应**：Ascend C 的 `TBuf`/`TQue` 显式 buffer 管理 → `bl.alloc` + `to_tensor/to_buffer`；
Ascend C 的 LocalTensor 视图切片 → `al.subview`（此前标 ❌ 的可在 950 改判 ⚠️）。

## 2. 数据搬运

| API | 作用 | 关键约束 |
|---|---|---|
| `al.copy(src, dst)` | UB→L1 或 UB→UB 复制 | src/dst 须同类型同形状；src 必须在 UB，dst 在 L1 或 UB；tensor 暂不支持（用 buffer） |
| `al.copy_from_ub_to_l1(src, dst)` [A5, **已废弃**] | UB 直写 L1，免 GM 往返 | 用 `al.copy` 替代 |
| `bl.fixpipe(src, dst, dma_mode, dual_dst_mode, pre_quant_mode, pre_relu_mode)` [A5] | **L0C→UB 数据通路**（Cube 结果搬回） | 仅 L0C→UB；src 必须是 `tl.dot` 结果；dst 必须 UB buffer；`dma_mode` ∈ NZ2DN/NZ2ND/NZ2NZ；支持搬运中量化（F32→BF16/F16、S32→I8）和预激活（ReLU/LeakyReLU/PReLU） |

**典型降级对应**：Ascend C 的 `Fixpipe`/`DataCopy L0C→UB` → `bl.fixpipe`（Cube 类/CV 融合算子的关键通路）；
搬运中量化+激活融合可替代单独的 cast/relu pass。

## 3. 核间/流水同步

| API | 作用 | 关键约束 |
|---|---|---|
| `al.sync_block_all(mode, event_id)` | 全局核间屏障，解决 GM 读后写/写后读/写后写依赖 | mode ∈ `all_cube`/`all_vector`/`all`/`all_sub_vector`；event_id ∈ [0,15] |
| `al.sync_block_set(sender, receiver, event_id, sender_pipe, receiver_pipe)` | 分离模式 flag 置位（计数器+1） | sender/receiver 仅 `"cube"`/`"vector"`；与 sync_block_wait 配对 |
| `al.sync_block_wait(...)` | 等待 flag（计数器 0 则阻塞，>0 则 -1 放行） | 参数同 set；PIPE 枚举：S/V/M/MTE1/MTE2/MTE3/ALL/FIX |
| `al.debug_barrier(sync_mode)` | VF 场景手动同步（调试用） | `SYNC_IN_VF` 枚举 12 种模式（VV_ALL/VST_VLD/…） |

**典型降级对应**：Ascend C 的 `SetFlag/WaitFlag` 事件链 → **首选编译器自动同步**
（auto_passes §1，GraphSyncSolver/sync_solver 选项）；仅当自动同步不满足（跨核 GM 依赖、
自定义流水）才用 `sync_block_set/wait`。**不要默认手写同步**——写错即死锁。

## 4. 执行模式控制

| API | 作用 | 关键约束 |
|---|---|---|
| `al.scope("vector" \| "cube")` | 显式指定代码块在 Vector/Cube 核执行 | 每 kernel 最多各 1 个 cube/vector scope；两 scope 内操作**并行执行**，跨 scope 数据依赖须用同步 API 显式声明 |
| `al.sub_vec_id()` | 返回 Vector 子核 ID [0,N)，手动切分 N 个 Vector 核的数据 | **仅 CV 混合场景有效**；纯 Cube/纯 Vector 算子使用会编译报错 |

**典型降级对应**：Ascend C 的 AIC/AIV 分工编程 → `al.scope` + `al.sub_vec_id` +
`sync_block_set/wait`（对应 zh Extension_Ops 的 `parallel`/`sync_block`，CV 融合算子的
手动控制面）。编译器自动方案（`enable_auto_bind_sub_block` 等 CV 选项）优先，手动 API 是兜底。

## 5. P2 使用决策树

```
Ascend C 手段判 ❌ 不可达？
 ├─ 是 buffer/视图管理     → bl.alloc / al.subview [950]
 ├─ 是 L0C→UB 搬运/量化    → bl.fixpipe [950]
 ├─ 是 UB→L1 直通          → al.copy [950]
 ├─ 是手工事件链/跨核同步   → 先 auto_passes §1 自动同步；不够再 sync_block_set/wait
 ├─ 是 AIC/AIV 显式分工     → 先 CV 编译选项；不够再 al.scope + sub_vec_id
 └─ 以上都无               → 才写"不可达 + 性能影响"（并考虑回流本卡）
```

## 维护记录

- 2026-09-11 初版：al 11 篇 + bl 5 篇全量蒸馏。注意多篇文档标 [A5] 的接口为 950 代硬件特性，
  卡片引用时须核对目标平台；`copy_from_ub_to_l1` 已被 `al.copy` 取代。
- 待办：al/bl 各 API 在 A2/A3 的可用性未逐条标注（文档未明示的按 950-only 保守处理）。
