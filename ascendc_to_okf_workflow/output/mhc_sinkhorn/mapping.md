# P2 语义转化 — mhc_sinkhorn（Ascend C 事实 → Triton 决策）

> 依据：`../ascendc_triton_semantic_mapping.md`。每条给出：Ascend C 事实（锚点）→ Triton 决策 → 可达性。
> **核心结论先行**：本算子 AIV-only、n≤8、全程 fp32，是 Triton 友好型。Ascend C 中最复杂的 Reg gather/scatter 布局转换在 Triton 中**整体不需要**——直接用 `[BLOCK_T, 8, 8]` padding tile + `tl.sum(axis=)` 表达行列规约（详见 §6）。

## 1. 分核策略 → grid

| Ascend C 事实 | Triton 决策 | 可达性 |
|---|---|---|
| T 轴切分，tNormCore=CeilDiv(T, AIV核数)，SetBlockDim(usedCoreNum)（base_tiling.cpp:125-127; tiling.cpp:327） | grid = AIV 物理核数；`NUM_BLOCKS = CeilDiv(T, BLOCK_T)`，核内 `for blk in range(pid, NUM_BLOCKS, NUM_CORE)`。BLOCK_T 即 tUbFactor 的 Triton 对应物 | ✅ |
| 正常核/尾核二分（tUbFactorTail/tUbTailTail） | 不二分，统一 `mask = offs_t < T` 处理尾块 | ✅（简化） |
| host 侧标量（n、num_iters、eps、T） | wrapper 计算后作为 kernel 参数传入 | ✅ |

## 2. tilingdata → BLOCK 超参

| tilingdata | Triton 对应 | 候选值 / 推导 |
|---|---|---|
| tUbFactor（UB 预算反推，8 对齐） | `BLOCK_T` | n=8 时单片 4·align(64)+…≈1.2KB(fp32)，UB 192KB 预算下 BLOCK_T 可远大于 8；autotune 候选 {32, 64, 128}，key=["T","N"] |
| n ∈ {4,6,8}，nAlign=8 | `N: tl.constexpr`（运行时值）+ `BLOCK_N = 8`（constexpr，padding 到对齐值） | mask `offs_n < N` 实现 4/6/8 三档 |
| tNormCoreLoop / tail 字段 | 循环边界 + mask | 全并入 `NUM_BLOCKS` 循环 |
| out_flag ∈ {0,1} | `OUT_FLAG: tl.constexpr` | kernel 内 `if OUT_FLAG:` 编译期裁剪，对应 TEMPLATE_MODE 0/1 |
| num_iters, eps | kernel 标量参数（eps 用 fp32 传） | 循环 `for it in range(num_iters)` |

## 3. tilingkey → constexpr

`tilingkey = out_flag`（tiling.cpp:308-315）→ 单 kernel + `OUT_FLAG: tl.constexpr` 即可，两个 TEMPLATE_MODE 实例合并。**决策**：out_flag=1 的调试输出路径用 constexpr 分支内联在同一 kernel（差异仅是若干次 store），无需拆两个 kernel。

## 4. 片上内存 → 预算校核

Ascend C 五队列双缓冲（arch35.h:102-115）→ Triton 无显式队列，等效 tile 预算：

| Ascend C buffer | Triton 等效 | 字节数（BLOCK_T=64, n=8, fp32） |
|---|---|---|
| inputQue_ (t·n·n) | load tile `[BLOCK_T, 8, 8]` | 64·64·4 = 16KB |
| normOutQue_ (t·n·nAlign) | 中间张量 m `[BLOCK_T, 8, 8]`（寄存器/tile 驻留，全程不落地） | 16KB |
| sumColQue_/sumRowQue_ | 规约中间 `[BLOCK_T, 8]` | 2KB×2 |
| outputQue_ | store 前与 m 复用 | ~0 |
| mask/mid/sum buffer (384B) | 无对应 | 0 |
| **合计** | | **~34KB ≪ 192KB UB** ✅ 有大幅富余，BLOCK_T 可到 128 |

关键差异：Ascend C 把 norm 中间结果放 normOutQue_（UB 队列）是因为迭代间要复用且需转置布局；Triton 中迭代直接作用在寄存器 tile `m` 上，**num_iters 轮迭代零搬运**——这是相对 Ascend C 实现减少 MTE 启动次数的结构性优势。

## 5. 流水编排 → Triton 可达手段

| Ascend C 手段 | 可达性 | Triton 处理 |
|---|---|---|
| CopyIn/Compute/CopyOut 三段 + 全队列 double buffer（arch35.h:107-115） | ⚠️ | 交给编译器：`for blk` 循环天然形成 load/compute/store 流水；Triton-Ascend 侧可开 `multibuffer=True`。卡片只写约束「批间需要搬运/计算掩盖」 |
| 循环内 SetFlag/WaitFlag(MTE3_V) 等 CopyIn 完成（arch35.h:482-484 等 4 处） | ❌ | 丢弃，编译器自动插同步。这些事件是因为 out_flag=1 时把 norm 中间结果搬出 GM 才需要的；out_flag=0 时纯计算无此依赖 |
| 迭代间数据零搬运（normOutQue_ 原地更新 2·num_iters 次） | ✅ | Triton 天然满足（寄存器 tile 迭代），且更彻底 |

## 6. Compute → Triton 计算骨架（核心转化）

| Ascend C 步骤（锚点） | Triton 表达 | 可达性 |
|---|---|---|
| CopyIn: DataCopyPad 连续搬 t·n·n（arch35.h:125-135） | `tl.load(x_ptr + offs_t[:,None,None]·n·n + offs_r[None,:,None]·n + offs_c[None,None,:], mask=三维 valid, other=0.0)` | ✅ |
| rowIndexReg 合成 + `DataCopyGather` 转置读（arch35.h:195-215） | **不需要**：直接以 [BLOCK_T, 8, 8] tile 承载，行列规约用 axis 区分 | ❌→语义等价替代（布局转换是向量效率手段，非语义） |
| CalcSoftmax: ReduceMax→Sub→Exp→ReduceSum→Div（arch35.h:216-229） | `mx = tl.max(m, axis=2)`; `e = tl.exp(m - mx[:,:,None])`; `m = e / tl.sum(e, axis=2)[:,:,None]`（axis=2 = 行内） | ✅ |
| softmax 阶段累和 sumoutReg = Σ(M⁰+eps)（arch35.h:233-234） | `sc = tl.sum(m + eps, axis=1)`（axis=1 = 列方向求列和） | ✅ |
| CalcColNorm: Adds(eps) + Div（arch35.h:288-300） | `m = m / (tl.sum(m, axis=1) + eps)[:,None,:]` | ✅ |
| CalcRowNorm: 行和 +eps 再 Div（arch35.h:358-373） | `m = m / (tl.sum(m, axis=2) + eps)[:,:,None]` | ✅ |
| 规约经 UB 中转广播（midBuffer/sumBuffer + LocalMemBar，arch35.h:216-229） | **不需要**：`tl.sum/tl.max` 结果直接 broadcast | ❌→语义等价替代 |
| ScatterOutFromNorm: `DataCopyScatter` 转置写回（arch35.h:426-435） | **不需要**：`tl.store(y_ptr + 同 load 的 offsets, m, mask=valid)` | ❌→语义等价替代 |
| out_flag=1 时 2·num_iters 轮 norm/sum 搬出（arch35.h:475-543） | `if OUT_FLAG:` 每轮 `tl.store(norm_ptr + it·stride + ..., m, ...)` 与 `tl.store(sum_ptr + ...)`，布局 [2·num_iters, T, n, 8] / [2·num_iters, T, 8] 按 infershape 公式写 | ✅ |
| StoreUnAlign 非对齐 sum 写回（arch35.h:362-367） | 普通 `tl.store` + mask | ✅（简化） |
| mask 三段式（lane mask ∧ n 有效位） | 统一 `mask = (offs_t < T) & (offs_r < N) & (offs_c < N)` 广播 | ✅（简化） |

### 不可达项汇总（写入卡片 §7 与 §10）

1. ❌ Reg gather/scatter 布局转换 → 替代：padding tile + axis 规约（**性能预期不低于原实现**：n≤8 时 gather 本就只为寄存器排布，Triton 编译器可自行向量化 8 对齐 tile）
2. ❌ UB 规约中转广播（LocalMemBar 往返）→ 替代：tl.sum 直接广播
3. ❌ 手工 MTE3_V 事件链 → 编译器自动同步
4. ⚠️ 队列双缓冲 → multibuffer 编译开关 + 循环结构保证

## 7. 精度策略 → 验收

- 全程 fp32，无 cast 点；eps 防除零必须保留（4 处语义点：softmax 列和、每次行/列归一分母）
- 标杆（P4 自建）：torch fp64 参考实现同一迭代公式 → 双标杆；判定 `rtol=1e-5, atol=1e-6`（fp32 计算 + 迭代误差累积，初值，回放验证时可调）
- 额外算子级判定：输出 y 每矩阵行和、列和与 1 的最大偏差 < 1e-4（双随机性直接校验）

## P2 质量门

- extraction.md 全部 tiling/流水/计算事实均有去向：转化 12 条、合并简化 4 条、显式丢弃 3 条（均注明理由）✅
- 骨架见 `triton_skeleton.py` ✅
