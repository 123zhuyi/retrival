# P2 语义转化 — add（Ascend C 事实 → Triton 决策）

> 依据：`../ascendc_triton_semantic_mapping.md`。每条给出：Ascend C 事实（锚点）→ Triton 决策 → 可达性。
> **核心结论先行**：本算子是纯广播 elementwise，语义 = `out = x + alpha·y`（aclnn 层），GEIR kernel 本体 = `x1 + x2`。Ascend C 侧的关键工程（BroadcastSch 调度、aclnn 的 Axpy/Mul+Add 四策略分解）在 Triton 中**全部不需要**：单个 program 顺序处理连续输出段，alpha 以 runtime 标量融入 kernel（优于 aclnn 的 Mul+Add 两 kernel 兜底路径），广播用展开 stride 地址折算。真正需要保留的语义资产只有两条：**fp32 中间精度 + RTNE 回写**（add_dag.h:48-51）、**输出展平分核 + 尾块 mask**。

## 1. 分核策略 → grid

| Ascend C 事实 | Triton 决策 | 可达性 |
|---|---|---|
| coreNum = GetCoreNumAiv()（add_tiling_arch35.cpp:182）；官方分核（op_common 源码 broadcast_tiling.cpp:494-607）：maxElemNum(BLOCK) → UB 内切分（ubFormer/ubOuter/ubTail）→ `blockFormer = ceil(fusedProduct/coreNum)`、`blockNum`、`blockTail` | 输出展平 1-D，`NUM_BLOCKS = cdiv(NUMEL, BLOCK)`；grid = AIV 物理核数 + `for blk in range(pid, NUM_BLOCKS, NUM_CORE)`（与前两张卡同一范式）。语义等价：Ascend C「每核 blockFormer 块 + 尾核 blockTail 块」静态均分 ≈ Triton「grid-stride 循环 + 尾块 mask」动态轮询，负载分布等价 | ✅（官方机制已对照，非惯例推断） |
| **preferMultiCore 回调**（broadcast_tiling.cpp:569-586）：blockNum < coreNum 时按 CACHE_LINE 步长缩小 maxElemNum 直到铺满核 | autotune 空间覆盖该语义：BLOCK 候选 {1024..8192}，BLOCK 过大→核空闲由小 BLOCK config 胜出；卡片 §10 注明「核数占用不足时优先降 BLOCK」 | ✅（策略等价） |
| schMode tilingkey（add_struct.h:21-23）→ innerKey = computeKey(恒1)×OFFSET + scheduleKey（bt:604-605）；scheduleKey 只区分 UB 内轴数 ≤5 与否（100/1100，broadcast_tiling_base.cpp:28-34） | 自定义 `SAME_SHAPE: tl.constexpr`：1 = 等形直通，0 = 通用广播（stride 折算）；wrapper 侧 shape 分析静态选定。原 key 的 NDDMA 5 维上限对 Triton 无意义 | ✅（原 key 语义已破译，仅 2 态且与 Triton 无关） |
| 连续输入合轴 DimensionCollapse（bt:151-269）/ 非连续 KEEP-MERGE-ABSORB 计划（bt:271-333, 344-492），合轴后 ≤8 维（bt:632-634） | wrapper 侧 `expand` + stride 折算天然覆盖合轴语义（等形 case 合轴后 = SAME_SHAPE 直通）；Triton 无需显式合轴 | ✅（语义替代） |
| 尾块单独参数 blockTail（bt:588） | `offs < NUMEL` mask | ✅ |

## 2. dtype 分派 → 单 kernel

| Ascend C 事实 | Triton 决策 | 可达性 |
|---|---|---|
| 4 路 constexpr 分派（add.cpp:35-50）：WithCast / MixDtype / WithoutCast / Bool | **单 kernel**：benchmark 域内 x/y/out 同型 fp32/fp16/bf16，统一 `load → .to(tl.float32) → alpha·y + x → .to(out_dtype) → store`；fp32 时两次 cast 为恒等（对应 add_dag.h 中 Cast<float,float>），无需分支 | ✅ |
| MixDtype（fp16×fp32→fp32，add_dag.h:60-74） | benchmark 无此 case；如需支持：wrapper 侧 promote 后走同一路径，out dtype=fp32 天然正确 | ⚠️（保留语义，暂不实现） |
| Bool 的 Mins(1)（add_dag.h:83） | benchmark 无；int dtype 同理。如需：`tl.minimum(a+b, 1)` | ⚠️（保留语义，暂不实现） |
| 整型 Wrap（WithoutCast，add_dag.h:30-41） | benchmark 无 | ⚠️ 暂不实现 |

## 3. alpha 消化 → 融合进 kernel

| Ascend C 事实 | Triton 决策 | 可达性 |
|---|---|---|
| aclnn 四策略：alpha==1→Add；Axpy 融合；AxpyV2；Mul+Add 两 kernel（aclnn_add.cpp:371-466） | **一律单 kernel**：`alpha` 作 runtime fp32 标量传入，`out = x_f32 + alpha * y_f32`。alpha==1 无分支（乘 1.0 无额外成本），比 aclnn 兜底路径少一次 GM 往返 | ✅（结构性更优，勿复刻四策略） |
| alpha cast 到 promoteType 校验（aclnn_add.cpp:212-223） | wrapper 侧 `float(alpha)` 断言 | ✅ |
| ⚠️ 数值注意：alpha 与 y 先乘后加 x（与 torch.add 一致：`self + alpha*other`），不是 `(x+y)·alpha` | 公式顺序照抄 torch | ✅ |

## 4. 广播 → stride 折算

| Ascend C 事实 | Triton 决策 | 可达性 |
|---|---|---|
| CopyInBrc：DMA 按广播形态展开搬入 UB（base 库） | wrapper：`out_shape = broadcast_shapes(x, y)`；`xe = x.expand(out_shape)` 取各维 stride（广播维=0）；kernel 内输出线性 offs 逐维 div/mod 分解坐标 → `in_off = Σ coord_d · in_stride_d` | ✅ |
| 连续直通情形（两输入同形连续，如 [4096,8192] 等形大 case） | `BROADCAST_MODE=0`：offs 直通，零整数除法 | ✅ |
| 通用广播（[M,1] / [1,N] / [N] / [1,C,1,1] / 4D×2D） | `BROADCAST_MODE=1`：NDIM≤4 constexpr 展开（benchmark 最大 4 维；ST 有 6~8 维形态，扩 NDIM 上限即可）；整数 div 每元素每维一次 | ✅ |
| 优化备注（非语义）：最内维 stride=0 的输入（如 [M,1]）按通用路径会对同一地址重复 load；优化版可按「外维坐标 × 1 元素 + tl.broadcast_to」加载最内维 1 宽 tile 再展开，省 L2/GM 带宽 | 记入卡片 §10，骨架保持通用正确版 | ⚠️ |

## 5. tilingkey / tilingdata → constexpr

| Ascend C | Triton | 备注 |
|---|---|---|
| tilingkey = schMode（add_tiling_arch35.cpp:90） | `BROADCAST_MODE: tl.constexpr`（语义重定义见 §1） | — |
| AddCompileInfo{coreNum, ubSize}（add_tiling_arch35.h:31-34） | 无对应（grid 取物理核数，UB 无需预算） | — |
| 13 个 binary 实例（add_binary.json） | 0 个特化 kernel（dtype 由指针类型推导，编译期自然分派） | Triton 侧大幅简化 |

## 6. 片上内存 → 预算

- Ascend C：UB 队列 LEVEL_2 进出（add_dag.h:39 等），无 GM workspace（tiling L143-146）
- Triton：BLOCK×3 tile（x/y/out），BLOCK=8192 fp32 时 96KB ≤ 192KB UB ✅；workspace = 0

## 7. 流水编排 → 可达性

| Ascend C 手段 | 可达性 | Triton 处理 |
|---|---|---|
| BroadcastSch/DAGSch 自动排布（base 库） | ❌ | 不需要：load→计算→store，Triton 编译器自动流水 |
| CopyInBrc 广播 DMA 展开 | ❌→替代 | stride 地址折算（§4） |
| Contiguous 中转（aclnn_add.cpp:358-364） | ❌→不需要 | wrapper 断言 contiguous（benchmark 输入天然连续）；非连续输入先 `.contiguous()` |
| Cast/ViewCopy 输出中转（aclnn_add.cpp:470-476） | ❌→不需要 | 直写 out |
| aclnn 四策略分解（Mul+Add 两 kernel） | ❌→不需要 | 单 kernel 融合（§3） |

## 8. 精度策略 → 验收

- 中间 fp32 + RTNE 回写（add_dag.h:48-51）：Triton `.to(tl.float32)` 计算、`.to(out_dtype)` 回写即等价（RTNE 为 Triton cast 默认舍入）
- benchmark 大数值 case（|x| 至 ~±175，alpha·y 叠加）：fp16/bf16 溢出属 golden 同行为（torch 同样 fp32 中间精度），阈值判据按 evaluator 默认（fp32 rtol=2⁻¹⁰/atol=2⁻¹⁶/max_abs 1e-2；fp16/bf16 2⁻⁹/2⁻⁶ 档）
- 累加序无关（无归约），无确定性顾虑

## 9. P2 质量门

- extraction.md 事实全部有去向：转化 8 条、结构性简化 5 条（BroadcastSch/DAGSch/CopyInBrc/Contiguous 中转/四策略分解）、保留语义 3 条（fp32 中间精度+RTNE、广播坐标折算、alpha 先乘后加）✅
- 骨架见 `triton_skeleton.py` ✅
- 进 P3
