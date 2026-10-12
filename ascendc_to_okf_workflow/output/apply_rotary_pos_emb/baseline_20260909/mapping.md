# P2 语义转化 — apply_rotary_pos_emb（Ascend C → Triton）

> 输入：extraction.md。每条 P1 事实在此有去向（转化/合并/显式丢弃+理由）。
> 可达性三态：✅ 直接可达 / ⚠️ 编译器自动或需权衡 / ❌ 不可达（含降级方案）。

## 0. 总体转化决策：4 条 tiling 路径 → 单 kernel + constexpr

arch35 四个 kernel 类（AB/BAB/ABAAndBA）**计算数学完全相同**（都调 common.h 的 Half/Quarter/Interleave VF），差异只在：① 循环结构（BS 打平 vs B×S 二维）；② cos/sin 广播是否载一次；③ Q/K 边界处理。这三类差异在 Triton 中全部由「strides 参数化 + constexpr 分支」吸收，不需要 4 个 kernel——**单 kernel + `ROTARY_MODE: tl.constexpr` + 布局 strides**。layout 差异（BSND/SBND/BNSD/TND）不进 kernel 分支，由 wrapper 算出 (stride_b, stride_s, stride_n) 三个元素步长传入（TND 时 stride_b=0、b=1）。

## 1. 分核策略 → grid 设计

| Ascend C 事实 | Triton 决策 | 可达性 | 等价论证 / 依据 |
|---|---|---|---|
| AB：BS 打平 `blockFactor=CeilDiv(bs,coreNum)`、SetBlockDim(blockNum)（tiling_ab L69-71,112）；ABA/BA：B×S 二维分核（tiling_aba L80-135,262）；BAB：usedCoreNum×ubFactorS 比选（tiling_bab L105-149,233） | **行（row=一个 (b,s,n) 位置的 D 维向量）为最小任务单元**，rows=B×S×N；**persistent-core**：grid=min(cdiv(rows, BLOCK_ROWS), vector_core_num)，核内连续段循环（2026-09-07 P4 定型的 elementwise 默认范式） | ✅ | 三模板分核轴（BS 或 B×S）都是「行空间」的不同切法；行间完全独立（rope 只耦合行内 D 维配对），打平成行轴是公共子结构。persistent-core 与源码「每核 blockFactor 连续段 + 核内 ubFactor 循环」同构 |
| Q/K 两路融合在一次 launch（apt.cpp 单 kernel 双 tensor；核内先 Q 后 K 复用 cos/sin，aba_and_ba.h:214-234） | **默认两次 launch**（q 一次、k 一次，共享同一 kernel）；cos/sin 复用由 L2 覆盖 | ✅ | cos/sin 体量 = S×RD（KB 级），第二次 launch 全命中 L2；两次 launch 省掉 Q/K 边界拼接逻辑（ab.h:142-161 的跨界块处理在 Triton 无须复刻）。如需极致融合可单 launch 双 masked load，标为可选变体 |
| 尾核少载（blockTailBS / blockFactor 取余） | `mask = row_offs < rows` 主尾统一 | ✅ | 与 fused_bias_leaky_relu 同范式 |
| legacy 小 shape key=1（每核 1 batch，tiling.cpp:326-332） | 不需要特判——persistent-core 的 cdiv 自然覆盖小 shape | ✅ | grid 公式在 rows<核数时自动退化为 rows 个 program |

## 2. tilingdata → BLOCK 超参

| tilingdata 字段 | 语义 | Triton 对应 | 候选值 / 推导式 |
|---|---|---|---|
| realDim (RD) / dSplitCoef | 旋转维长度 / 配对系数（half=2, interleave=1*, quarter=4） | `RD` 形参 + `ROTARY_MODE: tl.constexpr` | RD≤1024（D_LIMIT，tiling_base L33） |
| dAlign = CeilAlign(RD/coef, 32B/typeSize)×coef | UB 对齐行宽 | `BLOCK_DH: tl.constexpr` | `next_pow2(RD//pair)`，half/quarter 时 pair=RD/2、RD/4；interleave 时 pair=RD/2（对数） |
| ubFactorS/ubFactorB/ubFactorQN/KN | 核内 UB 切分 | `BLOCK_ROWS: tl.constexpr`（每 program 每次迭代行数） | {1, 2, 4, 8, 16}，与 BLOCK_DH 联合约束 UB（§4） |
| isPartialRope | realDim<D | wrapper：`out = x.clone()` 兜底直通段 + kernel 只写 [0,RD) 区域 | RD==D 时走 empty_like 快路径（constexpr 分支也可，但 wrapper 侧更省） |
| blockNumB/S、blockFactor 系列 | 分核 | 不需要——persistent-core 隐式覆盖 | — |
| 4095 blockCount 上限（tiling_ab L81 等） | DataCopy 硬件限制 | 丢弃 | Triton 无 DMA blockCount 概念 |

*interleave 的 dSplitCoef=1 是因搬运不切半（奇偶对在 VL 内解决）；Triton 侧三种模式都按「配对数 = RD/2」取向量宽，quarter 例外（RD/4 宽、4 段）。

## 3. tilingkey / 模板分支 → constexpr

| tilingkey / 分支 | 分支语义 | Triton 处理 |
|---|---|---|
| 20010(ABA) vs 20011(BA) vs 20020(BAB) | cos/sin B 维广播与否（IsBBoardcast 控制 cos/sin 是否每 S 段载一次） | `stride_cb` 参数：CosB==1 时传 0，否则传 cos 的 B 步长——广播退化为寻址，无分支 |
| 20030(AB) 的 SBND cosb==1 时 b 折进 n（tiling_ab L56-60） | 维度折叠优化 | 不需要——strides 参数化天然覆盖（b  stride 任意） |
| rotaryMode∈{1,2,3} → Half/Quarter/InterleaveVF | 三种配对数学 | `ROTARY_MODE: tl.constexpr` 单 kernel 三分支（配对结构差异大，合并为一个 kernel 三个 constexpr 段） |
| BUFFER_NUM=2 双缓冲（4 队列） | 流水掩盖 | ⚠️ 编译器注流，autotune num_stages {1,2,3} |
| dtype 模板（DTYPE_QUERY） | fp16/bf16/fp32 实例化 | 单 kernel：指针 dtype 自动分派 + `.to(tl.float32)` 统一计算域 |
| legacy key 1/3/4（小 shape/聚集搬出/cast） | 老平台路径 | 全部丢弃——Triton 目标平台为 arch35；BF16 的 fp32 计算域由统一 `.to(tl.float32)` 覆盖 |

## 4. 片上内存 → 隐式化 + 预算校核

| Ascend C buffer（以 ABA 为例） | Triton 等效 tile | 字节数校核 |
|---|---|---|
| qIn/qOut：ubFactorB×S×N×dAlign×T×2(dbuf) | 每次迭代的 [BLOCK_ROWS, BLOCK_DH] 输入/输出 tile ×2（两半） | BLOCK_ROWS=8, RD=128(half→BLOCK_DH=64), fp16：8×64×2B×2 半×2(in/out) ≈ 4KB |
| cos/sin：ubFactorS×dAlign×T×2 | [BLOCK_DH] ×4（c1/c2/s1/s2） | 64×2B×4 = 512B，全行共享 |
| fp32 Reg 计算域 | 计算寄存器 tile（编译器分配） | BLOCK_ROWS×BLOCK_DH×4B×6 寄存器组量级 ≈ 12KB |
| **合计** | 每迭代活跃数据 < 32KB | vs UB 192KB/核：**富余两个数量级**，BLOCK_ROWS 可放心开大；瓶颈在行内 BLOCK_DH×BLOCK_ROWS 的指令并行度而非容量 |

- D≤1024 → 单行 rope 区域最大 1024 元素（fp32 4KB），单行即可放得下；BLOCK_ROWS 取大行收益更大（多行摊薄 cos/sin 载入）。

## 5. 流水编排 → Triton 可达手段

| Ascend C 手段 | 锚点 | 可达性 | Triton 写法 / 降级 | 预期影响 |
|---|---|---|---|---|
| CopyIn→Compute→CopyOut 三段式（核内 ubFactor 循环） | aba_and_ba.h:236-244 等 | ✅ | 核内连续段 for 循环内 load→compute→store | 无 |
| LoopModeParams 多维跨步 DMA（BNSD 的 B→N→S 三维） | aba_and_ba.h:284-291 | ✅ | strides 参数化寻址（`b*sxb + s*sxs + n*sxn`），Triton 编译器生成对应 DMA | 无（连续 D 维仍是主搬运方向） |
| DataCopyPad 越界保护 | 各处 | ✅ | `tl.load(ptr+offs, mask=..., other=0)` | 无 |
| TQue 双缓冲 ×4 队列 | aba_and_ba.h:139-148 等 | ⚠️ 编译器自动 | num_stages autotune；约束「行循环的 GM 往返需掩盖」 | 编译器注流质量决定 |
| MicroAPI DeInterleave/Interleave（奇偶解/重交织） | common.h:272-274, 681-683 | ⚠️ 部分可达 | **默认**：stride=2 的两次 tl.load/tl.store（de=2i, do=2i+1）；**变体**：若 Triton-Ascend 支持 `tl.reshape(BLOCK,2)+tl.split`/`tl.interleave` 可用连续 load+寄存器重排——两态都给出，实测择优 | strided-2 访存可能触发非对齐 DMA，interleave 模式预期带宽弱于 half（预警） |
| MicroAPI UpdateMask 逐 VL 尾处理 | common.h:101 等 | ✅ | `mask = d_offs < RD_half` 统一 | 无 |
| RegTensor<float> 全程 fp32 计算 | common.h 全部 VF | ✅ | `.to(tl.float32)` 计算 + `.to(out_dtype)` 末段回写（RTNE） | 必须保留（精度） |
| partial rope 双段搬运（CopyIn/OutHalfMode 等） | common.h:781-877 | ✅ | wrapper `out=x.clone()` + kernel 只算/只写 [0,RD) | clone 多一次全量拷贝；可选优化：kernel 内补写直通段 |
| BA 广播 cos/sin 只载一次 | aba_and_ba.h:185-196 | ✅ | stride_cb=0；行循环内 cos/sin 小 tile 自然驻留 L1/L2 | 无 |
| Q/K 跨界块拼接（ab.h:142-161） | tiling_ab 的 QN+KN 合并轴 | 丢弃 | 两次 launch 不需要跨界处理 | 无 |
| workspace 16MB 保留 | tiling 各 GetWorkspaceSize | 丢弃 | Triton 不需要 | 无 |

**❌ 不可达项：0 个**。最险的是 DeInterleave/Interleave 的 Reg 级重排，标 ⚠️：Triton 用 strided-2 load/store 等价实现，语义完全可达，仅访存模式可能次优。

## 6. Compute → Triton 计算骨架

统一式 `out = x·cos + rotate(x)·sin`，fp32 计算域；pair 索引 i∈[0, RD/2)（quarter 为段内 i∈[0, RD/4)）：

| 模式（VF 锚点） | Triton 逐指令翻译 | 可达性 |
|---|---|---|
| half（common.h:111-119）：out₁=x₁c₁−x₂s₁，out₂=x₂c₂+x₁s₂ | `x1=load(d+i); x2=load(d+i+h)`；`o1=x1*c1-x2*s1; o2=x2*c2+x1*s2` | ✅ |
| quarter（common.h:193-209）：段内同 half，4 段 | 同上结构 ×2（段 1↔2、3↔4，偏移 q=RD/4） | ✅ |
| interleave（common.h:270-280）：out[2i]=xe·ce−xo·se，out[2i+1]=xo·co+xe·so | `xe=load(d+2i); xo=load(d+2i+1)`；`oe=xe*ce-xo*se; oo=xo*co+xe*so` | ✅（访存 ⚠️ 见 §5） |

骨架全文见 `triton_skeleton.py`——单 kernel + ROTARY_MODE constexpr + strides 布局归一化 + persistent-core 行循环。

## 7. 精度策略 → 标杆与验收

- fp32 计算域全程保留（对齐 common.h 全部 RegTensor<float>）；唯一舍入点在末段 `.to(out_dtype)`（RTNE）。
- torch golden：`out = x.float()*cos.float() + rotate(x.float())*sin.float()`，三模式参数化 rotate；末段 `.to(dtype)`。
- 算子级验收：fp32 输入 rtol=1e-5/atol=1e-5（一乘一加，无累加）；fp16/bf16 对 fp32 标杆 ≤1 ulp（单次 RTNE）；**模式专项**：half 验证前后半场交叉、interleave 验证奇偶位、quarter 验证四段；**partial rope 专项**：[RD, D) 必须 bitwise 等于输入。

## 8. 显式丢弃 / 不涉及清单

| 项 | 处理 | 理由 |
|---|---|---|
| 4 个 tiling 模板 / 4 个 kernel 类 | 合并为单 kernel | 差异仅为循环结构与广播寻址，§0 已论证 |
| blockNum/Factor/Tail 系列、usedCoreNum 比选 | 丢弃 | persistent-core 范式隐式覆盖 |
| ubFactor 反解公式（UB 预算） | 丢弃（语义保留为 §4 校核） | Triton 无显式 UB 管理 |
| 4095 blockCount 上限 | 丢弃 | DMA 硬件概念 |
| Q/K 跨界块、N 合并轴 | 丢弃 | 两次 launch |
| workspace | 丢弃 | 无跨核中间量 |
| legacy 3 key + BF16 cast 模板参 | 丢弃（进卡片架构差异节） | 目标平台 arch35 |
| AutoContiguous | wrapper `.contiguous()` | 框架职责 |
| DeepSeekInterleave VF | 不涉及 | 在仓未用资产（extraction §1.1） |

## P2 质量门自查

extraction §4.1.1~§6 每条事实有去向；❌=0，⚠️=3（双缓冲注流、interleave strided 访存、Reg 重排变体）均有两态表述无绝对禁令；骨架见 triton_skeleton.py。
