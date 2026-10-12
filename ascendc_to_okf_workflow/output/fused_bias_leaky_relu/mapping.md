# P2 语义转化 — fused_bias_leaky_relu（Ascend C → Triton）

> 输入：extraction.md。每条 P1 事实在此有去向（转化/合并/显式丢弃+理由）。
> 可达性三态：✅ 直接可达 / ⚠️ 编译器自动 / ❌ 不可达（含降级方案）。

## 1. 分核策略 → grid 设计

| Ascend C 事实 | Triton 决策 | 可达性 | 等价论证 / 依据 |
|---|---|---|---|
| 打平一维 + `blockFactor=CeilAlign(CeilDiv(totalNum,coreNum),32B/typeSize)` 静态均分 + 核内 ubFactor 循环（tiling L169-170；kernel h:79-81, 117-126） | **persistent-core**：grid = min(CeilDiv(numel, BLOCK), vector_core_num)，kernel 内每 program 处理一段连续 block 区间（start_block=pid*per_prog … 尾块 mask） | ✅ | **2026-09-07 P4 修订**：初版取「一块一 program、grid=NUM_BLOCKS」，实测为次优起点（grid 超核数串行排队，对照实验 case6 仅 0.69x），三组最终版全部收敛到 persistent-core——而它正是源码「每核 blockFactor 连续段 + 核内 ubFactor 循环」的忠实映射，等价论证直接成立 |
| host 侧 tiling 标量（blockFactor/ubFactor/属性） | wrapper 预计算；`negative_slope`/`scale` 作 kernel runtime 标量形参；BLOCK 作 `tl.constexpr` | ✅ | mapping 速查表 §1「tilingdata 标量一律上移 wrapper」 |
| 空 tensor 短路（tiling L155-164） | wrapper `if numel == 0: return out` | ✅ | 不进 kernel |

## 2. tilingdata → BLOCK 超参

| tilingdata 字段 | Triton 对应 | 候选值 / 推导式 |
|---|---|---|
| ubFactor（单循环 UB 元素数，fp16/fp32 双缓冲均按 24B/elem 预算） | `BLOCK: tl.constexpr` | {2048, 4096, 8192, 16384}，对齐 128；ubFactor 参考值 = ubSize/24 ≈ 8K（192KB UB）量级 |
| blockFactor（每核均分量） | 不需要——grid-stride 范式下由 `num_programs` 与 BLOCK 隐式覆盖 | — |
| negativeSlope / scale | kernel 形参（runtime float 标量） | 默认值 0.2 / 1.414213562373 由 wrapper 注入 |
| 尾块 tailNum_ | `mask = offs < numel` | 主尾统一写法，无单独分支 |

## 3. tilingkey 模板分支 → constexpr

| tilingkey / 模板分支 | 分支语义 | Triton 处理 |
|---|---|---|
| BUFFER_MODE ∈ {0,1}（tiling_key.h:26；host 判定 loopCountIfDouble≥2，tiling L177-179） | 队列双缓冲开关 | **不需要 constexpr**——双缓冲在 Triton 属编译器注流（multibuffer/num_stages），标 ⚠️ 交给编译器；host 的「能循环≥2 次才开双缓冲」策略语义由 autotune 的 num_stages 候选空间覆盖 |
| DTYPE_X 宏分派（fp16/fp32） | dtype 模板 | **单 kernel**：指针 dtype 自动分派，`NEED_CAST` 等价于 kernel 内 `.to(tl.float32)` 统一计算路径（fp32 输入时 cast 是恒等，编译器消除）——mapping 速查表 §1「N 路分派壳→单 kernel」范式 |

## 4. 片上内存 → 隐式化 + 预算校核

| Ascend C buffer | Triton 等效 tile | 字节数校核（BLOCK=8192 时） |
|---|---|---|
| inputQueueX + inputQueueBias + outputQueueY（VECIN/VECOUT ×2 双缓冲） | 两次 `tl.load` 的 [BLOCK] tile + 一次 `tl.store` 寄存器 tile | fp16: 8192×2B×3 = 48KB（单缓冲视角） |
| xFp32/biasFp32/yFp32（VECCALC，fp16 路径） | 计算寄存器（fp32 tile），编译器自动分配 | 8192×4B×3 = 96KB（含于计算路径，非驻留） |
| **合计校核** | fp16 峰值 ≈ 双缓冲 6×16KB + 计算 96KB 量级 vs UB 192KB/核 | **富余**——elementwise 双输入单输出是 UB 最省的形态之一，BLOCK=8192 无溢出风险；BLOCK=16384 fp16 时（队列 192KB 单缓冲视角 96KB + fp32 计算 192KB）需谨慎，autotune 上限建议 8192 |

## 5. 流水编排 → Triton 可达手段

| Ascend C 手段 | 可达性 | Triton 写法 / 降级 |
|---|---|---|
| CopyIn→Compute→CopyOut 三段式循环 | ✅ | load → compute → store（单 BLOCK 内天然三段；跨块流水由编译器） |
| TQue 双缓冲（buffer_num=2） | ⚠️ 编译器自动 | 约束写入卡片：「需要双缓冲掩盖 GM 往返」；autotune num_stages 候选 {2,3,4} |
| DataCopyPad 尾块补零（padParams 实际不补零，仅越界保护） | ✅ | `tl.load(ptr+offs, mask=offs<numel)`；本算子 other 值无关（越界元素不参与 store） |
| TPipe EnQue/DeQue 事件同步 | ⚠️ | 丢弃，编译器自动插同步（速查表 §4） |

## 6. Compute → Triton 计算骨架

| Vector 指令（h:161-169） | Triton 对应 | 可达性 |
|---|---|---|
| Cast(fp16→fp32, CAST_NONE) | `x = x.to(tl.float32)` | ✅ 精确提升，必须保留 |
| Add(x, bias) | `t = x + b` | ✅ |
| LeakyRelu(t, ns) | `u = tl.where(t >= 0, t, t * negative_slope)` | ✅（速查表 §3 Compare+Select） |
| Muls(u, scale) | `y = u * scale` | ✅ 标量广播 |
| Cast(fp32→fp16, CAST_ROUND=RTNE) | `y.to(out_dtype)` | ✅ RTNE 即 Triton cast 默认舍入 |
| **可选强度折减** | 预折 `scale_pos=scale`、`scale_neg=scale*negative_slope`，`y = t * tl.where(t>=0, scale_pos, scale_neg)`——3 步变 2 步（一次 where 一次 mul） | ✅ 数学等价（实数域）；浮点域 `(t*ns)*scale` vs `t*(ns*scale)` 有一次舍入差，验收阈值须容忍（见 triton_skeleton.py 注释） |

## 7. 精度策略 → 标杆与验收

- fp32 计算域全程保留；fp16 唯一舍入点在末段 RTNE 回写。
- torch 标杆：`t = (x.float() + bias.float()); ref = torch.where(t >= 0, t, t * ns) * scale; ref = ref.to(out_dtype)`——与 kernel 舍入序列逐一对齐。
- 算子级验收：fp32 输入 rtol=1e-6/atol=1e-6；fp16 输入对 fp32 计算标杆允许 ≤1 ulp（RTNE 单次舍入）。

## 8. 显式丢弃 / 不涉及清单

| 项 | 处理 | 理由 |
|---|---|---|
| blockFactor 32B 对齐（ubBlockSize） | 丢弃 | Triton BLOCK 由 autotune 对齐 128 覆盖；对齐语义属 DMA 粒度优化 |
| useDoubleBuffer host 判定公式 | 丢弃（语义保留为约束） | 编译器注流，见 §5 |
| workspace=0 | 不涉及 | 无跨核中间量 |
| AutoContiguous | wrapper `x.contiguous()` 断言/调用 | 框架职责，kernel 只认连续 ND |
| aclnn 分解 / 多 kernel | 不涉及 | 本算子无 aclnn 层、无分解，「勿复刻分解」清单为空 |

## P2 质量门自查

extraction.md §4.1.1~§6 每条事实在上表有去向；❌ 项 0 个（纯向量直通算子，无 Reg/NZ/手工同步）；⚠️ 项（双缓冲、事件同步）均有约束化表述。骨架见 triton_skeleton.py。
