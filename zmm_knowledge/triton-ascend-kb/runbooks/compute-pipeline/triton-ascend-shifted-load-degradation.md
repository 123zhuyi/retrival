---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.3"
title: Triton-Ascend 移位 load 降级契约（大位移 ~25x 标量化，2D tile ±1 不合并）
description: 通过对已加载 tile 做地址偏移（e±K）实现配对/滑窗的「移位 load」写法，在 Triton-Ascend 上大范围降级：大位移（K=PAIR 级）整体 ~25x 劣化；±1 移位仅 1D 平坦流可被后端合并，2D tile 形式同样 2.5~3.5x 劣化。配对/重排一律走寄存器路径（tl.split/tl.gather/tl.interleave）。
tags: [shifted_load, sliding_window, pairing, tl_gather, backend_limitation, scalarization, register_rearrange]
confidence: verified
status: verified
created_at: "2026-09-09T00:00:00Z"
updated_at: "2026-09-09T00:00:00Z"
---

## Triton-Ascend 移位 load 降级契约（来源：ApplyRotaryPosEmb P4 优化轮回退记录，2026-09-08）

【优化维度】：流水｜访存结构
【泛化层级】：条件性（迁移条件：想用「二次偏移 load」实现配对、滑窗、旋转等数据重叠复用；失效边界：后端合并能力随版本演进，复测为准）

<原则：不要用 `tl.load(ptr + off)` 后 `tl.load(ptr + off + K)` 的移位 load 对来表达配对/滑窗——Triton-Ascend 后端对移位流的合并窗口极窄。数据已在 UB/寄存器时，用 `tl.gather`（行内置换）或 `tl.split/tl.reshape`（规则配对）做寄存器级重排，避免第二次全局访存。>

### 实测数据（geomean，45-case）

| 写法 | 结果 | 判定 |
|---|---|---|
| flat 1D 流 e±PAIR 大位移移位 load + tl.where | 0.2155x（~25x 降级） | 回退 |
| 2D tile ±1 移位 load 配对 | 2.1170x（vs 同结构非移位 2.4945x，且 7 case 一致性回退） | 回退 |
| ±1 移位 1D 平坦流 | 可被后端合并（仅此种形态有效） | 限定可用 |
| `tl.gather(x_ub, swap_idx, 1)` 行内 UB 置换 | 消除第二次全局读，单点 +26.6x（相对其组内基线） | ✅ |
| `tl.split(tl.reshape(x,(…,2)))` 寄存器配对 | 连续 load 前提下的安全路径 | ✅ |

### 判别与复刻

- 信号：kernel 出现「同一份数据两次 load、偏移相差常数 K」；性能随 K 增大断崖式劣化。
- 规则配对（奇偶/半宽）→ tl.split/tl.interleave（见 [strided-2/interleave 访存契约](../memory-sram/triton-ascend-strided2-interleave-contract.md)）；任意置换 → host 预建 swap LUT + `tl.gather`；两者都避免第二次 GM 往返。
- 注意 `tl.gather` 本身有残余开销（大 shape 下 ~2x vs 纯拷贝上限），超大 case 需权衡；但它**正确**，移位 load 在该后端既慢又可能踩 mask 缺陷。

<!-- okf:related:start -->

# 相关

- 实践案例: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](../../ops/nsa-compress-attention-infer.md) — 主题相关：共享 sliding_window
- 实践案例: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](../../ops/nsa-compress-attention.md) — 主题相关：共享 sliding_window
- 相关主题: [Memory Access Optimization: indirect load scalarization](../memory-sram/exp_grouped_attn_stage1_001.md) — 同属优化知识，主题相关
- 相关主题: [重叠 T-tiling 保证跨 tile 滑动窗口连续性](../memory-sram/overlapping-t-tiling.md) — 主题相关：共享 sliding_window
- 相关主题: [寄存器移位复用滑动窗口，消除重复 HBM 读](../memory-sram/register-shift-sliding-window.md) — 主题相关：共享 sliding_window
- 相关主题: [tl.trans 标量化灾难与寄存器重排替代（split/interleave/reshape 安全集）](triton-ascend-trans-scalarization.md) — 主题相关：共享 backend_limitation, register_rearrange, scalarization

<!-- okf:related:end -->
