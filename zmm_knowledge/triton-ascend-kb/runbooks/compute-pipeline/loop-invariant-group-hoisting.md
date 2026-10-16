---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.1"
title: 循环不变量按复用组外提（cos/sin 类小表每组一次，实测单点 +100%）
description: 广播小表（cos/sin、scale、bias 等）在复用组（如 (b,s)）内恒等时，必须把循环结构改成「组为外循环、组内行展开为 tile」，让小表每组只 load 一次；逐行重载实测慢一倍（2.0956→4.2060）。L1/L2 驻留只省带宽、不省重复 load 的指令与地址计算，不能替代显式外提。
tags: [loop_invariant_hoisting, broadcast_table, cos_sin, group_outer_loop, tile_reuse, persistent_core]
confidence: verified
status: verified
created_at: "2026-09-09T00:00:00Z"
updated_at: "2026-09-09T00:00:00Z"
---

## 循环不变量按复用组外提（来源：ApplyRotaryPosEmb P4 直给卡组决定性优化，2026-09-08）

【优化维度】：流水｜循环结构
【泛化层级】：普遍性（迁移条件：kernel 存在「组内恒等的小表」——RoPE cos/sin、量化 scale、norm weight 等广播输入；无失效边界）

<原则：当输入小表在某个分组维度内恒等（cos/sin 在 (b,s) 组内对全部 head 行恒等），循环结构必须以**组**为外循环、组内行展开为 tile，小表在组头 load 一次广播到 tile。逐行（或逐块跨组）循环会让小表重复 load N 次——即便全部命中 L1/L2，重复 load 的指令发射与地址计算本身即翻倍开销。**cache 驻留 ≠ 复用**：驻留省的是带宽，外提省的是指令流。>

### 实测（geomean，45-case，同代码仅改循环结构）

| 结构 | geomean | 说明 |
|---|---|---|
| 逐行打平循环，cos/sin 每行重载 | 2.0956 | persistent-core 行循环 |
| (b,s) 组外循环 + 组级外提 | **4.2060** | +100%，单点决定性杠杆 |

### 复刻模式

```python
for bs in range(bs_start, bs_end):          # 组 = 复用不变量的最大作用域
    c_base = b * scb + s * scs              # 标量
    c1 = tl.load(cos_ptr + c_base + i, ...) # 组头载一次
    ...                                      # 组内 [BLOCK_N, ...] tile 全部复用 c1/s1
```

附带收益：bs 为标量循环使行解码退化为标量 div/mod，同时规避「向量 div/mod × 特定访存」的后端缺陷面（见 [strided-2/interleave 访存契约](../memory-sram/triton-ascend-strided2-interleave-contract.md)）。grid = min(组数, 物理核数)，连续块划分（相邻组共享小表行，L2 友好）。

### 迁移自查

- 列出 kernel 全部输入，标注每个在哪些循环维度上恒等；
- 循环嵌套顺序 = 恒等范围从大到小；最内层必须是没有不变量的纯数据维；
- 若 AscendC 源码有「载一次复用」注释/结构（如队列外提），Triton 侧**必须复刻其语义**（形式不必复刻）。

<!-- okf:related:start -->

# 相关

- 相关主题: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](../memory-sram/exp_gemma_rmsnorm_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: loop invariant code motion](../memory-sram/exp_lightning_attn_decode_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](../memory-sram/exp_outer_product_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: cos sin reuse](../memory-sram/exp_rope_npu_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: cos sin reuse](exp_rope_001.md) — 同属优化知识，主题相关
- 相关主题: [循环不变量外提（LICM）减少重复地址计算](loop-invariant-code-motion.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
