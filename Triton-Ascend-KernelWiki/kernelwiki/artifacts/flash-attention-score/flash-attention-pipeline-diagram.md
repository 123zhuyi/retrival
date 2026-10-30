# FlashAttention Pipeline Stage Diagram

Derived from static analysis of CANN ops-transformer FlashAttentionScore (commit d28d8f99).
`confidence: inferred`, `reproducibility: snippet`.

## 1. 四阶段 Pipeline 总览

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     FlashAttention Pipeline                             │
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐ │
│  │  Stage 1    │    │  Stage 2    │    │  Stage 3    │    │  Stage 4 │ │
│  │  Cube1      │───>│  Vector1   │───>│  Cube2      │───>│  Vector2 │ │
│  │             │    │             │    │             │    │          │ │
│  │ IterateBmm1 │    │ ProcessVec1 │    │ IterateBmm2 │    │ProcessV2 │ │
│  │             │    │             │    │             │    │          │ │
│  │ Q·Kᵀ       │    │ pse+scale   │    │ P·V         │    │ rescale  │ │
│  │ matmul      │    │ +mask       │    │ matmul      │    │ +dropout │ │
│  │             │    │ +FlashSoftmax│   │             │    │ +output  │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────────┘ │
│                                                                          │
│  依赖链: Bmm1 ──> Vec1 ──> Bmm2 ──> Vec2                              │
│  数据流: Q,K → Bmm1结果 → Vec1结果(P) → Bmm2结果 → 最终输出            │
└──────────────────────────────────────────────────────────────────────────┘
```

## 2. Atlas A2 训练系列 (arch32) CV 流水

```
时间轴 ──────────────────────────────────────────────────────────────────>

Cube:   │ Bmm1_blk0 │ Bmm1_blk1 │ Bmm1_blk2 │ Bmm2_blk0 │ Bmm2_blk1 │ ...
        │  (preload) │  (preload) │            │            │           │
        │            │            │            │            │           │
Vec0:   │            │ Vec1_blk0  │ Vec1_blk1  │ Vec2_blk0  │ ...      │
        │            │            │            │            │           │
Vec1:   │            │ Vec1_blk0  │ Vec1_blk1  │ Vec2_blk0  │ ...      │
        │            │ (V0/V1各半) │            │            │           │

        │◄── preload ──►│◄── CV 并行 ─────────────────────────────────►│

说明:
- Cube 双发(preload): 提前发射2块Bmm1, Cube利用率最高
- C:V = 1:2, 1个Cube核对应2个Vector核
- Vec计算快, Cube搬运时长覆盖Vec计算, 最终 MTE2 bound 或 Cube bound
- Bmm1结果通过 workspace (GM) 传递给 Vector
```

### A2 CV 数据通路

```
         L1            L0C           UB            GM
        ┌───┐        ┌─────┐      ┌──────┐      ┌───────┐
  Q,K ─>│   │─Bmm1──>│     │      │      │      │       │
        │   │        │结果  │─cpy─>│Vec1  │      │       │
        │   │<─V─────│     │      │处理   │      │       │
        └───┘        └─────┘      │      │      │       │
                                  │P矩阵 │─cpy─>│Bmm2   │
                                  │      │      │输入    │
                                  └──────┘      │       │
                                               │       │
                        ┌─────┐      ┌──────┐  │       │
  V ───────────────────>│     │─Bmm2─>│     │  │       │
                        │ L1  │      │L0C  │  │       │
                        └─────┘      │结果  │  │       │
                                     └─────┘  │       │
                                               │       │
                        ┌─────┐                │       │
  Attention Out <───────│ UB  │<───────────────│       │
                        └─────┘                └───────┘
```

## 3. Ascend 950PR/950DT (arch35) CV 流水

```
时间轴 ──────────────────────────────────────────────────────────────────>

Cube:   │ Bmm1_0 │ Bmm1_1 │ Bmm1_2 │ Bmm2_0 │ Bmm2_1 │ ...
        │(preload)│(preload)│(preload)│         │        │
        │         │         │         │         │        │
Vector: │         │ Vec1_0  │ Vec1_1  │ Vec2_0  │ ...   │
        │         │         │         │         │        │

        │◄──── preload 3次 ────►│◄── CV 并行 ──────────────►│

说明:
- Cube 三次 preload: 完成3次mm1后才开启mm2
- 优化启动阶段CV流水, 更紧密
- AIC可直接输出到UB (融合架构), mm1结果留在UB
- Dv>128时 mm2 输出需先保存到GM
```

### 950PR UB 直连通路

```
         L1            L0C           UB            GM
        ┌───┐        ┌─────┐      ┌──────┐      ┌───────┐
  Q,K ─>│   │─Bmm1──>│     │─────>│Vec1  │      │       │
        │   │        │结果  │ UB!  │处理   │      │       │
        │   │<─V─────│     │直连   │      │      │       │
        └───┘        └─────┘      │P矩阵 │      │       │
                                  │      │      │       │
                                  │Bmm2  │      │       │
                                  │左矩阵│      │       │
                                  │(L1复用│      │       │
                                  │ 可选) │      │       │
                                  └──────┘      │       │
                                               │       │
                        ┌─────┐      ┌──────┐  │       │
  V ───────────────────>│     │─Bmm2─>│L0C  │  │       │
                        │ L1  │      │结果  │─>│Dv>128 │
                        └─────┘      │      │  │时写GM │
                                     └─────┘  │       │
                                               │       │
                        ┌─────┐                │       │
  Attention Out <───────│ UB  │<───────────────│       │
                        └─────┘                └───────┘
```

## 4. V 侧 Double Buffer

```
时间轴 ──────────────────────────────────────────────>

ping:   │DataCopy_0│  Clc_0  │         │DataCopy_2│  Clc_2  │
        │          │         │         │          │         │
pong:   │          │DataCopy_1│ Clc_1  │          │DataCopy_3│ ...
        │          │         │         │          │         │

说明:
- ping/pong 两块数据使用独立 UB 空间
- 同一数据: DataCopy → Clc 串行
- 不同数据切片: 并行处理 (搬运与计算重叠)
- FA 类算子 V 侧计算过程复杂, 简单 double buffer 无法覆盖所有情况
  → 不同 shape 使用不同的计算流水排布
```

## 5. Pipeline 优化总结

| 优化技术 | 适用架构 | 说明 |
|----------|---------|------|
| V 侧 double buffer | A2, 950PR | ping-pong 数据搬运/计算重叠 |
| Cube 双发 preload | A2 | 提前2块 Cube 计算, Cube bound |
| Cube 三次 preload | 950PR | 提前3次 mm1, 启动阶段更紧密 |
| UB 直连 (AIC→AIV) | 950PR | AIC 直接输出到 UB, 免 GM 搬运 |
| L1 复用 (EnableL1Reuse) | A2 | Bmm2 左矩阵 P 留在 L1 复用 |
| TSCM 复用 (MatmulPolicy) | 950PR | Bmm2 左矩阵留在 TSCM/L1 复用 |
| NZ 格式 + SameAB | A2, 950PR | Cube 为主核, L0C 累加, 减少带宽依赖 |
| BigDoubleBuffer | A2 | UB 空间换时间, 更大 double buffer |
