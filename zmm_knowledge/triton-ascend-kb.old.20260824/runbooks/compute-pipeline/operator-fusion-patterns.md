---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.4"
title: 算子融合模式与约束决策
description: 将数据依赖的连续算子融合为单个 kernel，消除中间 HBM 读写；必须满足 UB 容量、grid 结构一致、无多消费者等约束。
tags: [fusion, gemm, activation, rmsnorm, rope, quantization, async_task, hivm]
confidence: verified
status: verified
created_at: "2026-08-10T00:00:00Z"
updated_at: "2026-08-10T00:00:00Z"
---

## DG-11 算子融合模式与约束决策

【优化维度】：计算｜融合
【泛化层级】：条件性（迁移条件：数据依赖的连续算子；失效边界：独立并行、多消费者、UB 放不下或对齐冲突）

<原则：融合把多个算子的 HBM 往返消除，将中间结果保留在 UB/寄存器内。对内存受限算子（elementwise、norm、activation）通常带来 25–50% 提升。但融合必须满足 UB 容量、相同 grid 结构、无多消费者、对齐兼容四个约束，否则反而导致 UB overflow 或编译失败。>

**为什么融合能提速**：

```
未融合：GEMM → [HBM 写 C] → RMSNorm → [HBM 读 C] → [HBM 写 Y]
已融合：GEMM → [UB 内] → RMSNorm → [HBM 写 Y]
```

每消除一对 HBM 读/写，约省 15–20% 延迟。

**常见融合模式与收益**：

| 模式 | 收益 | 关键约束 |
|---|---|---|
| GEMM + Activation | 15–20% | activation 在 fp32 accumulator 上计算 |
| GEMM + Bias | 10–15% | bias 向量寄存器驻留 |
| GEMM + RMSNorm | ~35% | 输出 tile UB 内完成归一化 |
| GEMM + RoPE | ~25% | Q/K head 布局兼容 |
| Skip + LayerNorm | ~30% | residual 与输入同 grid |
| RMSNorm + Quantization | ~50% | scale 在 UB 内计算并写出 |
| Conv1D + State + SiLU | ~35% | 状态更新用双缓冲避免竞争 |

**Cube/Vector 异构融合（dl.async_task）**：

```python
import triton_ascend.language as dl

@triton.jit
def fused_gemm_norm(...):
    with dl.async_task(scope="cube"):
        acc = tl.dot(a_tile, b_tile)

    with dl.async_task(scope="vector"):
        rms = tl.sqrt(tl.sum(acc * acc, axis=1) / N + eps)
        out = (acc / rms[:, None]) * weight
        tl.store(Y_ptr + ..., out)
```

Cube 计算 tile `i+1` 的同时，Vector 归一化 tile `i`。

**融合约束清单**：

```
□ 操作 A 的输出是操作 B 的输入（数据依赖）
□ 融合后所有中间 tensor 能放入 UB（≤ 192KB，matmul 保守 ≤ 64KB）
□ 两个操作使用相同 grid / block 结构
□ 中间结果不被多个下游消费者使用
□ 对齐要求兼容（如 GEMM BLOCK_N=256 vs GroupNorm cpg=64 可能冲突）
□ 激活函数在 fp32 累加器上计算，避免 fp16 溢出
```

**坏实践（反例）**：
```python
# 强行融合独立操作，grid 不一致
op1_grid = (batch, M)
op2_grid = (batch, N)
```
后果：编译失败或性能倒退。✅ 正解即只融合数据依赖且 grid 一致的操作。

**双缓冲状态融合**：当多个 program 并发写入共享状态（conv_state、KV cache）时，写独立 buffer + host 侧 `copy_` 同步，避免 WAR 依赖和伪共享。

**决策流程**：

```
1. 是否存在数据依赖链 A→B？
   NO → 不融合
2. 中间 tensor 是否多消费者？
   YES → 不融合（保留中间 HBM）
3. 融合后 tile 是否超 UB？
   YES → 增大 BLOCK 或拆分为两阶段
4. grid/对齐是否兼容？
   NO → 不融合或调整 BLOCK
5. 使用 dl.async_task 区分 Cube/Vector？
   YES → 异构并行
```

<!-- okf:related:start -->

# 相关

- 实践案例: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../../ops/mla-preprocess.md) — 主题相关：共享 fusion, rmsnorm, rope
- 实践案例: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../../ops/mla-prolog-v3.md) — 主题相关：共享 rmsnorm, rope
- 实践案例: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](../../ops/mla-prolog.md) — 主题相关：共享 fusion, rmsnorm, rope
- 实践案例: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 同属优化知识，主题相关
- 实践案例: [RMSNorm-RoPE fused kernel on Ascend 910B1](../../ops/rmsnorm-rope.md) — 主题相关：共享 rmsnorm, rope
- 实践案例: [RoPE paired-channel pointer addressing](../../ops/rope-pointer-addressing.md) — 主题相关：共享 rope
- 开发指南: [Ascend NPU 特有语法与模式](../../reference/ascend-patterns.md) — 同属优化知识，主题相关
- 开发指南: [Elementwise 算子优化](../../reference/op-class-elementwise.md) — 主题相关：共享 fusion
- 开发指南: [融合算子综合页](../../reference/op-class-fused_ops.md) — 主题相关：共享 fusion, gemm, rmsnorm, rope
- 开发指南: [MatMul / GEMM 高性能算子](../../reference/op-class-matmul.md) — 主题相关：共享 gemm
- 开发指南: [归一化算子优化](../../reference/op-class-normalization.md) — 主题相关：共享 rmsnorm
- 开发指南: [RoPE（旋转位置编码）算子优化](../../reference/op-class-rope.md) — 主题相关：共享 rope
- 相关主题: [Gemm Basic Skeleton](../../reference/skeleton-gemm_basic.md) — 同属优化知识，主题相关
- 相关主题: [Rmsnorm Fused Skeleton](../../reference/skeleton-rmsnorm_fused.md) — 同属优化知识，主题相关
- 相关主题: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 主题相关：共享 rmsnorm
- 相关主题: [GEMM 操作数保持 low_dtype，fp32 累加](../algorithm-numerics/gemm-low-dtype-fp32-accum.md) — 主题相关：共享 gemm
- 相关主题: [Normalization Compute 优化聚合](../algorithm-numerics/traj-normalization-compute.md) — 主题相关：共享 rmsnorm
- 相关主题: [Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion](../algorithm-numerics/traj-normalization-memory_access.md) — 主题相关：共享 rmsnorm
- 相关主题: [Activation Backward 内存访问优化聚合](../memory-sram/traj-activation-memory_access.md) — 主题相关：共享 activation
- 相关主题: [Matmul 内存访问优化：分块策略、L0 复用与算子融合](../memory-sram/traj-matmul-memory_access.md) — 主题相关：共享 rmsnorm
- 相关主题: [RoPE 内存访问优化：cos/sin 复用、ALIGNED 向量化加载与动态特化](../memory-sram/traj-rope-memory_access.md) — 主题相关：共享 rope
- 相关主题: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../meta-skills/attention-fusion-failure-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [量化融合算子 int8 输出精度不一致的逐层下钻诊断链](../meta-skills/quant-fusion-int8-precision-diagnosis.md) — 主题相关：共享 quantization
- 相关主题: [Attention 硬件适配优化聚合](../portability/traj-attention-hardware_adaptation.md) — 主题相关：共享 hivm
- 相关主题: [SDA（Scaled Dot Attention）Ascend 硬件适配：HiVM 分块、多缓冲预取与 Cube/Vector 均衡](../portability/traj-misc-hardware_adaptation.md) — 主题相关：共享 hivm
- 相关主题: [Llama4 RoPE 在 Ascend NPU 上的硬件适配：UB Tiling、FMA 与交错复数布局](../portability/traj-rope-hardware_adaptation.md) — 主题相关：共享 rope
- 相关主题: [dl.async_task 实现 Cube/Vector 异构并行](async-task-cube-vector.md) — 主题相关：共享 async_task, fusion
- 相关主题: [exp_matmul_vec_parallel_001 — matmul 向量并行 + activation fusion](instances/traj-exp-matmul-vec-parallel-001.md) — 主题相关：共享 gemm
- 相关主题: [SwiGLU 计算优化：门控融合、展平布局与激活量化](traj-activation-compute.md) — 主题相关：共享 activation, quantization
- 相关主题: [Activation Scheduling 优化聚合](traj-activation-scheduling.md) — 主题相关：共享 activation
- 相关主题: [Matmul Compute 优化聚合](traj-matmul-compute.md) — 主题相关：共享 gemm, quantization
- 相关主题: [Matmul Scheduling 优化聚合](traj-matmul-scheduling.md) — 主题相关：共享 gemm
- 相关主题: [Normalization Scheduling 优化聚合](traj-normalization-scheduling.md) — 主题相关：共享 rmsnorm
- 相关主题: [Reduction Compute 优化聚合](traj-reduction-compute.md) — 主题相关：共享 rmsnorm
- 相关主题: [RoPE 计算优化：cos/sin 复用、向量化旋转与多模态位置编码融合](traj-rope-compute.md) — 主题相关：共享 rope
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](traj-rope-scheduling.md) — 主题相关：共享 rope

<!-- okf:related:end -->
