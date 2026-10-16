---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: "1.2"
title: Tiling 预算与布局驱动的块大小选择
description: matmul/elementwise 的 BLOCK 选择必须同时满足 UB 192KB 预算、512B 行宽对齐和输入布局（BTD/BDT）特征，否则 UB overflow 或 cache-line 不对齐会同时触发。
tags: [tiling, ub_budget, block_size, layout_btd, layout_bdt, autotune]
confidence: verified
status: verified
created_at: "2026-08-10T00:00:00Z"
updated_at: "2026-09-30"
---

## OPT-12 Tiling 预算与布局驱动的块大小选择

【优化维度】：内存｜Tiling·SRAM 预算
【泛化层级】：条件性（迁移条件：所有涉及 UB 的 kernel；失效边界：纯标量/无 tile 的微型 kernel）

<原则：Ascend 每个 AI Core 的 UB 为 192KB。Tile 选择必须同时通过三重检查：UB 容量、512B 行宽对齐、输入布局（BTD/BDT）的连续访问方向。任一检查失败都会导致编译错误、UB overflow 或有效带宽骤降。>

**UB 容量预算公式**：

```
# elementwise / reduction：单 tile
BLOCK_SIZE * dtype_bytes <= 192KB

# matmul：同时驻留 A、B、C 三块，保守按 1/3 UB 给单块
BLOCK_M * BLOCK_K * dtype_bytes + BLOCK_K * BLOCK_N * dtype_bytes <= 192KB / 3
```

| dtype | 元素大小 | elementwise 最大 BLOCK | matmul 推荐 (M, N, K) |
|---|---|---|---|
| fp8 | 1B | 196608 | (256, 256, 256) |
| fp16 / bf16 | 2B | 98304 | (128/256, 128/256, 256) |
| fp32 | 4B | 49152 | (128, 128, 128/256) |

**标准 matmul 搜索空间**（已满足 512B 对齐 + UB 预算）：

```python
configs = [
    triton.Config({'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': 256, 'GROUP_SIZE': gs})
    for bm in [128, 256]
    for bn in [128, 256]
    for gs in [8, 16, 32]
]
```

**布局驱动的异构 Tile 策略**：

| 布局 | 连续维度 | 推荐策略 | 典型场景 |
|---|---|---|---|
| BTD `[B,T,D]` | D 在最内层 | D tile 大可连续 load：`BLOCK_D=512-1024` | Prefill，seq_len ≥ 512 |
| BDT `[B,D,T]` | T 在最内层 | D tile 小防跨步：`D_CHK_SIZE=16-32`；T tile 大：`T_CHK_SIZE=128-256` | Decode，seq_len 小 |

**坏实践（反例）**：
```python
# BDT 布局还大 D tile，导致跨步访问
BLOCK_D = 1024  # 行间距 = seq_len，cache-line 不连续
```
后果：每次 load 跨越大段内存，HBM burst 无法合并，`aiv_mte2_time` 显著升高。✅ 正解即 BDT 用异构小 D tile + 大 T tile。

**决策流程**：

```
1. 确定 dtype -> 计算 UB 最大 BLOCK
2. 检查尾轴是否 512B 对齐 -> 不对齐则 pad 或改 BLOCK
3. 确定布局 BTD/BDT -> 选择 D/T tile 大小
4. 对 matmul：BLOCK_K ≥ 512 优先 256；K < 256 降至 64-128
5. autotune 验证 UB 不溢出且 bandwidth 最高
```

## 已知实例（P3 挂载）

### Experience YAML 反链
- `exp_perf_001` — 块大小优化 （收益 20%）
  - 通用块大小调优原则适用于所有Triton算子开发，910b2/910b3平台需结合算子类型（元素级/矩阵级）选择2的幂次块大小，优先测试128/256/512三个档位
- `exp_causal_conv1d_update_007` — Scheduling Optimization （收益 BDT 布局 + 3D 任务分解 + 双缓冲状态更新：组合贡献约 30-40% 整体性能提升（vs BTD + naive tiling））
  - 此经验是关于 Update kernel 的整体架构设计决策，涵盖数据布局、tile 大小、Grid 策略、状态更新机制等维度。这些设计决策相互耦合：BDT 布局决定了 D_CHK_SIZE 必须小（避免跨步），小 D_CHK_SIZE 又决定了需要 T-tiling 来并行化…

### 边界扩展（近期实测，架构特异）
- **SelfAttention K3 tile 阶梯 + 编译降档（flash_attention + 投影 GEMM，ascend910b，2026-09-30）**
  - **候选 tile 阶梯优先保大 BSK、避免 UB fallback 骤减 BSK**：大 HD（BD≥224）的 attention 若走
    「BSK//=2 优先于 BSQ//=2」的纯 UB 收缩，会把 BSK 从 128 骤降到 32（KV 迭代数暴增）。改为
    候选阶梯 `[(64,128),(64,96),(64,64),(64,32),(32,128),…]` 按「面积最大可行」选、BSK 优先于 BSQ
    → 大 HD case 仍尽量保 BSK，KV 迭代不膨胀。
  - **编译降档回退**：tile 过大触发 `MLIRCompilationError`（host 侧 launch 同步抛）时，自动降档到
    阶梯下一档，替代「编译爆炸直接剔除」的离线静态裁剪（见 [tiling-candidate-compile-cost-gate]）。
    fp32 大 shape 用离线逐 shape 标定表 `{(S,D,H):(BSQ,BSK)}` 定点，避免每次运行时猜。
  - **证据**：opt_iter_2（纯 UB fallback）→ opt_iter_9（阶梯+表+降档+FULL_D 分裂）
    = 0.5577→0.7151（+28.2%，同卡 6；完整分析见
    online-softmax-flash-attention.md SelfAttention 互补 merge 实例）。

<!-- okf:related:start -->

# 相关

- 实践案例: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](../../ops/moe-token-permute-with-routing-map.md) — 同属优化知识，主题相关
- 实践案例: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](../../ops/nsa-compress-attention.md) — 同属优化知识，主题相关
- 实践案例: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](../../ops/quant-lightning-indexer.md) — 同属优化知识，主题相关
- 相关主题: [Triton API Quick Reference for Ascend Development](../../reference/triton-api-cheatsheet.md) — 主题相关：共享 autotune
- 相关主题: [累加/原地修改算子 autotune 必须每次重置输出 buffer](../autotuning/autotune-debug.md) — 主题相关：共享 autotune
- 相关主题: [autotune 必须探索 num_warps 与 num_stages](../autotuning/autotune-num-warps-stages.md) — 主题相关：共享 autotune
- 相关主题: [autotune 配置空间须覆盖 num_warps(2/4/8)×num_stages(2/3/4) 与 tile 组合](../autotuning/exp_perf_017.md) — 同属优化知识，主题相关
- 相关主题: [Tiling 候选须过编译成本闸：超大分块档位编译爆炸直接剔除，host 公式分档可替代 autotune](../autotuning/tiling-candidate-compile-cost-gate.md) — BLOCK 选择约束族：本卡补编译成本维度（UB 预算之外）
- 相关主题: [µs 级 kernel 禁用 autotune；num_stages/num_warps 在 Triton-Ascend 不生效](../autotuning/triton-ascend-microkernel-autotune-noise.md) — 主题相关：共享 autotune
- 相关主题: [matmul core mapping 策略选择（列/行/对角）](../compute-pipeline/core-mapping-strategy-selection.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: bdt layout rationale](../compute-pipeline/exp_causal_conv1d_update_007.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: autotune block m](../compute-pipeline/exp_gelu_bwd_001.md) — 同属优化知识，主题相关
- 相关主题: [流水图驱动优化：先可视化找气泡，按气泡边类型分派优化手段](../meta-skills/pipeline-diagram-driven-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../meta-skills/quant-matmul-autotune-strategy.md) — 主题相关：共享 autotune
- 相关主题: [形状驱动算子决策树](../meta-skills/shape-driven-operator-decision-tree.md) — 同属优化知识，主题相关
- 相关主题: [BLOCK_SIZE <65536 且子块内存 ≤L1 缓存，超限按维度多次嵌套切分](../portability/exp_ascend_split_001.md) — 同属优化知识，主题相关
- 相关主题: [波前调度 MatMul：对角遍历 + dot_pad_only_k + autotune tile 长宽比](../portability/exp_matmul_ascend_wavefront_001.md) — 同属优化知识，主题相关
- 相关主题: [layout (B,T,H,D)→(B,H,T,D) 消除跨 Head 离散访存](layout-bthd-to-bhtd.md) — 同属优化知识，主题相关
- 相关主题: [UB 占用决定 BLOCK_ROWS/tile 大小，必须按算子特征分别自适应](ub-aware-block-adaptive.md) — 主题相关：共享 ub_budget

<!-- okf:related:end -->
