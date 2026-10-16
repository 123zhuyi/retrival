---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: "1.2"
title: UB 占用决定 BLOCK_ROWS/tile 大小，必须按算子特征分别自适应
description: 不同算法 live_buffers 差异大，必须按 UB 估算为每类 kernel 提供 BLOCK 选择函数，禁止共用单一函数
tags: [ub_budget, block_rows_adaptive, live_buffers, tile_size]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-10-10T00:00:00Z"
---

## OPT-5 UB-aware BLOCK 自适应

【优化维度】：内存｜Tiling·SRAM 预算
【泛化层级】：条件性（迁移条件：不同算子 live_buffers_count 差异大；失效边界：所有算子 live_buffers 相同时可共用）

<原则：UB 估算规则 BLOCK_ROWS * L * dtype_itemsize * live_buffers_count <= UB_TOTAL。不同算法 live_buffers_count 差异大（ext.sort ~1 buffer，软件 bitonic ~8 buffers，FFT butterfly 2-3 buffers），必须分别提供 UB-aware 的 BLOCK 选择函数。目标 grid=cdiv(n_rows,BLOCK_ROWS) <= num_cores。>

<骨架（占位名伪码）：
UB_TOTAL = 192 * 1024  # 192KB（按芯片/场景调整）
live_buffers = {"ext_sort": 1, "bitonic": 8, "fft": 3}[algo]
dtype_bytes = 4 if dtype == fp32 else 2
max_block_rows = UB_TOTAL // (L * dtype_bytes * live_buffers)
BLOCK_ROWS = min(max_block_rows, n_rows) if n_rows > num_cores else 1
>

**坏实践（反例）**：所有算子共用单一 BLOCK 函数
```python
def block_rows(L, dtype):       # 未考虑 live_buffers 差异
    return UB_TOTAL // (L * dtype_bytes)
# bitonic 8 buffers 用这个会 UB overflow
```
后果：live_buffers 多的算子 UB overflow，寄存器压力，性能退化。✅ 正解即按算法特征分别自适应。

**已知实例**：math-compute M4（Sort _block_rows_for_extsort UB=8192 fp32 / _block_rows_for_L 50KB 8 buffers 分离、FFT BLOCK_SIZE_B=64）；Gdn2Fwd（状态递推，ascend910b1 UB=192KB）——fp32 状态 [BK,BV] 驻留 UB 的实测边界：[128,128] fp32 + 输入向量 + broadcast 中间量编译报 ub overflow（2123776 bits，PlanMemory Failed）；[128,64] fp32（32KB）编译通过且为性能最优点，BV cap=64 平衡点（cap=32 劣化 24%）（2026-09-14；证据：op_94_94_Gdn2Fwd_20260914_1219_8124/report.md + `.claude/template/linear-recurrent.md` L1.8）；CrossformerAttention（attention_standard_flash，ascend910b，2026-10-10）——flash 注意力大 head_dim 下 O 累加器 BLOCK_M×BLOCK_D fp32 是 UB 主导项：BD=512 fp16/bf16 的 BLOCK_M 结构性≤16（BM=16,BN=16 → 大 D case ~2x 零回退；BM=32,BN=16 → O=64KB 挤垮 UB，12 case triton compile overflow，证据: op_0_15_CrossformerAttention_20261010_1653_8516/output/opt_iter_3/verify/verify_result_optimized.json passed=34/46）；BN 过大会挤垮 D9 流水（BD=512 BN=64 → kt+vv=128KB 占满 UB 关 auto multi-buffer，case7 慢 10x，BN 回 16 反升 2x）；整体几何平均 0.0032→0.0043（证据: output/iter_1/perf_result.json.speedup_vs_torch → output/opt_iter_2/optimized_perf_result.json.speedup_vs_torch）

<!-- okf:related:start -->

# 相关

- 相关主题: [Memory Access Optimization: adaptive tiling strategy](exp_layernorm_bwd_001.md) — 同属优化知识，主题相关
- 相关主题: [Tiling 预算与布局驱动的块大小选择](tiling-budget-and-layout-driven-selection.md) — 主题相关：共享 ub_budget

<!-- okf:related:end -->
