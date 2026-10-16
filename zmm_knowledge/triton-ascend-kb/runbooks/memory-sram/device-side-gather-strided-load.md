---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: "1.2"
title: device-side gather 消除 strided tile load（访存结构重组，本例 89x）
description: kernel 内 strided tile load（小 segment 大 stride + 多次重读）的 per-instruction issue 开销可占总延迟 99%；用 device-side gather kernel 把数据物化为连续 workspace，后续 kernel 全连续加载，是复合块算子的首要杠杆。
tags: [device_side_gather, strided_load, workspace, mte_bound, memory_restructure, opt23]
confidence: draft
status: draft
created_at: "2026-08-26T00:00:00Z"
updated_at: "2026-08-26T00:00:00Z"
---

## device-side gather 访存重组模式（来源：53_MobileViTAttention R1，0.0158→1.4064 单点 89x）

【泛化层级】：条件性（迁移条件：kernel 存在跨大步长的小段 strided load 且被重复读取；失效边界见末节）

**机制（约束）**：strided tile load（如 256B segment、6KB stride、12x 重读）的 per-instruction issue 开销与数据量不成比例——本例占总延迟 99%。识别信号：msprof 单 kernel 直采 **MTE2+MTE3 占比 ~83%（纯搬运 bound）**且 **CUBE < 5%**（排除算力极限）。此类浪费 simulator 可明确区分于"算力极限"：MMAD < 50% ≠ 无优化空间。

**判别法**：
1. msprof 单 kernel 直采（非整图），看 MTE2+MTE3 占比与 CUBE 占比；
2. 微基准对照：同数据量 contiguous vs strided load 的延迟比（本例实测确认标量化/片段化访存为瓶颈）。

**复刻规则**：
```python
# _qkv_gather_kernel：按 (task, head) 将 Q/K/V 物化为连续 workspace
# 后续 attention kernel 全部连续加载，消除 strided issue 开销与重读
# UB 预算定量式（决定窗口上限）：
#   direct/onekv attn 常驻 = 2×np2×64×4B (k+v) + 2×BQ×np2×4B (scores+pr) ≤ ~192KB
#   np2=512 起 KV 常驻结构性不可行；np2=256 时 BQ 须 ≤32（估 ~208KB 边缘）
```

**失效边界**：数据本就连续存储；workspace 超 UB（改用窗口分块或直接直读）；host 端布局调整（reshape/permute 零拷贝视图）即可消除 strided 的更简单情形——优先 host 视图，gather 是 device 端兜底。

**已知实例（host 侧布局重组变体，正向）**：grid_sample（gather_scatter，ascend910b/Ascend910_9372）——host 侧 NCHW→NHWC permute+contiguous（+32 元素冗余分配防列越读）使 kernel 内随机 gather 变为 C 维段连续段 load，geomean 0.0060→0.2076（34.6×，vs aclnn 注入基线口径）；小 C 不豁免：C=4/8 实测 NHWC 仍快 28-33×，证伪"C≤8 退化 NCHW"猜测，阈值下沉 _NHWC_MIN_C=1（2026-09-17；证据：op_12_12_grid_sample_20260917_1453_7647/output/opt_iter_0/log.md + opt_iter_1_ir_1/optimized_perf_result.json.speedup_vs_torch）

<!-- okf:related:start -->

# 相关

- 相关主题: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../meta-skills/attention-fusion-failure-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend strided-2/interleave 访存契约（三个正确性缺陷 + 寄存器重排正解）](triton-ascend-strided2-interleave-contract.md) — 主题相关：共享 strided_load

<!-- okf:related:end -->
