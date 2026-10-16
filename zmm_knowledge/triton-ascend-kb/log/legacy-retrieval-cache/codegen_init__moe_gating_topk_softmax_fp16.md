---
query_type: codegen_init
query_keywords: [moe, gating, topk, softmax, fp16]
created: 2026-06-08
last_updated: 2026-06-10T14:21:00+08:00
hit_count: 3
context_summary: "MoE Gating TopK Softmax 算子 — per-row softmax + topk selection + finished mask + row_idx generation"
top_units: [C1, C5, OP1, A3, B1, E1, T1, S1]
stale: false
---

## 检索问题（Problem Statement）

为 MoeGatingTopKSoftmax 算子生成高性能 Triton-Ascend kernel，实现 per-row softmax + top-k 选择 + finished mask 处理 + row_idx 生成。

**触发上下文**：代码生成初始化，从零构建算子知识包  
**检索范围**：moe_gating_topk_softmax_fp16，per-row 操作，num_experts 通常 8-64

---

## 知识单元列表

### 硬件约束知识单元

知识单元 C1  [硬件约束]  UB 总用量 ≤ 192KB
  每个 program 的 tile 数据量之和不超过 192KB（Atlas 800T A2）。
  fp16: 单 tile ≤ 98304 elements；fp32: ≤ 49152 elements。
  本算子每行 num_experts 通常 8-64，远小于 UB 限制，可完整加载整行。
  来源: hardware/constraints_cheatsheet.md C1

知识单元 C2  [硬件约束]  Grid 维度上界 65535
  grid 必须是 tuple，各维度乘积 ≤ 65535。超出需用 grid-stride loop。
  本算子 batch_size 可能很大，需考虑 persistent kernel 或 grid-stride loop。
  来源: hardware/constraints_cheatsheet.md C2

知识单元 C3  [硬件约束]  512B 行宽对齐
  matmul 的块行宽（BLOCK_N × dtype_bytes）必须是 512B 整数倍。
  本算子无 matmul，不涉及此约束。
  来源: hardware/constraints_cheatsheet.md C3

知识单元 C4  [硬件约束]  禁止 tl.swizzle2d
  Ascend 上行为未定义，替换为 grouped_launch_diagonal 或对角 tiling。
  来源: hardware/constraints_cheatsheet.md C4

知识单元 C5  [性能约束]  int64 CMP/ADD scalar fallback
  int64 的比较和加法会退化为标量路径，导致 aiv_scalar_ratio 升高。
  解决：用 int32 或先 .to(tl.float32) 再做比较/加法。
  来源: hardware/constraints_cheatsheet.md P1/P2

知识单元 C6  [性能约束]  fp16/bf16 accumulation 未升精度
  fp16/bf16 累加导致数值溢出/精度问题。
  解决：softmax 的 exp/sum 在 fp32 中进行，最终结果 cast 回 fp16。
  来源: hardware/constraints_cheatsheet.md P7

---

### 算子分类知识单元

知识单元 OP1  [算子分类]  operator_class = "fused_softmax_topk"
  判断依据：forward 包含 softmax(inner dim) + topk + where(mask) + arange/reshape
  详细页面：pages/ops/softmax.md + pages/ops/gather_scatter.md
  经验目录：raw/experiences/softmax/ + raw/experiences/gather_scatter/

---

### API 知识单元

知识单元 A1  [API用法]  tl.load / tl.store
  用于从 HBM 加载每行数据到寄存器，以及写回 topk_values/topk_indices/row_idx。
  每行 num_experts 较小，可完整加载整行到寄存器。
  Ascend 特有：使用 care_padding=False 消除 MTE2→Vector 同步屏障（仅当 padding 元素被 store mask 过滤时安全）。
  推荐：tl.make_block_ptr + tl.advance 替代手动 2D 偏移。
  API文档: raw/triton-ascend-develop-guide/triton_api/Memory_Pointer_Ops/

知识单元 A2  [API文档]  tl.max / tl.sum / tl.reduce — 归约
  softmax 求 max 和 sum 的核心 API。
  tl.max(x, axis=0) 获取每行最大值。
  tl.sum(x, axis=0) 获取每行指数和。
  API文档: raw/triton-ascend-develop-guide/triton_api/Reduction_Ops/

知识单元 A3  [API用法]  tl.where — 条件选择
  用于处理 finished mask：finished 为 True 时，topk_indices 设为 num_experts（sentinel）。
  tl.where(condition, true_val, false_val)
  API文档: raw/triton-ascend-develop-guide/triton_api/Indexing_Ops/

知识单元 A4  [API用法]  tl.exp — 指数函数
  softmax 核心计算：exp(x - max_x)。
  必须在 exp 前先减去行内最大值防止溢出。
  API文档: raw/triton-ascend-develop-guide/triton_api/Math_Ops/

知识单元 A5  [API文档]  tl.constexpr — 编译时常量
  用于 BLOCK_SIZE, NUM_EXPERTS, K 等参数。
  使用场景：HAS_FINISHED, NUM_EXPERTS, K 等布尔/整数开关。
  API文档: raw/triton-ascend-develop-guide/triton_api/triton/

知识单元 A6  [API用法]  tl.atomic_max / tl.sort — 不推荐使用
  Ascend Triton 对 tl.sort 支持有限，topk 建议用手动比较实现。
  对于小 k（k ≤ 8），用循环比较实现 topk 比调用 sort 更高效。

---

### 瓶颈预判知识单元

知识单元 B1  [潜在瓶颈]  fp16 exp 溢出
  触发条件：x 不减 max 直接 exp
  症状: inf/NaN 输出
  修复: 先减行内最大值（数值稳定化）
  关联经验: exp_stability_001
  修复方向: techniques/numerical_stability.md

知识单元 B2  [潜在瓶颈]  topk 在小维度效率低
  触发条件：num_experts ≤ 64（MoE 典型场景）
  症状: 分支多，向量化利用率低
  修复: 手动实现 selective topk（维护 k 个最大值及其索引），避免完整排序
  关联经验: exp_moe_gather_001

知识单元 B3  [潜在瓶颈]  finished mask 引入分支
  触发条件：finished 为 None 时无 mask
  症状: 分支不平衡，部分核空转
  修复: 用 tl.where 替代 if/else，或 constexpr HAS_FINISHED 消除分支
  关联经验: exp_fused_gating_001

知识单元 B4  [潜在瓶颈]  row_idx 生成时的 int64 索引
  触发条件：batch_size * k 可能超过 int32 范围
  症状: aiv_scalar_ratio 升高
  修复: 使用 int32 分步计算 arange

---

### 调优经验知识单元

知识单元 E1  [调优经验]  exp_softmax_inner_002 — 向量化内层 Softmax
  适用条件：inner dim N ≥ TILE_N，fp16/bf16/fp32
  性能提升：30%
  核心优化：向量化 reduction 策略，在线 max+sum 单趟，care_padding=False
  适用阶段：softmax 核心计算
  经验文件：raw/experiences/softmax/exp_softmax_inner_002.yaml
  详细页面：pages/ops/softmax.md
  composite_score: 75, status: stable

知识单元 E2  [调优经验]  exp_softmax_online_001 — Online Softmax 算法
  适用条件：BLOCK_M ≥ 64, BLOCK_N ≥ 64，fp16/bf16
  性能提升：35%
  核心优化：迭代更新 running max 和 sum，单趟完成，消除完整矩阵物化
  适用阶段：softmax 核心计算（大维度时）
  经验文件：raw/experiences/softmax/exp_softmax_online_001.yaml
  详细页面：pages/ops/softmax.md
  composite_score: 80, status: stable

知识单元 E3  [调优经验]  exp_fused_gating_001 — 门控激活融合
  适用条件：门控操作，fp16/bf16
  性能提升：25%
  核心优化：operator_fusion + 数值稳定化 + precision_upcasting
  适用阶段：gating 逻辑（softmax + topk 融合）
  经验文件：raw/experiences/activation/exp_fused_gating_001.yaml
  详细页面：pages/ops/elementwise.md
  composite_score: 70, status: stable

知识单元 E4  [调优经验]  exp_moe_gather_001 — MoE Gather 优化
  适用条件：topk_num ≤ 8，BLOCK_D ≥ 128
  性能提升：20%
  核心优化：vectorized metadata loading + loop unrolling + memory alignment
  适用阶段：topk 索引处理
  经验文件：raw/experiences/gather_scatter/exp_moe_gather_001.yaml
  详细页面：pages/ops/gather_scatter.md
  composite_score: 65, status: stable

知识单元 E5  [调优经验]  exp_stability_001 — 数值稳定性
  适用条件：所有含 exp 的算子
  核心优化：exp 前减 max，fp32 accumulator
  经验文件：raw/experiences/general_stability/exp_stability_001.yaml
  composite_score: 90, status: stable

---

### 调优策略知识单元

知识单元 T1  [调优策略]  数值稳定化（减 max + fp32 accumulator）
  适用场景：所有 softmax 类算子
  核心方法：
    1. 先求行内 max：m = tl.max(x)
    2. 计算稳定 exp：exp(x - m)
    3. sum 在 fp32 中累积
    4. 最终除以 sum 后 cast 回输入 dtype
  关键代码：
    ```python
    x_f32 = x.to(tl.float32)
    m = tl.max(x_f32, axis=0)
    e = tl.exp(x_f32 - m)
    s = tl.sum(e, axis=0)
    out = (e / s).to(x.dtype)
    ```
  页面：techniques/numerical_stability.md

知识单元 T2  [调优策略]  per-row kernel（1D grid）
  适用场景：每行独立计算，batch 维度大
  核心方法：
    1. 1D grid，每个 program 处理一行
    2. grid-stride loop 处理超出核数的 batch
    3. num_experts 较小可完整加载整行
  关键参数：BLOCK_SIZE = num_experts（向上取到 2 的幂次）
  页面：techniques/parallelism_and_loop.md

知识单元 T3  [调优策略]  小 k loop unrolling（topk 优化）
  适用场景：k 较小（k ≤ 8）
  核心方法：
    1. 维护 k 个 (value, index) 对
    2. 遍历所有 experts，逐个比较更新 topk
    3. k 为 constexpr，编译器可自动 unroll
  关键代码：
    ```python
    topk_vals = tl.full([K], float('-inf'), tl.float32)
    topk_idx = tl.zeros([K], tl.int32)
    for e in range(NUM_EXPERTS):
        val = softmax_vals[e]
        # 与最小值比较，更新
        if val > topk_vals[K-1]:
            # 插入排序更新 topk
    ```
  页面：techniques/ilp.md

---

### 代码骨架知识单元

知识单元 S1  [代码骨架]  softmax inner dim 骨架
  适用任务：per-row softmax（内层维度归约）
  骨架路径：pages/ops/softmax.md（Online Softmax 算法）
  关键结构：
    1. 1D grid，每 program 一行
    2. 在线算法：单趟 max+sum
    3. 第二趟：归一化 + 写回

知识单元 S2  [代码骨架]  MoE Dispatch 骨架
  适用任务：MoE scatter + gather
  骨架路径：pages/skeletons/moe_dispatch.py
  关键结构：三阶段结构（dispatch → expert GEMM → gather）
  注意：本算子只涉及 gating 阶段，不调用完整 MoE dispatch

---

## 复用提示

以下 query_keywords 变体可部分复用本缓存：
- `[moe, gating, softmax, fp16]` — 精确/相似命中
- `[softmax, topk, fp16]` — 相似命中（复用 C/A/B/E/T 系列，替换 OP/S 系列）
- `[moe, routing, fp16]` — 相似命中（复用 C/E4/T2 系列，替换 A/B 系列）

---

## 附加：precision_debug 检索结果（迭代修复补丁）

**追加时间**: 2026-06-11
**触发**: iter_2 precision 验证失败
**缓存文件**: cache/precision_debug__topk_tie_softmax_fp16_bf16.md

### 关键修复知识单元

- **P0_exp_precision_001**: 必须在 fp32 softmax 概率上做 topk，而非 logits
- **P1_tie_break_left_001**: `tl.max(return_indices=True, tie_break_left=True)` 匹配 PyTorch tie-breaking
- **P1_topk_logit_softmax_order_001**: ties 时 logits 排序和 softmax 排序可能不同
