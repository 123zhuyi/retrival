---
query_type: codegen_init
query_keywords: [groupnorm, swish, normalization, fp16, fp32, bf16, fused]
created: 2026-06-08
last_updated: 2026-06-08T11:23:00
hit_count: 2
context_summary: "GroupNormSwish 融合算子代码生成 — GroupNorm + Swish 激活单 kernel 融合，支持 2-8D 张量"
top_units: [C1, C2, C5, OP, A1, A3, A4, B_norm_1, E_groupnorm_001, E_swiglu_001, T1, T2, T3, S_groupnorm_swish]
stale: false
---

## 检索问题（Problem Statement）

为 GroupNormSwish 融合算子生成高性能 Triton-Ascend kernel。算子语义：
1. 输入 x 按 num_groups 分组
2. 每组计算 mean 和 var
3. 标准化：x_norm = (x - mean) / sqrt(var + eps)
4. 仿射变换：x_norm * weight + bias
5. Swish 激活：x_norm * sigmoid(swish_scale * x_norm)
6. 同时输出 mean 和 rstd

**触发上下文**：代码生成初始化，从零构建融合算子知识包
**检索范围**：groupnorm + swish + normalization + elementwise_fusion

---

## 知识单元列表

### 硬件约束知识单元

知识单元 C1  [硬件约束]  UB 容量上限与 tile 规划
  UB 192 KB，fp16 最大 ~98304 elements；fp32 最大 ~49152 elements。
  GroupNormSwish 每 program 处理一个 group 的 channels_per_group × HW 个元素。
  当 HW 很大时需分块处理（for-loop tiling）。
  来源: hardware/constraints_cheatsheet.md C1

知识单元 C2  [硬件约束]  Grid 大小与 multi-round 调度
  grid 维度乘积 ≤ 65535。GroupNormSwish 的 grid = N × num_groups。
  当 N × num_groups > 65535 时需用 persistent kernel 或 grid-stride loop。
  来源: hardware/constraints_cheatsheet.md C2

知识单元 C5  [性能约束]  int64 CMP/ADD scalar fallback
  int64 比较和加法退化为标量路径。tl.arange 返回 int64，比较前需 .to(tl.float32)。
  来源: hardware/constraints_cheatsheet.md P1/P2

---

### 算子分类知识单元

知识单元 OP  [算子分类]  operator_class = "normalization + elementwise_fusion"
  判断依据：GroupNorm（mean/var/归一化/仿射）+ Swish（sigmoid + mul）融合
  详细页面：pages/ops/normalization.md, pages/ops/elementwise.md
  经验目录：raw/experiences/normalization/, raw/experiences/activation/
  融合策略：GroupNorm 与 Swish 可在同一 kernel 内完成，消除中间 HBM 写回

---

### API 知识单元

知识单元 A1  [API用法]  tl.load / tl.store 与 care_padding=False
  tl.load(..., care_padding=False) 消除 MTE2→Vector 同步屏障，提升 ILP。
  GroupNormSwish 中当 padding 值被 mask 过滤时安全使用。
  API文档: raw/triton-ascend-develop-guide/triton_api/Memory_Pointer_Ops/

知识单元 A3  [API用法]  tl.sum 归约操作
  tl.sum(x, axis=0) 用于计算 group 内元素的和。fp16/bf16 输入必须先转 fp32 再 sum。
  API文档: raw/triton-ascend-develop-guide/triton_api/Reduction_Ops/

知识单元 A4  [API用法]  tl.rsqrt 替代 1/sqrt
  inv_std = tl.rsqrt(var + eps) 比 1.0 / tl.sqrt(var + eps) 更快，使用硬件 rsqrt 指令。
  API文档: pages/ops/normalization.md

知识单元 A5  [API用法]  tl.sigmoid + Swish 激活
  Swish(x) = x * tl.sigmoid(swish_scale * x)。必须在 fp32 accumulator 上施加，不提前 downcast。
  API文档: raw/triton-ascend-develop-guide/triton_api/Math_Ops/

知识单元 A10 [API文档]  tl.constexpr — 编译时常量
  使用场景：HAS_WEIGHT, HAS_BIAS, CHANNELS_PER_GROUP 等编译期分支消除。
  API文档: raw/triton-ascend-develop-guide/triton_api/triton/

---

### 瓶颈预判知识单元

知识单元 B_norm_1  [潜在瓶颈]  fp16 累加溢出
  sum(x) 和 sum(x^2) 在 fp16 下对大 group_size 易溢出。必须在 fp32 下累加。
  关联经验: E_groupnorm_001
  修复方向: techniques/compute_and_branch.md

知识单元 B_norm_2  [潜在瓶颈]  weight/bias 重复加载
  每个 group 处理时从 HBM 加载 weight/bias，造成冗余访问。
  可通过 persistent kernel 缓存或 grid-stride 减少加载次数。
  关联经验: exp_rmsnorm_001

知识单元 B_swish_1  [潜在瓶颈]  Swish 激活 fp16 精度损失
  Swish 的 sigmoid 在 fp16 下对大值精度不足，必须在 fp32 accumulator 上施加。
  关联经验: E_swiglu_001
  修复方向: techniques/compute_and_branch.md

---

### 调优经验知识单元

知识单元 E_groupnorm_001  [调优经验]  exp_groupnorm_001 — GroupNorm 向量化 + fp32 累加
  适用条件：GroupNorm，fp16/bf16，channels_per_group 可变
  性能提升：25%
  核心优化：fp32 累加 X_sum 和 X_var_sum + 三趟算法（Mean → Variance → Normalization）
  经验文件：raw/experiences/normalization/exp_groupnorm_001.yaml

知识单元 E_swiglu_001  [调优经验]  exp_swiglu_fwd_001 — Swish/SiLU fp32 中间精度策略
  适用条件：Swish/SiLU 激活，fp16/bf16
  性能提升：15%
  核心优化：sigmoid 在 fp32 accumulator 上施加，不提前 downcast 至 fp16
  经验文件：raw/experiences/activation/exp_swiglu_fwd_001.yaml

知识单元 E_online_mean_var_001  [调优经验]  Welford 在线算法
  适用条件：需要稳定计算均值和方差
  性能提升：30%
  核心优化：单遍稳定计算均值和方差，避免数值不稳定
  经验文件：raw/experiences/normalization/exp_online_mean_var_001.yaml

---

### 调优策略知识单元

知识单元 T1  [调优策略]  BLOCK_SIZE 选择
  适用场景：HW 维度分块处理
  核心方法：推荐值 {128, 256, 512, 1024, 2048, 4096}，根据 HW 大小选择
  页面：pages/techniques/tiling.md

知识单元 T2  [调优策略]  Persistent Kernel + grid-stride
  适用场景：N × num_groups >> num_aicores
  核心方法：grid = min(N*num_groups, 65535)，每个 program 通过 grid-stride 处理多个 group
  页面：pages/techniques/persistent_kernels.md

知识单元 T3  [调优策略]  care_padding=False 启用 ILP
  适用场景：所有 tl.load 调用
  核心方法：tl.load(..., care_padding=False) 消除 MTE2→Vector 同步屏障
  页面：pages/techniques/ilp.md

知识单元 T4  [调优策略]  融合消除中间 Tensor
  适用场景：GroupNorm + Swish 连续执行
  核心方法：将 Swish 激活内联进 GroupNorm kernel，消除独立的 elementwise kernel
  页面：pages/techniques/operator_fusion.md

---

### 代码骨架知识单元

知识单元 S_groupnorm_swish  [代码骨架]  GroupNormSwish 融合 kernel
  适用任务：GroupNorm + Swish 激活融合
  关键结构：
    1. Persistent kernel 模式（grid = min(N*num_groups, 65535)）
    2. 每个 program 处理一个 (n, g) pair，通过 grid-stride 处理多个
    3. Pass 1: 加载 group 数据到 fp32，计算 mean
    4. Pass 2: 计算 var（或在线 Welford 单遍）
    5. Pass 3: normalize + weight/bias + Swish，store 结果
    6. 同时 store mean 和 rstd 到输出 buffer
  骨架参考：pages/ops/normalization.md + pages/ops/elementwise.md

---

### 矛盾与注意事项

知识单元 W1  [矛盾提示]  GroupNorm 与 Swish 融合精度要求
  GroupNorm 的 mean/var 计算和 Swish 的 sigmoid 都需要 fp32 精度。
  融合时不能在任何中间步骤 downcast 到 fp16，必须在最终 store 时才 cast。

知识单元 W2  [调试建议]  多维张量 stride 计算
  输入支持 2-8D 张量，需正确计算 group 内元素的 stride 和 offset。
  channels_per_group = C // num_groups，每组处理 channels_per_group × HW 个元素。
