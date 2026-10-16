# 知识检索七步流程

> 本文档定义 triton-knowledge-retrieval Skill 的核心检索流程。
> 四种检索模式 (codegen_init / perf_tuning / compile_error / precision_debug)
> 均以此为骨架，按需调整 Step 权重。

---

## 检索模式路由

| 检索模式 | 触发场景 | 重点 Step | 弱化/跳过 |
|---------|---------|----------|----------|
| `codegen_init` | 从零生成 Triton Ascend kernel | Step 0.5,1,2,3,4,6 全量 | — |
| `perf_tuning` | 已有正确 kernel，优化性能 | Step 0,0.5,1,4,5 | Step 3 (API 已知), Step 6 (骨架已知) |
| `compile_error` | 编译/语法/API 报错 | Step 0 → 0.5 → 错误路由 | Step 2-6 按需 |
| `precision_debug` | 精度/NaN/cos_sim 不达标 | Step 0 → 0.5 → 精度路由 | Step 2-6 按需 |

---

## Step 0: 缓存命中检查 (必做)

### 目的
避免对相同/相似算子重复检索，直接复用已验证的知识单元列表。

### 操作
1. 从 task_desc 提取关键词 (算子名、核心操作、shape 特征、dtype)
2. 构造缓存查找 key: `{query_type}__{kw1}_{kw2}_..._{kwn}.md`
3. 查找 `knowledge/cache/` 目录

### 匹配规则
```
精确匹配: 文件名完全一致
  → 检查 frontmatter stale 字段
    - stale: false → 直接输出缓存内容，SKIP Step 1-6
    - stale: true → 跳过此缓存，执行完整检索

相似匹配: 关键词重叠 ≥ 60%
  → 读取最相关缓存文件
  → 继承通用知识单元 (C/OP/A 系列)
  → 仅对差异部分执行对应 Step (如 E/T 系列根据新 shape/dtype 重新匹配)
  → 合并输出完整知识单元列表

无匹配:
  → 执行完整 Step 1-6
```

### 缓存 key 示例
```
codegen_init__matmul_fp16_m1024_n512_k256.md
codegen_init__rmsnorm_fp32_batch32_seq128_hidden768.md
compile_error__tl_load_negative_offset.md
precision_debug__groupnorm_fp16_nan.md
perf_tuning__matmul_cube_underutilized.md
```

---

## Step 0.5: 问题分解 (可选，复合问题时启用)

### 目的
将复合问题（同时涉及编译+精度+性能中的两类以上）拆解为独立子问题，各子问题独立检索后交叉关联聚合。

### 触发条件 (任一满足)
- 问题描述同时涉及编译+精度 / 编译+性能 / 精度+性能 两类以上症状
- 上一轮检索结果不足 10 条知识单元
- Profiling 结果与预测瓶颈偏差超过 2 个维度
- 用户明确表示当前检索结果不完整

### 操作
加载 `retrieval/problem_decomposer.md`，执行:
1. **问题分类** — 按三维分类体系 (编译/精度/性能) 标记症状
2. **子问题拆解** — 每种类型生成独立子问题，各自按优先路由检索
3. **依赖排序** — [编译问题] → [精度问题] → [性能问题]
4. **交叉关联** — 因果/冲突/互补/泛化四类关联分析
5. **上下文聚合** — P0-P3 优先级聚合 + 冲突解决

若只有单一问题类型，跳过此步骤，直接执行 Step 1-6。

---

## Step 1: 约束预检 (必做)

### 目的
确保每个 kernel 满足 Ascend NPU 的硬件和性能约束，避免编译错误和严重性能退化。

### 操作
读取 `knowledge/pages/hardware/constraints_cheatsheet.md`，为当前算子提取所有适用约束。

### 必出约束 (C1-C5)

```
知识单元 C1 [硬件约束] UB 总用量 ≤ 192KB
  每个 program 的 tile 数据量之和不超过 192KB (Atlas 800T A2)。
  fp16: 单 tile ≤ 98304 elements；fp32: ≤ 49152 elements。
  matmul 同时持有 A/B/C tile，实际可用约 1/3，约 64KB per matrix。

知识单元 C2 [硬件约束] Grid 维度上界 65535
  grid 必须是 tuple，各维度乘积 ≤ 65535。超出需用 grid-stride loop。

知识单元 C3 [硬件约束] 512B 行宽对齐
  matmul 的块行宽 (BLOCK_N × dtype_bytes) 必须是 512B 整数倍。
  fp16/bf16: BLOCK_N 须为 256 的倍数；fp32: 128 的倍数。

知识单元 C4 [硬件约束] 禁止 tl.swizzle2d
  Ascend 上行为未定义，替换为 grouped_launch_diagonal 或对角 tiling。

知识单元 C5 [性能约束] int64 CMP/ADD scalar fallback
  int64 比较和加法会退化为标量路径，导致 aiv_scalar_ratio 升高。
  用 int32 或先 .to(tl.float32) 再做比较/加法。
```

### 条件性约束

| 算子类型 | 额外约束 |
|---------|---------|
| matmul | P6: 大矩阵(M,N≥512) 行主序 tiling → L2 bank conflict，需对角 tiling |
| normalization / elementwise | P3: 无 care_padding=False 时 MTE2→Vector 依赖，ILP 损失 |
| 所有 | P7: fp16/bf16 累加未升精度 → 数值溢出，需 fp32 accumulator |
| 大 batch / 多 row | P4: block 数 >> 核数 → 多轮调度开销，需 persistent kernel |

---

## Step 2: 算子分类 (必做)

### 目的
确定算子类型，为后续 Step 3-6 的按需检索提供方向。

### 分类决策表

| 判断信号 | 算子类型 | 详细页面 |
|---------|---------|---------|
| `tl.dot` / mm / bmm / linear / matmul | matmul | `pages/ops/matmul.md` |
| softmax(QK^T) @ V / attention / flash_attn | attention | `pages/ops/attention.md` |
| mean/var/rms + 归一化 / LayerNorm / RMSNorm / GroupNorm | normalization | `pages/ops/normalization.md` |
| sum/max/min(dim=) / logsumexp | reduction | `pages/ops/reduction.md` |
| gather/scatter / 索引计算指针 / embedding | gather_scatter | `pages/ops/gather_scatter.md` |
| per-element，输入输出同形状 / add/mul/relu/gelu | elementwise | `pages/ops/elementwise.md` |
| cos/sin 旋转位置编码 | rope | `pages/ops/rope.md` |
| softmax / log-softmax | softmax | `pages/ops/softmax.md` |
| cross-entropy / label smoothing | loss | `pages/ops/loss.md` |
| SSM / Mamba / chunk scan | ssm | `pages/ops/ssm.md` |
| 卷积 / conv1d | convolution | `pages/ops/convolution.md` |
| 多种混合 | fused | `pages/ops/fused_ops.md` |

### 输出格式
```
知识单元 OP [算子分类] operator_class = "<类型>"
  判断依据: <信号>
  详细页面: pages/ops/<op>.md
  经验目录: raw/experiences/<category>/
```

---

## Step 3: API 知识单元提取 (必做，perf_tuning 模式跳过)

### 目的
列出 kernel 实现中必然用到的 API，标注 Ascend 特有约束。

### 按算子类型的 API 关联表

| 算子类型 | 必列 API |
|---------|---------|
| matmul | tl.load/store, tl.dot, dl.async_task, dl.compile_hint, autotune, constexpr |
| attention | tl.load/store, tl.dot, tl.sum/reduce, tl.exp/rsqrt, autotune, constexpr |
| normalization | tl.load/store, tl.sum, tl.rsqrt, dl.compile_hint, autotune, constexpr |
| elementwise/rope | tl.load/store, tl.exp/sqrt/rsqrt/cos/sin, constexpr |
| reduction | tl.load/store, tl.sum/max/reduce, tl.atomic_add/max, constexpr |
| gather_scatter | tl.load/store, tl.where/gather, tl.atomic_add, constexpr |
| softmax/loss | tl.load/store, tl.sum, tl.exp/log, constexpr |

### API 知识单元模板
```
知识单元 A{N} [API用法] {API 名称}
  用于: {使用场景}
  Ascend 特有约束: {不同于 GPU Triton 的限制}
  推荐: {最佳实践}
  文档: raw/triton-ascend-develop-guide/triton_api/{分类}/
```

---

## Step 4: 瓶颈预判与调优经验匹配 (必做)

### 4.1 瓶颈知识单元
为当前算子类型输出已知瓶颈，格式:
```
知识单元 B{N} [潜在瓶颈] {瓶颈名称}
  触发条件: {条件}
  症状: {profiling 指标异常}
  修复方向: {技术页}
  关联经验: {经验 ID 列表}
```

### 4.2 调优经验评分规则

| 匹配条件 | +分 | 冲突条件 | -分 |
|---------|----|---------|----|
| 同 operator family | +3 | 建议拆多 kernel | -4 |
| 同 compute pattern | +2 | 会改变用户语义 | -4 |
| 同 shape regime | +2 | 只适用于特化语义但当前无 | -3 |
| 同 dtype | +2 | shape/dtype 显著不匹配 | -2 |
| 直接覆盖当前瓶颈 | +2 | — | — |
| 同 Ascend 代际 | +1 | — | — |

### 4.3 质量权重 (v4 新增)
从 `raw/experiences/_index.yaml` 读取每条经验的 status 和 composite_score:

| status + composite_score | quality_weight_coefficient |
|--------------------------|---------------------------|
| stable + composite ≥ 80 | × 1.3 |
| stable + composite ≥ 60 | × 1.0 |
| draft + 任意 | × 0.8 |
| review + 任意 | × 0.4 |
| deprecated | 不输出 |

最终: `final_rank = condition_match_score × quality_weight_coefficient`，输出 Top 5。

### 4.4 经验知识单元模板
```
知识单元 E{N} [调优经验] {经验ID} — {核心技术}
  适用条件: {形状/dtype/场景}
  性能提升: {boost%}
  核心优化: {1-2句描述}
  composite_score: {分数} | status: {状态}
  来源: raw/experiences/{category}/{id}.yaml
```

---

## Step 5: 调优策略知识单元 (按需)

根据 Step 4 瓶颈识别结果，从对应 techniques/ 页面提取策略。

| 瓶颈 | 技术页 | 策略 |
|------|--------|------|
| L2 bank conflict | `techniques/core_mapping.md` | 对角 tiling |
| UB 超限 | `techniques/tiling.md` | tile size 计算 |
| PipeUtil 低 | `techniques/ilp.md` | care_padding=False + for-loop |
| 多轮调度 | `techniques/persistent_kernels.md` | grid-stride loop |
| 内存带宽瓶颈 | `techniques/memory_access.md` | block_ptr / 连续访问 |
| Online 算法需求 | `techniques/online_algorithms.md` | Online Softmax |

输出格式:
```
知识单元 T{N} [调优策略] {策略名}
  适用场景: {何时需要}
  核心方法: {简要描述}
  页面: pages/techniques/{t}.md
```

---

## Step 6: 代码骨架知识单元 (按需)

| 任务类型 | 骨架文件 |
|---------|---------|
| GEMM / M,N,K≥256 | `skeletons/gemm_basic.py` |
| Flash Attention / GQA | `skeletons/flash_attention_causal.py` |
| RMSNorm / LayerNorm | `skeletons/rmsnorm_fused.py` |
| MoE Dispatch | `skeletons/moe_dispatch.py` |

输出格式:
```
知识单元 S{N} [代码骨架] {骨架名}
  适用任务: {任务类型}
  骨架路径: pages/skeletons/{name}.py
  关键结构: {最重要的 1-2 个设计点}
```

---

## 检索完成后的动作

### 1. 缓存写入
将完整知识单元列表写入 `knowledge/cache/{query_type}__{keywords}.md`，包含:
- YAML frontmatter (query_type, query_keywords, created, last_updated, hit_count, stale)
- 完整知识单元列表 (C/OP/A/B/E/T/S/X 系列)
- 末尾「复用提示」区 (列出相似算子/关键词变体)

### 2. 索引更新
更新 `knowledge/cache/README.md` 的已有缓存索引表和快速相似度匹配表。

### 3. 质量自检
- 知识单元总数 10-30 条 (少于 10 说明检索不完整，多于 40 需过滤低置信度)
- 每条知识单元有类型标签和来源
- C1-C5 基础约束必须包含
