# cache/ — 扁平化缓存层

> **用途**：每次知识检索完成后，将检索问题、分析、路径、结果写入一条 Markdown 缓存记录。  
> **设计原则**：缓存层与具体 SKILL 完全解耦。任何 SKILL 只需按 §1 接口规范发送检索请求，知识库负责查找匹配缓存或执行完整检索后返回结果。  
> **存储结构**：所有缓存记录扁平存放在本目录根下，无子目录层级。

---

## §1 检索接口规范（SKILL 接入必读）

### 检索请求格式

SKILL 调用知识库时，发送一个**结构化检索请求**，必须包含以下字段：

```yaml
# 知识库检索请求（发送给知识库的标准格式）
query_type: codegen_init | compile_error | precision_debug | perf_tuning
query_keywords: [<关键词列表，用于缓存命名和检索匹配>]
context:
  problem: "<一句话描述本次检索要解决的具体问题>"
  trigger: "<触发本次检索的上下文事件，见下方 query_type 说明>"
  scope: "<检索范围：算子类型/错误类型/瓶颈类型，取决于 query_type>"
```

### query_type 说明

| query_type | 语义 | trigger 典型内容 | scope 典型内容 |
|-----------|------|----------------|--------------|
| `codegen_init` | 代码生成初始化 | 开始生成前，从零构建算子知识包 | 算子类型+形状+dtype，如 `matmul_fp16_M4096_N4096` |
| `compile_error` | 编译错误 | 编译失败，错误类型+错误信息 | 错误类别：`ImportError` / `SyntaxError` / `GridError` / `APIError` |
| `precision_debug` | 精度调优 | 精度验证失败，max_err+cosine_sim+症状 | 精度症状：`NaN` / `cosine_low` / `max_err_high` / `boundary` |
| `perf_tuning` | 性能调优 | 瓶颈分析完成，bottleneck 类型+测量值 | 瓶颈类型：`CubeUtil_low` / `MTE2_high` / `VecUtil_low` / `MemBW_low` |

### 检索响应格式

知识库返回一个**知识单元列表**，每条包含：

```
知识单元 <ID>  [<类型标签>]  <标题>
  <具体知识内容>
  来源: <文件路径或经验ID>
```

类型标签：`[硬件约束]` / `[性能约束]` / `[API用法]` / `[API文档]` / `[调优经验]` / `[调优策略]` / `[代码骨架]` / `[潜在瓶颈]` / `[矛盾提示]` / `[调试建议]`

---

## §2 缓存文件命名规则

缓存文件名由 `query_type` + `query_keywords` 拼接，反映检索的类型和核心内容：

```
<query_type>__<keyword1>_<keyword2>_...kw<n>.md
```

**命名示例**：

| query_type | query_keywords | 文件名 |
|-----------|---------------|--------|
| `codegen_init` | `[matmul, fp16, groupnorm, swish]` | `codegen_init__matmul_fp16_groupnorm_swish.md` |
| `codegen_init` | `[split_qkv, rmsnorm, rope, fp16]` | `codegen_init__split_qkv_rmsnorm_rope_fp16.md` |
| `compile_error` | `[ImportError, tl_dot, rank2]` | `compile_error__ImportError_tl_dot_rank2.md` |
| `precision_debug` | `[NaN, rmsnorm, fp16, accumulator]` | `precision_debug__NaN_rmsnorm_fp16_accumulator.md` |
| `perf_tuning` | `[CubeUtil_low, matmul, diagonal_tiling]` | `perf_tuning__CubeUtil_low_matmul_diagonal_tiling.md` |

**命名约束**：
- 所有字母小写，关键词间用下划线 `_` 分隔，`query_type` 与关键词间用双下划线 `__` 分隔
- 关键词选取原则：**检索类型的核心特征词**，4-6个足够
- 禁止用时间戳作为关键词（时间戳只出现在 frontmatter 中）

---

## §3 缓存文件格式（Frontmatter + 内容）

```markdown
---
query_type: codegen_init | compile_error | precision_debug | perf_tuning
query_keywords: [<关键词列表，与文件名一致>]
created: YYYY-MM-DD
last_updated: YYYY-MM-DDTHH:MM:SS
hit_count: <被命中复用的次数>
context_summary: "<一句话描述此缓存覆盖的检索场景>"
top_units: [<最核心的知识单元ID列表，3-5条>]
stale: false | partial | true
---

## 检索问题（Problem Statement）

<检索请求的 context.problem 内容>

**触发上下文**：<检索请求的 context.trigger 内容>  
**检索范围**：<检索请求的 context.scope 内容>

---

## 知识单元列表

<完整知识单元内容，格式同 §1 检索响应格式>

---

## 复用提示

<标注哪些 query_keywords 变体可以部分复用本缓存，以及需要增量更新哪些单元>
```

---

## §4 缓存命中规则

### 4.1 整体流程

知识库收到检索请求后，按以下顺序执行：

```
Step 0：stale 预检
  对所有候选文件先检查 stale 字段（见 §4.4）
  stale: true  → 跳过该文件，进入 Step 3（完整检索 + 覆写）
  stale: partial → 可用，但在响应中附带过期警告，继续 Step 1/Step 2
  stale: false  → 正常进入 Step 1/Step 2

Step 1：精确匹配
  条件: query_type 相同 AND query_keywords 集合完全一致（忽略顺序）
  操作:
    → 直接返回缓存的知识单元列表
    → 更新 frontmatter: last_updated = 当前时间, hit_count +1

Step 2：相似匹配（见 §4.2 算法）
  条件: query_type 相同 AND Jaccard(query_keywords, cached_keywords) ≥ 0.60
  操作:
    → 返回已缓存的交集部分知识单元，标注"部分命中（N/M 个关键词匹配）"
    → 对差异关键词（query_keywords - cached_keywords）执行增量检索（见 §4.3）
    → 将增量结果追加到现有知识单元列表，合并后一起返回
    → 创建新缓存文件（文件名按本次 query_keywords 命名），不覆盖已有文件

Step 3：无匹配
  条件: 无 query_type 相同的文件，或所有候选 Jaccard < 0.60
  操作:
    → 执行完整检索（按 index.md §2 对应流程）
    → 将结果写入新缓存文件（文件名按 §2 规则生成）
    → 返回完整知识单元列表
```

---

### 4.2 相似度算法（Jaccard）

使用**集合 Jaccard 相似度**衡量关键词重叠程度：

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|

其中:
  A = 检索请求的 query_keywords 集合（小写，去重）
  B = 缓存文件 frontmatter 中的 query_keywords 集合（小写，去重）

阈值: Jaccard ≥ 0.60 视为相似匹配
```

**计算示例**：

| 请求关键词 A | 缓存关键词 B | A∩B | A∪B | Jaccard | 结论 |
|---|---|---|---|---|---|
| [split_qkv, rmsnorm, rope, fp16] | [split_qkv, rmsnorm, rope, fp16] | 4 | 4 | 1.00 | 精确命中 |
| [split_qkv, rmsnorm, fp16] | [split_qkv, rmsnorm, rope, fp16] | 3 | 4 | **0.75** | 相似命中 ✓ |
| [rmsnorm, fp16] | [split_qkv, rmsnorm, rope, fp16] | 2 | 4 | **0.50** | 不命中 ✗ |
| [matmul, fp16, swish] | [matmul, fp16, groupnorm, swish] | 3 | 4 | **0.75** | 相似命中 ✓ |
| [matmul, attention, fp16] | [split_qkv, rmsnorm, rope, fp16] | 2 | 5 | **0.40** | 不命中 ✗ |

**快速判断技巧**：4 个关键词缓存文件中，≥ 3 个匹配即触发相似命中（3/4 = 0.75 ≥ 0.60）。

---

### 4.3 增量合并规则

相似命中时，对差异关键词执行增量检索后，按以下规则将增量知识单元合并到已有列表：

```
合并规则（按知识单元类型分类处理）：

【C 系列 — 硬件约束单元】
  处理: 去重（C1-C5 在所有检索中相同，直接复用，不重复输出）
  依据: 硬件约束与具体算子无关，全局唯一

【OP 系列 — 算子分类单元】
  处理: 覆盖（用本次检索的 OP 单元替换缓存中的 OP 单元）
  依据: 算子类型因请求不同而不同

【A 系列 — API 知识单元】
  处理: 合并（保留缓存中有效的 A 单元，追加新差异部分的 A 单元，按 API 名去重）
  依据: 不同算子可能共用部分 API，相同 API 只输出一次

【B 系列 — 潜在瓶颈单元】
  处理: 合并追加（将差异部分的 B 单元追加到现有 B 列表，不去重）
  依据: 不同算子组件可能有不同瓶颈，全部保留供参考

【E 系列 — 调优经验单元】
  处理: 合并 + 按 composite_score 降序排序，取 Top-5
  依据: 经验匹配结果按质量分排序，保证最优经验优先

【T 系列 — 调优策略单元】
  处理: 覆盖（用本次检索的 T 单元替换，策略需反映完整算子设计）
  依据: 调优策略是整体架构决策，不适合拼接

【S 系列 — 代码骨架单元】
  处理: 覆盖（选择与本次请求最匹配的骨架）
  依据: 骨架是完整模板，拼接会导致矛盾

【X 系列 — 矛盾提示单元】
  处理: 合并追加（保留所有矛盾提示，全部输出）
  依据: 矛盾提示越全越安全，不可丢弃
```

合并后的新知识单元列表写入新缓存文件（文件名按本次 query_keywords 生成，§2 规则）。

---

### 4.4 stale 缓存处理

**stale 状态说明**：

| stale 值 | 含义 | 命中时行为 | 覆写规则 |
|---------|------|----------|---------|
| `false` | 正常 | 直接复用 | 不覆写 |
| `partial` | 部分过期（含 review 状态经验） | 复用有效部分 + 附警告 | 不覆写，但建议重新检索 |
| `true` | 完全过期（含 deprecated 经验，或超过 90 天未命中） | 跳过，执行完整检索 | **覆写**：完整检索完成后用新结果覆盖原文件 |
| `stale_by_age` | 仅因时效过期（无 deprecated 经验） | 执行完整检索 | **覆写** |

**stale 恢复流程**（`stale: true` 时）：

```
1. 跳过该缓存文件，不使用其中的知识单元
2. 执行完整检索（index.md §2 Step 1-6）
3. 用新检索结果覆写原缓存文件（frontmatter 更新: stale=false, last_updated, hit_count 重置为1）
4. 更新 cache/README.md §5 对应行的「命中次数」
```

**stale 标记由谁写入**：`stale` 字段由 `maintenance/periodic_tasks.md` Task 3 统一写入，SKILL 不直接修改。Hit_count 更新由知识库自身在命中时修改。

---

## §5 已有缓存索引

| 文件名 | query_type | 核心关键词 | 知识单元数 | 命中次数 |
|-------|-----------|----------|---------|---------|
| [codegen_init__causal_conv1d_varlen_bfloat16_silu_conv_states.md](codegen_init__causal_conv1d_varlen_bfloat16_silu_conv_states.md) | codegen_init | causal_conv1d, varlen, bfloat16, silu, conv_states | 20 | 1 |
| [codegen_init__decode_mla_attention_paged_kv_cache_softmax_bfloat16.md](codegen_init__decode_mla_attention_paged_kv_cache_softmax_bfloat16.md) | codegen_init | decode_mla, attention, paged_kv_cache, softmax, bfloat16 | 18 | 1 |
| [codegen_init__elementwise_add_float32_float16_bfloat16_broadcasting.md](codegen_init__elementwise_add_float32_float16_bfloat16_broadcasting.md) | codegen_init | elementwise_add, float32, float16, bfloat16, broadcasting | 27 | 2 |
| [codegen_init__matmul_fp16_groupnorm_swish.md](codegen_init__matmul_fp16_groupnorm_swish.md) | codegen_init | matmul, fp16, groupnorm, swish | 24 | 1 |
| [codegen_init__split_qkv_rmsnorm_rope_bfloat16.md](codegen_init__split_qkv_rmsnorm_rope_bfloat16.md) | codegen_init | split_qkv, rmsnorm, rope, bfloat16 | 22 | 1 |
| [codegen_init__split_qkv_rmsnorm_rope_fp16.md](codegen_init__split_qkv_rmsnorm_rope_fp16.md) | codegen_init | split_qkv, rmsnorm, rope, fp16 | 22 | 1 |
| [perf_tuning__VecUtil_low_rmsnorm_rope_constexpr_load_order.md](perf_tuning__VecUtil_low_rmsnorm_rope_constexpr_load_order.md) | perf_tuning | VecUtil_low, rmsnorm_rope, constexpr, int32_offset, load_order | 23 | 1 |
| [codegen_init__kv_cache_rope_backward_elementwise_fp32_fp16_bf16.md](codegen_init__kv_cache_rope_backward_elementwise_fp32_fp16_bf16.md) | codegen_init | kv_cache, rope, backward, elementwise, fp32, fp16, bf16 | 22 | 3 |
| [codegen_init__groupnorm_fp32_fp16_normalization_elementwise_affine.md](codegen_init__groupnorm_fp32_fp16_normalization_elementwise_affine.md) | codegen_init | groupnorm, fp32, fp16, normalization, elementwise, affine | 24 | 1 |
| [codegen_init__embedding_dense_backward_fp32_fp16_bf16_scatter_add_index.md](codegen_init__embedding_dense_backward_fp32_fp16_bf16_scatter_add_index.md) | codegen_init | embedding_dense_backward, fp32, fp16, bf16, scatter_add, index | 22 | 1 |
| [codegen_init__moe_routing_init_gather_scatter_argsort.md](codegen_init__moe_routing_init_gather_scatter_argsort.md) | codegen_init | moe, routing, init, gather, scatter, argsort | 18 | 1 |
| [codegen_init__groupnorm_swish_normalization_fp16_fp32_bf16_fused.md](codegen_init__groupnorm_swish_normalization_fp16_fp32_bf16_fused.md) | codegen_init | groupnorm, swish, normalization, fp16, fp32, bf16, fused | 22 | 1 |
| [codegen_init__moe_gating_topk_softmax_fp16.md](codegen_init__moe_gating_topk_softmax_fp16.md) | codegen_init | moe, gating, topk, softmax, fp16 | 20 | 1 |
| [compile_error__ub_overflow_block_hw_groupnorm.md](compile_error__ub_overflow_block_hw_groupnorm.md) | compile_error | ub_overflow, block_hw, groupnorm, tiling | 3 | 1 |

---

## §6 快速相似度匹配

> 收到检索请求后，先查此表快速定位最相关缓存文件：

**codegen_init 类**：
```
含 rmsnorm + rope                → codegen_init__split_qkv_rmsnorm_rope_fp16.md（精确/相似）
含 split_qkv                    → codegen_init__split_qkv_rmsnorm_rope_fp16.md（部分复用 QKV 分割）
含 rmsnorm + rope + bfloat16    → codegen_init__split_qkv_rmsnorm_rope_bfloat16.md（精确命中）
含 matmul + groupnorm           → codegen_init__matmul_fp16_groupnorm_swish.md（精确/相似）
含 matmul + swish/silu          → codegen_init__matmul_fp16_groupnorm_swish.md（部分复用激活策略）
大矩阵 matmul (M,N≥512)         → codegen_init__matmul_fp16_groupnorm_swish.md（复用对角tiling单元）
含 decode_mla + attention        → codegen_init__decode_mla_attention_paged_kv_cache_softmax_bfloat16.md（精确命中）
含 paged_kv_cache + softmax      → codegen_init__decode_mla_attention_paged_kv_cache_softmax_bfloat16.md（部分复用）
含 mla + attention               → codegen_init__decode_mla_attention_paged_kv_cache_softmax_bfloat16.md（相似命中）
含 causal_conv1d + varlen        → codegen_init__causal_conv1d_varlen_bfloat16_silu_conv_states.md（精确命中）
含 causal_conv1d + conv_states   → codegen_init__causal_conv1d_varlen_bfloat16_silu_conv_states.md（部分复用）
含 conv1d + silu + bfloat16      → codegen_init__causal_conv1d_varlen_bfloat16_silu_conv_states.md（相似命中）
含 elementwise_add + float32     → codegen_init__elementwise_add_float32_float16_bfloat16_broadcasting.md（精确命中）
含 elementwise + broadcasting     → codegen_init__elementwise_add_float32_float16_bfloat16_broadcasting.md（相似命中）
含 add + float16/bfloat16        → codegen_init__elementwise_add_float32_float16_bfloat16_broadcasting.md（相似命中）
含 add + alpha + scaled          → codegen_init__elementwise_add_float32_float16_bfloat16_broadcasting.md（相似命中 + 增量补丁）
含 groupnorm + fp32/fp16         → codegen_init__groupnorm_fp32_fp16_normalization_elementwise_affine.md（精确命中）
含 groupnorm + normalization     → codegen_init__groupnorm_fp32_fp16_normalization_elementwise_affine.md（精确/相似命中）
含 layernorm + fp32/fp16         → codegen_init__groupnorm_fp32_fp16_normalization_elementwise_affine.md（相似命中，替换 OP/E 系列）
含 rmsnorm + fp32/fp16           → codegen_init__groupnorm_fp32_fp16_normalization_elementwise_affine.md（相似命中，替换 OP/E 系列）
含 embedding + backward + scatter → codegen_init__embedding_dense_backward_fp32_fp16_bf16_scatter_add_index.md（精确命中）
含 embedding_dense_backward      → codegen_init__embedding_dense_backward_fp32_fp16_bf16_scatter_add_index.md（精确命中）
含 scatter_add + index           → codegen_init__embedding_dense_backward_fp32_fp16_bf16_scatter_add_index.md（相似命中）
```

**compile_error 类**：
```
含 negative_offset + tl_load + rope → compile_error__negative_offset_tl_load_rope_static_offsets.md（精确命中）
```

**precision_debug 类**：（暂无缓存，首次遇到后自动创建）

**perf_tuning 类**：
```
VecUtil_low + rmsnorm/rope 融合    → perf_tuning__VecUtil_low_rmsnorm_rope_constexpr_load_order.md（精确命中）
VecUtil_low + constexpr 优化       → perf_tuning__VecUtil_low_rmsnorm_rope_constexpr_load_order.md（部分复用 T1/T2）
VecUtil_low + int32_offset 优化    → perf_tuning__VecUtil_low_rmsnorm_rope_constexpr_load_order.md（部分复用 T1）
含 kv_cache + rope + backward      → codegen_init__kv_cache_rope_backward_elementwise_fp32_fp16_bf16.md（精确命中）
含 rope + backward + elementwise   → codegen_init__kv_cache_rope_backward_elementwise_fp32_fp16_bf16.md（相似命中）
含 kv_cache + scatter + elementwise → codegen_init__kv_cache_rope_backward_elementwise_fp32_fp16_bf16.md（部分复用）
```

**新增 groupnorm/swish 类**：
```
含 groupnorm + swish + fused       → codegen_init__groupnorm_swish_normalization_fp16_fp32_bf16_fused.md（精确命中）
含 groupnorm + swish               → codegen_init__groupnorm_swish_normalization_fp16_fp32_bf16_fused.md（相似命中）
含 groupnorm + normalization + fp16/fp32 → codegen_init__groupnorm_fp32_fp16_normalization_elementwise_affine.md（相似命中）
含 groupnorm + activation          → codegen_init__groupnorm_swish_normalization_fp16_fp32_bf16_fused.md（部分复用）
```

**新增 moe/routing 类**：
```
含 moe + routing + gather/scatter  → codegen_init__moe_routing_init_gather_scatter_argsort.md（精确命中）
含 moe + gather + scatter          → codegen_init__moe_routing_init_gather_scatter_argsort.md（相似命中）
含 argsort + index_permutation     → codegen_init__moe_routing_init_gather_scatter_argsort.md（部分复用）
含 moe_init_routing                → codegen_init__moe_routing_init_gather_scatter_argsort.md（精确命中）
```

**新增 moe/gating/topk 类**：
```
含 moe + gating + topk + softmax   → codegen_init__moe_gating_topk_softmax_fp16.md（精确命中）
含 moe + gating + softmax          → codegen_init__moe_gating_topk_softmax_fp16.md（相似命中）
含 softmax + topk + fp16           → codegen_init__moe_gating_topk_softmax_fp16.md（相似命中）
含 moe + routing + fp16            → codegen_init__moe_gating_topk_softmax_fp16.md（部分复用 T2/E4）
```

---

## §7 SKILL 接入说明

知识库缓存层设计为**通用插件**，SKILL 接入时只需关注：

1. **发送检索请求**：按 §1 格式构造 `query_type` + `query_keywords` + `context`
2. **接收知识单元列表**：使用返回的知识单元指导代码生成/调试/优化
3. **无需关心缓存内部**：知识库自动处理命中/未命中/stale 逻辑

各 query_type 建议在 SKILL 的哪些步骤触发，见 `integration.md §2`。

---

## §8 注意事项

- 所有缓存文件扁平放在本目录，**不创建子目录**
- 缓存文件只**新建或整体覆盖**（精确命中时覆盖 frontmatter），不做行级修改
- 相似命中时**创建新文件**（含增量内容），不修改已有文件
- `stale` 字段由**定期维护任务**（`maintenance/periodic_tasks.md`）统一更新，SKILL 不直接修改
| codegen_init__gelu_activation_elementwise_float32_none_tanh.md | codegen_init | gelu, activation, elementwise, float32, none, tanh | 2026-08-06 | 1 | GELU 激活函数(精确+近似) Triton-Ascend kernel 生成 |
