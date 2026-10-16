
# 问题分解器与交叉关联引擎

> **用途**：将复杂/复合问题拆解为独立子问题，并分析知识单元间的关联关系，提升检索精度与知识利用率。
> **触发条件**：遇到复合问题（同时涉及多个症状/类型）或单次检索结果不足以解决问题时使用。

---

## §1 问题三维分类体系

在执行 Stage C 检索前，先对问题进行分类，确定检索优先级：

```
问题分类
  ├── [编译问题]  CompileError / SyntaxError / ImportError / API参数错误
  │     优先路由: Tier 1 (官方文档) → Tier 0 API cheatsheet → Tier 0.5 general_debug
  │     关键词提取: API名称 / 错误码 / 硬件约束关键词
  │     交叉关联: 同类API的已知限制 + hardware constraints
  │
  ├── [精度问题]  数值溢出 / NaN / cosine_sim < 0.999 / max_err 超阈值
  │     优先路由: Tier 0.5 general_stability → Tier 0 ops/<op>.md §数值稳定性
  │     关键词提取: dtype / 算子类型 / 累加维度
  │     交叉关联: 相同dtype下其他算子的精度处理方式
  │
  └── [性能问题]  加速比不达标 / 利用率低 / profiling瓶颈
        优先路由: Tier 0.5 general_perf → Tier 0 techniques → experiences/<op>
        关键词提取: bottleneck类型 / tiling配置 / 访存模式
        交叉关联: 相同瓶颈类型在其他算子上的解决方案
```

**分类细化规则**：

| 关键词/症状 | 分类 | 补充路由 |
|------------|------|---------|
| `Error:` / `Traceback` / `assert` | 编译问题 | Tier 1 `debug_guide/debugging.md` |
| `nan` / `inf` / `cosine_sim` | 精度问题 | `general_stability/` + `data_types.md` |
| `CubeUtil` / `VectorUtil` / `ms` | 性能问题 | `profiling.md` + 对应 techniques |
| 多类症状同时出现 | 复合问题 → 分解 | 本文件 §2 |

---

## §2 问题多维分解规则（Problem Decomposer）

### 分解流程

```
输入: 复合问题描述

Step D1: 识别问题维度
  → 提取所有症状关键词
  → 按三维分类标记每个症状的类型（编译/精度/性能）
  → 若只有一种类型：不需要分解，直接用 §1 路由

Step D2: 拆解独立子问题
  → 每种类型的症状生成一个子问题
  → 每个子问题独立检索（使用各自的优先路由）
  → 子问题数量通常 2-4 个，超过 4 个说明问题描述过于宽泛

Step D3: 识别子问题间依赖
  → 检查是否存在因果依赖（如：编译错误 → 精度问题不存在，需先解决编译）
  → 依赖顺序：[编译问题] 先于 [精度问题] 先于 [性能问题]

Step D4: 交叉关联分析（见 §3）
  → 对各子问题的检索结果做交叉关联
  → 合并去重，生成统一的知识单元包
```

### 分解示例

**示例 1：性能问题分解**

```
原始问题: "matmul kernel 在大矩阵时性能差，CubeUtil 只有 40%"

分解:
  子问题 P1 [性能] CubeUtil < 70% 的已知原因
    → 检索: techniques/core_mapping.md + general_perf/ + matmul/经验

  子问题 P2 [配置] 大矩阵(M,N≥1024)的最优 tiling 配置
    → 检索: techniques/tiling.md + exp_matmul_diag_tiling_001

  子问题 P3 [硬件] Ascend 910B CubeCore 调度约束
    → 检索: hardware/ascend_910b.md + hardware/ascend_patterns.md

  关联分析:
    P3 × P1 → 约束交叉验证（排除硬件限制导致的假性瓶颈）
    P2 × P1 → 解决方案匹配（哪种 tiling 能提升 CubeUtil）
```

**示例 2：精度+性能复合问题分解**

```
原始问题: "RMSNorm kernel 速度可以，但 max_err 偶尔超阈值，且 n_rows 很大时慢"

分解:
  子问题 Q1 [精度] RMSNorm fp16 精度不稳定
    → 检索: general_stability/ + normalization/ + data_types.md (fp32累加器)
    → 优先路由: Tier 0.5 general_stability

  子问题 Q2 [性能] n_rows 大时性能下降
    → 检索: persistent_kernels.md + exp_rmsnorm_001
    → 优先路由: Tier 0 techniques/persistent_kernels

  依赖分析: Q1 先解决（精度是正确性前提）→ 再优化 Q2
  
  关联分析:
    Q1 的解法（fp32累加器）可能影响 Q2 的 tile 大小设计（UB用量变化）
    → 需要 [矛盾提示] 标注
```

**示例 3：编译问题分解**

```
原始问题: "tl.dot 报 AssertionError，同时 tl.load 在某些形状下挂起"

分解:
  子问题 R1 [编译] tl.dot AssertionError
    → 检索: Tier 1 triton_api/Linear_Algebra_Ops/ + constraints_cheatsheet C3（512B对齐）

  子问题 R2 [编译] tl.load 特定形状挂起
    → 检索: Tier 1 triton_api/Memory_Pointer_Ops/ + general_debug/

  依赖分析: R1 和 R2 独立，可并行检索
  
  关联分析:
    R1 涉及 BLOCK_N 对齐 → 可能导致 R2 的 load stride 也需要调整
    → 两者关联：修改 BLOCK_N 后需同时检查 load 的 stride 计算
```

---

## §3 交叉关联引擎（Cross-Reference Engine）

> **目的**：单条知识孤立时推理能力有限，通过知识间关联可以发现隐含的解决方案。

### 四类关联类型

#### 关联类型 1：因果关联（Causal）

```
A现象 ← B原因（B导致A）

检索触发规则：
  发现现象A → 自动拉取 A 的已知原因B → 检索B的解决方案

已知因果链：
  bank_conflict 性能差        ← 内存地址非均匀分布（L2 bank collision）
    → 自动拉取: bank_conflict.md + memory_access_optimization.md

  CubeUtil低（matmul）        ← BLOCK_K 选择不当 / 对角tiling缺失
    → 自动拉取: core_mapping.md + exp_matmul_diag_tiling_001

  精度NaN（normalization）    ← fp16累加溢出
    → 自动拉取: data_types.md + exp_rmsnorm_001 §fp32_accumulator

  aiv_scalar_ratio 高         ← int64 CMP/ADD scalar fallback
    → 自动拉取: data_types.md + constraints_cheatsheet C5

  Grid调度效率低（n_rows大）  ← block数 >> 核数
    → 自动拉取: persistent_kernels.md + 对应op的 persistent 经验
```

#### 关联类型 2：冲突关联（Conflict）

```
A策略 ⊗ B约束 = 不可同时满足（需要权衡）

检索触发规则：
  检索到某优化策略 → 自动检查是否存在相冲突的硬件约束

已知冲突关联：
  增大 BLOCK_SIZE（性能优化）  ⊗  UB ≤ 192KB（硬件约束）
    → 必须标注 [矛盾提示]，给出 tile size 上限公式

  GroupNorm 融合入 GEMM       ⊗  512B 行对齐（BLOCK_N 需为256倍数）
    → GroupNorm channel_per_group 可能 ≠ 256，强制两 kernel
    → 自动标注 [矛盾提示] X 类知识单元

  增大 BLOCK_K（提高 Cube 效率）⊗  BLOCK_K=256 vs BLOCK_K=64-128 的矛盾
    → 触发已知矛盾提示（tiling.md vs performance_tuning.md）

  fp32 accumulator（精度保障）  ⊗  UB 用量翻倍（fp16→fp32）
    → tile size 需重新计算，自动关联 constraints_cheatsheet C1
```

#### 关联类型 3：互补关联（Synergy）

```
A策略 + B策略 = 叠加收益

检索触发规则：
  检索到策略A → 推荐同时检索策略B（二者常配合使用）

已知互补对：
  ILP 优化（care_padding=False）  →  推荐同时检索 persistent kernel
    （ILP 提高单次吞吐，persistent 减少调度开销，互补）

  对角 tiling（matmul）           →  推荐同时检索 dl.async_task 分离调度
    （解决 bank conflict 后，进一步提升 Cube/Vector 并行度）

  Online Softmax                  →  推荐同时检索 fp32 accumulator
    （在线算法 + 精度保障，FlashAttention 标准组合）

  block_ptr + tl.advance         →  推荐同时检索 multibuffer 编译hint
    （连续内存访问 + 多缓冲预取，内存带宽最大化）
```

#### 关联类型 4：泛化关联（Generalization）

```
A（算子特定经验）→ B（通用技术页）

检索触发规则：
  匹配到某算子专用经验 → 自动关联对应的通用技术页（补充背景理解）

已知泛化关联：
  exp_matmul_diag_tiling_001  →  techniques/core_mapping.md
    （具体经验 → 对角tiling方法论 + 参数选择规则）

  exp_rmsnorm_001（persistent）→  techniques/persistent_kernels.md
    （具体经验 → grid-stride loop 通用模式）

  exp_rope_npu_001             →  techniques/memory_access_optimization.md
    （cos/sin 复用 → 寄存器驻留通用策略）

  exp_attn_flash_001           →  techniques/online_algorithms.md
    （Flash Attention → Online Softmax 理论基础）
```

---

## §4 上下文聚合规则（Context Aggregator）

将子问题检索结果和交叉关联结果重新聚合，生成**有优先级排序的知识单元包**。

### 聚合优先级

```
P0 [强制]   [硬件约束] 类知识单元
              → 所有聚合结果必须包含，不可降级或省略

P1 [高]     [编译问题] → [API用法] + [API文档] + Tier 1 官方示例
            [精度问题] → [调试建议] + [性能约束] dtype相关
            [性能问题] → 最高分 [调优经验] + [潜在瓶颈]

P2 [中]     交叉关联发现的间接知识单元（因果/互补关联）

P3 [低]     [调优策略] 通用策略（按需输出，上下文充足时才列）
```

### 冲突解决规则

```
规则 CR1: [调优经验] 与 [硬件约束] 冲突
  → 硬件约束强制覆盖，经验建议降级
  → 必须输出 [矛盾提示] 知识单元，标注冲突详情

规则 CR2: 多条经验建议相反参数
  → 输出 [矛盾提示]，列出双方条件和来源文件
  → 不擅自选择其中一条，交由 LLM 根据当前形状判断

规则 CR3: 子问题检索结果有重叠知识单元
  → 按 P0>P1>P2>P3 保留最高优先级的版本
  → 合并前检查是否存在冲突关联

规则 CR4: 跨算子泛化知识与当前算子专用知识冲突
  → 优先采用算子专用知识（Tier 0.5 experiences 优先于 Tier 0 techniques 通用策略）
```

### 聚合输出格式

```markdown
## 问题分解检索结果

**原始问题**: <完整问题描述>
**分解子问题**: P1 / P2 / P3

---

### P0 强制知识单元（硬件约束）
<C 系列知识单元列表>

### P1 核心解决方案（按问题类型）
**[编译问题] 子问题 R1 检索结果**
<A 系列知识单元>

**[性能问题] 子问题 P1+P2 检索结果**
<B+E 系列知识单元>

### P2 交叉关联知识单元
<因果/互补关联发现的间接知识单元，标注关联类型>

### 矛盾与注意事项
<W/X 系列知识单元，含冲突解决方案>
```

---

## §5 与 Stage C 的整合

Problem Decomposer 作为 Stage C 的**前置可选步骤**：

```
Stage C 入口
    │
    ▼
Step 0: 缓存命中检查（现有）
    │ 无匹配
    ▼
[新增] 问题类型判断
    ├─ 单一类型/新算子 → 直接执行 Step 1-6（现有流程）
    └─ 复合问题        → 执行本文件 §2 分解 → 各子问题独立执行 Step 1-6
                                              → §4 聚合结果
                                              → 自动保存（现有规则）
```

**触发条件**：以下任一满足时启用问题分解：
1. 问题描述中同时出现两类以上症状关键词
2. 上一轮检索结果不足 10 条知识单元
3. 上一轮 Profiling 结果与预测瓶颈偏差超过 2 个以上维度
4. 用户在确认环节明确表示当前检索结果不完整
