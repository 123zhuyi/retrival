# 知识单元类型与格式规范

> 本文档定义 Triton Ascend 知识检索 Skill 使用的 10 种知识单元类型、
> 标准格式模板和优先级体系。

---

## 一、知识单元类型

| 序号 | 类型标签 | 含义 | 优先级 | 生成阶段 |
|------|---------|------|--------|---------|
| 1 | `[硬件约束]` | 必须满足的硬件限制，违反会导致编译错误或运行崩溃 | **P0** | Step 1 |
| 2 | `[性能约束]` | 违反会导致严重性能退化，但不崩溃 | **P0** | Step 1 |
| 3 | `[算子分类]` | 算子类型分类结果，决定后续检索方向 | **P0** | Step 2 |
| 4 | `[API用法]` | 需要使用的 Triton/Ascend API 及其正确用法 | **P1** | Step 3 |
| 5 | `[API文档]` | API 文档所在路径，供查阅详细参数 | **P2** | Step 3 |
| 6 | `[潜在瓶颈]` | 该算子类型在此硬件上已知的性能瓶颈点 | **P1** | Step 4 |
| 7 | `[调优经验]` | 来自 experiences/ 的具体优化经验，含条件和提升幅度 | **P1** | Step 4 |
| 8 | `[调优策略]` | 来自 techniques/ 的通用优化策略 | **P2** | Step 5 |
| 9 | `[代码骨架]` | 可直接参考的完整骨架代码 | **P2** | Step 6 |
| 10 | `[矛盾提示]` / `[调试建议]` | 知识库矛盾说明 / 数值排查方法 | **P3** | 按需 |

### 优先级规则

- **P0 (强制)**: 所有算子必须包含，codegen_init 和 perf_tuning 都不可跳过
- **P1 (必要)**: 直接影响正确性和性能，codegen_init 必须包含
- **P2 (推荐)**: 提升性能，codegen_init 按需包含，perf_tuning 重点输出
- **P3 (辅助)**: 仅在出现具体问题时触发 (compile_error / precision_debug)

---

## 二、标准格式模板

### 硬件约束知识单元

```
知识单元 C{N} [硬件约束] {约束名称}
  约束: {具体限制描述}
  计算: {tile size 计算公式} (可选)
  规避: {违规时的替代方案}
  来源: pages/hardware/constraints_cheatsheet.md
```

**示例**:
```
知识单元 C1 [硬件约束] UB 总用量 ≤ 192KB
  约束: 每个 program 的 tile 数据量之和不超过 192KB (Atlas 800T A2)
  计算: fp16: 单 tile ≤ 98304 elements；fp32: ≤ 49152 elements
  规避: matmul 同时持有 A/B/C tile，实际可用约 1/3，约 64KB per matrix
  来源: pages/hardware/constraints_cheatsheet.md
```

### 性能约束知识单元

```
知识单元 C{N} [性能约束] {约束名称}
  风险: {违反时的性能退化描述}
  解决: {推荐做法}
  来源: pages/hardware/constraints_cheatsheet.md
```

### 算子分类知识单元

```
知识单元 OP [算子分类] operator_class = "{类型}"
  判断依据: {匹配的信号}
  详细页面: pages/ops/{op}.md
  经验目录: raw/experiences/{category}/
```

### API 用法知识单元

```
知识单元 A{N} [API用法] {API 名称}
  用于: {使用场景}
  Ascend 特有约束: {不同于 GPU Triton 的限制}
  推荐写法: {最佳实践代码示例}
  文档: raw/triton-ascend-develop-guide/triton_api/{分类}/
```

### 潜在瓶颈知识单元

```
知识单元 B{N} [潜在瓶颈] {瓶颈名称}
  触发条件: {何时出现}
  症状: {profiling 指标异常表现}
  影响: {性能影响量化}
  关联经验: {经验ID列表}
  修复方向: pages/techniques/{t}.md
```

### 调优经验知识单元

```
知识单元 E{N} [调优经验] {经验ID} — {核心技术}
  适用条件: {形状/dtype/场景约束}
  性能提升: {boost%}
  核心优化: {1-2句核心要点}
  composite_score: {分数} | status: {stable|draft|review}
  来源: raw/experiences/{category}/{id}.yaml
```

### 调优策略知识单元

```
知识单元 T{N} [调优策略] {策略名称}
  适用场景: {何时需要}
  核心方法: {简要描述}
  关键参数: {最重要的参数或代码片段}
  页面: pages/techniques/{t}.md
```

### 代码骨架知识单元

```
知识单元 S{N} [代码骨架] {骨架名称}
  适用任务: {任务类型}
  骨架路径: pages/skeletons/{name}.py
  关键结构: {骨架中最重要的 1-2 个设计点}
```

### 矛盾提示 / 调试建议知识单元

```
知识单元 W{N} [{类型}] {标题}
  描述: {具体问题或矛盾描述}
  建议: {推荐的处理方式}
```

---

## 三、质量要求

| 指标 | 要求 |
|------|------|
| 知识单元总数 | 10-30 条 (codegen_init)；5-15 条 (perf_tuning/compile_error/precision_debug) |
| 类型完整性 | C1-C5 必须全部出现 (codegen_init)；至少 C1, C2, C5 (perf_tuning) |
| 来源可追溯 | 每条知识单元必须标注具体来源文件路径 |
| 优先级标记 | P0/P1/P2/P3 必须显式标注或在上下文中明确 |
| 去重 | 同类型+同来源的知识单元合并为一条，不重复输出 |

## 四、质量权重参数 (v4 新增)

从 `raw/experiences/_index.yaml` 读取每条经验的元数据:

| 字段 | 含义 | 取值范围 |
|------|------|---------|
| `status` | 知识单元生命周期状态 | `stable` / `draft` / `review` / `deprecated` |
| `composite_score` | 四维综合质量分 | 0.0 ~ 1.0 |
| `frequency_score` | 被成功应用频率 (权重 0.25) | 0.0 ~ 1.0 |
| `performance_score` | 应用后性能提升比 (权重 0.40) | 0.0 ~ 1.0 |
| `precision_score` | 精度验证通过率 (权重 0.25) | 0.0 ~ 1.0 |
| `coverage_score` | 适用场景宽度 (权重 0.10) | 0.0 ~ 1.0 |

输出时按 `final_rank = condition_match_score × quality_weight_coefficient` 降序排列。
