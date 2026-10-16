---
schema_version: okf.v1
kind: guide
type: migration_guide
source_family: reference
title: Legacy 知识单元类型与格式规范（triton-knowledge-retrieval）
description: 原 triton-knowledge-retrieval Skill 使用的 10 种知识单元类型、标准格式模板和优先级体系。已被 OKF 卡结构取代，仅作历史参考。
tags: [legacy, knowledge_unit, format, triton_knowledge_retrieval]
confidence: inferred
status: archived
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/references/knowledge-unit-format.md
created_at: '2026-08-10T00:00:00Z'
updated_at: '2026-08-10T00:00:00Z'
---

sources:
  - url: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/references/knowledge-unit-format.md
    role: primary

> legacy 规范，已被 OKF 卡结构取代，仅作历史参考。

# 知识单元类型与格式规范

> 本文档定义 Triton Ascend 知识检索 Skill 使用的 10 种知识单元类型、  
> 标准格式模板和优先级体系。

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

- **P0 (强制)**: 所有算子必须包含
- **P1 (必要)**: 直接影响正确性和性能
- **P2 (推荐)**: 提升性能，按需包含
- **P3 (辅助)**: 仅在出现具体问题时触发

## 二、标准格式模板

```
知识单元 {ID} [{类型标签}] {标题}
  {具体知识内容}
  来源: {文件路径或经验ID}
```

## 三、与 OKF 的映射关系

| 原知识单元 | OKF 对应 |
|-----------|---------|
| 硬件约束 / 性能约束 | `runbooks/` canonical 条目 |
| 调优经验 | `runbooks/` 的「已知实例」反链 |
| 调优策略 | `runbooks/` canonical 条目 |
| API 用法 / API 文档 | `reference/` 卡 |
| 代码骨架 | `reference/skeleton-*.md` |
| 算子分类 | `reference/op-class-*.md` |

---

*原文档其余内容见 primary source。*

<!-- okf:related:start -->

# 相关

- 讲解接口: [Scheduling Optimization: hardware unit isolation](../runbooks/compute-pipeline/exp_fused_norm_rope_matmul_001.md) — 同属优化知识，主题相关
- 相关主题: [Legacy 缓存读写协议（triton-knowledge-retrieval）](SPEC-Retrieve-legacy.md) — 主题相关：共享 legacy, triton_knowledge_retrieval

<!-- okf:related:end -->
