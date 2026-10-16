---
schema_version: okf.v1
kind: guide
type: migration_guide
source_family: reference
title: Legacy 缓存读写协议（triton-knowledge-retrieval）
description: 原 triton-knowledge-retrieval Skill 的缓存系统规范，定义缓存命名、格式、命中与 stale 处理规则。已被 triton-knowledge-query / OKF 检索取代，仅作历史参考。
tags: [legacy, cache_protocol, retrieval, triton_knowledge_retrieval]
confidence: inferred
status: archived
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/references/cache-protocol.md
created_at: '2026-08-10T00:00:00Z'
updated_at: '2026-08-10T00:00:00Z'
---

sources:
  - url: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/references/cache-protocol.md
    role: primary

> legacy 规范，已被 `triton-knowledge-query` / OKF 检索取代，仅作历史参考。

# 缓存读写协议

> 本文档定义 triton-knowledge-retrieval Skill 的缓存系统规范。  
> 缓存位于 `knowledge/cache/` 目录，采用扁平化文件结构。

## 一、缓存文件命名

```
{cache_dir}/{query_type}__{keyword1}_{keyword2}_..._{keywordN}.md
```

- `query_type`: `codegen_init` | `perf_tuning` | `compile_error` | `precision_debug`
- `keywords`: 小写，下划线分隔，最多 5 个关键词
- 双下划线 `__` 分隔 query_type 和 keywords

## 二、缓存文件格式

每个缓存文件包含 YAML frontmatter 和 Markdown 内容两部分。

### 2.1 Frontmatter 必填字段

```yaml
---
query_type: codegen_init
query_keywords: [matmul, fp16, m1024, n512, k256]
created: YYYY-MM-DD
last_updated: YYYY-MM-DDTHH:MM:SS
hit_count: <被命中复用的次数>
context_summary: "<一句话描述此缓存覆盖的检索场景>"
top_units: [<最核心的知识单元ID列表，3-5条>]
stale: false | partial | true
---
```

## 三、命中规则

- 精确匹配：`query_type` 相同且 `query_keywords` 集合完全一致（忽略顺序）
- 相似匹配：`query_type` 相同且 Jaccard(keywords) ≥ 0.60
- stale 状态：`false` 正常复用；`partial` 附过期警告；`true` 跳过并完整检索

## 四、stale 处理

- `stale` 字段由定期维护任务统一更新，SKILL 不直接修改
- `hit_count` 由知识库在命中时更新
- 相似命中时创建新缓存文件，不修改已有文件

---

*原文档其余内容见 primary source。*

<!-- okf:related:start -->

# 相关

- 实现示例: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../ops/mla-prolog-v3.md) — 同属优化知识，主题相关
- 实现示例: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](../ops/nsa-compress-with-cache.md) — 同属优化知识，主题相关
- 讲解接口: [Triton-Ascend kernel 调试排查清单](../runbooks/meta-skills/debugging-checklist.md) — 同属优化知识，主题相关
- 讲解接口: [kernel 内用 tl.cdiv，host 侧用 triton.cdiv（禁止混用）](../runbooks/portability/cdiv-kernel-vs-host.md) — 同属优化知识，主题相关
- 讲解接口: [Triton API Quick Reference for Ascend Development](triton-api-cheatsheet.md) — 同属优化知识，主题相关
- 配套样例: [Triton Block Size 通用调优原则](../runbooks/compute-pipeline/traj-general_perf-block_size.md) — 同属优化知识，主题相关
- 配套样例: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](../runbooks/compute-pipeline/traj-rope-scheduling.md) — 同属优化知识，主题相关
- 配套样例: [Ascend 大 Shape 连续分块内存访问优化](../runbooks/memory-sram/traj-general_ascend-memory_access.md) — 同属优化知识，主题相关
- 配套样例: [Triton Kernel 通用语法避坑清单](../runbooks/meta-skills/traj-general_api-syntax_pitfall.md) — 同属优化知识，主题相关
- 配套样例: [Triton-Ascend 切片语法避坑：tl.arange 不是张量](../runbooks/meta-skills/traj-general_ascend-syntax_pitfall.md) — 同属优化知识，主题相关
- 配套样例: [Ascend Triton 语法规范避坑聚合](../runbooks/meta-skills/traj-general_ascend-syntax_spec.md) — 同属优化知识，主题相关
- 配套样例: [Triton-Ascend 调试排查典型问题聚合](../runbooks/meta-skills/traj-general_debug-debugging.md) — 同属优化知识，主题相关
- 相关主题: [Legacy 知识单元类型与格式规范（triton-knowledge-retrieval）](SPEC-Knowledge-Unit.md) — 主题相关：共享 legacy, triton_knowledge_retrieval

<!-- okf:related:end -->
