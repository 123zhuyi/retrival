# 缓存读写协议

> 本文档定义 triton-knowledge-retrieval Skill 的缓存系统规范。
> 缓存位于 `knowledge/cache/` 目录，采用扁平化文件结构。

---

## 一、缓存文件命名

```
{cache_dir}/{query_type}__{keyword1}_{keyword2}_..._{keywordN}.md
```

- `query_type`: `codegen_init` | `perf_tuning` | `compile_error` | `precision_debug`
- `keywords`: 小写，下划线分隔，最多 5 个关键词
- 双下划线 `__` 分隔 query_type 和 keywords

### 命名示例

```
cache/codegen_init__matmul_fp16.md
cache/codegen_init__split_qkv_rmsnorm_rope_fp16.md
cache/codegen_init__elementwise_add_broadcast.md
cache/compile_error__tl_load_negative_offset.md
cache/precision_debug__groupnorm_fp16_overflow.md
cache/perf_tuning__matmul_cube_underutilized.md
```

---

## 二、缓存文件格式

每个缓存文件包含 YAML frontmatter 和 Markdown 内容两部分。

### 2.1 Frontmatter 必填字段

```yaml
---
query_type: codegen_init
query_keywords: [matmul, fp16, m1024, n512, k256]
created: 2026-07-10
last_updated: 2026-07-10T14:30:00
hit_count: 3
context_summary: "fp16 matmul M=1024 N=512 K=256 GEMM 算子代码生成初始化检索"
top_units: [C1, C2, C3, OP_matmul, E_matmul_001, T_diag_tiling]
stale: false
---
```

| 字段 | 必填 | 说明 |
|------|------|------|
| `query_type` | 是 | 检索类型 |
| `query_keywords` | 是 | 与文件名一致的关键词列表 |
| `created` | 是 | 首次创建日期 |
| `last_updated` | 是 | 最后更新 timestamp |
| `hit_count` | 是 | 累计命中次数 |
| `context_summary` | 是 | 一句话描述缓存覆盖的检索场景 |
| `top_units` | 是 | 最核心的 3-5 个知识单元 ID |
| `stale` | 是 | `false`(正常) / `true`(过期，跳过) / `partial`(部分过期，告知用户) |

### 2.2 内容格式

```markdown
# {query_type} 检索结果缓存

**算子**: {算子名称和形状描述}
**检索时间**: {YYYY-MM-DDTHH:MM}
**检索模式**: {query_type}
**知识单元总数**: {N}

---

## 硬件约束知识单元

知识单元 C1 [硬件约束] UB 总用量 ≤ 192KB
  ...

## 性能约束知识单元

知识单元 C5 [性能约束] int64 scalar fallback
  ...

## API 知识单元

知识单元 A1 [API用法] tl.load / tl.store
  ...

## 瓶颈预判知识单元

知识单元 B1 [潜在瓶颈] ...
  ...

## 调优经验知识单元

知识单元 E1 [调优经验] ...
  ...

## 调优策略知识单元

知识单元 T1 [调优策略] ...
  ...

## 代码骨架知识单元

知识单元 S1 [代码骨架] ...
  ...

## 复用提示

- 相似算子: {列表}
- 关键词变体: {列表}
- 部分命中的典型场景: {列表}
```

---

## 三、缓存读写协议

### 3.1 缓存读取 (Step 0)

```
输入: query_type, keywords[]
输出: 完整知识单元列表 或 部分缓存 + 增量方案

流程:
1. 构造缓存 key: "{query_type}__{'_'.join(keywords)}"
2. 在 cache/ 目录精确查找 → 命中:
   a. 读 frontmatter.stale:
      - false → 直接输出全部内容，hit_count+1，结束
      - true → 跳过，继续下一步
      - partial → 告知用户"部分可能过期"，询问是否复用
   b. 未命中精确 → 继续下一步
3. 相似匹配: 关键词重叠率 = |交集| / max(|缓存keys|, |输入keys|) ≥ 60%
   a. 找重叠率最高的缓存文件
   b. 读取该缓存
   c. 继承通用知识单元 (C/OP/A 系列)
   d. 仅对差异部分重新执行对应检索 Step
   e. hit_count+1
4. 无匹配 → 返回 None，触发完整检索
```

### 3.2 缓存写入 (检索完成后)

```
输入: query_type, keywords[], 完整知识单元列表
输出: 缓存文件已写入

流程:
1. 检查是否已存在精确匹配文件:
   - 存在 → 覆盖写入最新版本
   - 不存在 → 新建文件
2. 更新 frontmatter:
   - hit_count: 已有则 +1，新建则 =1
   - last_updated: 当前 timestamp
   - stale: false
3. 写入内容:
   - 完整知识单元列表
   - 末尾追加「复用提示」区
4. 更新 cache/README.md:
   - §5「已有缓存索引」表格新增/更新行
   - §6「快速相似度匹配」区添加/更新条目
```

---

## 四、cache/README.md 维护

### 4.1 文件结构

```markdown
# 知识检索缓存索引

## §1 缓存目录说明

## §2 缓存命名规范 (同本文档 §1)

## §3 缓存格式规范 (同本文档 §2)

## §4 缓存命中规则 (同本文档 §3.1)

## §5 已有缓存索引

| 缓存文件 | query_type | keywords | 创建时间 | 命中次数 | stale |
|---------|-----------|----------|---------|---------|-------|

## §6 快速相似度匹配

- 含 rmsnorm + rope → cache/codegen_init__rmsnorm_rope_fp16.md
- 含 matmul + swish → cache/codegen_init__matmul_fp16_groupnorm_swish.md
- ...

## §7 维护记录
```

### 4.2 更新时机

每次缓存写入后，必须更新 §5 索引表和 §6 快速匹配表。

---

## 五、缓存失效策略

### 自动失效

| 条件 | 操作 |
|------|------|
| `stale: true` | 检索时跳过，执行完整检索 |
| `stale: partial` | 告知用户部分可能过期，询问是否复用 |
| 知识库源文件更新 | Agent 手动标记相关缓存 `stale: true` |
| 缓存超过 90 天未命中 | 标记 `stale: true` (定期维护任务) |

### 手动失效

当以下情况发生时，Agent 应手动标记缓存:
- 新经验文件添加，覆盖了已缓存的算子类型
- 硬件约束文档更新 (constraints_cheatsheet.md 变更)
- API 文档重大更新
- 发现缓存中的知识单元与最新经验矛盾

---

## 六、批量操作

### 清理过期缓存

```bash
# 列出 stale: true 的缓存文件
grep -l "stale: true" knowledge/cache/*.md

# 清理 (需确认)
# rm knowledge/cache/*stale*.md  # 或删除 stale:true 的文件
```

### 重建索引

当缓存文件被手动修改/添加/删除后:
```bash
# 生成新的 cache/README.md §5 索引表
python3 scripts/rebuild_cache_index.py  # 可选辅助脚本
```
