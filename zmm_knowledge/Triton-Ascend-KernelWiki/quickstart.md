# Quickstart

## 安装到知识库项目

```bash
cd plugins-community/triton-ascend-knowledge
bash init.sh project claude /path/to/project --knowledge-root /path/to/knowledge-base
```

支持的工具参数：`claude`、`opencode`、`trae`、`cursor`、`copilot`。未指定时默认 project + opencode + 当前目录。

默认安装 `all` profile，即完整贡献者能力。也可以按用途安装部分 skill：

```bash
# 只消费知识库：只安装 triton-knowledge-query
bash init.sh project claude /path/to/project --profile consumer --knowledge-root /path/to/knowledge-base

# 只提交/整理 Issue：只安装 triton-knowledge-issue-report
bash init.sh project claude /path/to/knowledge-base --profile issue

# 贡献知识库：安装完整知识编译、治理、检索、Issue 能力
bash init.sh project claude /path/to/knowledge-base --profile contributor

# 高级用法：精确指定 skill 子集
bash init.sh project claude /path/to/knowledge-base --skills triton-knowledge-query,triton-knowledge-issue-report
```

## 运行只读检索

```bash
python3 skills/triton-knowledge-query/scripts/knowledge_query.py discover
python3 skills/triton-knowledge-query/scripts/knowledge_query.py preflight --task "softmax 算子 UB 溢出，如何调整 BLOCK_SIZE"
python3 skills/triton-knowledge-query/scripts/knowledge_query.py search --query "BLOCK_SIZE UB 溢出"
```

## 提交前治理检查

```bash
python3 skills/triton-knowledge-query/scripts/knowledge_query.py verify
python3 skills/triton-knowledge-ingest/scripts/okf_graph.py --knowledge-root /path/to/knowledge-base verify
python3 skills/triton-knowledge-lint/scripts/knowledge_lint.py --knowledge-root /path/to/knowledge-base
```

## 知识库骨架

`examples/knowledge-base-skeleton/` 展示了知识库根目录应有的结构（`reference/`、`ops/`、6 个 `runbooks/<category>/`、`graph/`、`search/`、`log/`、`metadata/`）。初始化新知识库时可以复制该骨架。
