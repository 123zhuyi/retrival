# raw/experiences/ — 经验库目录说明

> **用途**：存放从专家经验和实战算子生成中提取的单条经验 YAML 文件，按算子类目分目录组织。
> **关联**：`_index.yaml`（全量索引）、`retrieval/quality_scoring.md`（质量评分）、`builder/auto_ingestion.md`（自动入库）

---

## 目录结构

```
raw/experiences/
├── README.md               # 本文件：目录说明 + _index.yaml schema
├── _index.yaml             # 全量经验索引（所有经验条目汇总，LLM 检索入口）
├── matmul/                 # GEMM / 矩阵乘经验
├── attention/              # Attention / Flash Attention 经验
├── normalization/          # LayerNorm / RMSNorm / GroupNorm 经验
├── activation/             # GELU / SiLU / Swish / 激活经验
├── reduction/              # 归约经验（sum/max/min）
├── gather_scatter/         # Gather / Scatter / MoE dispatch 经验
├── rope/                   # RoPE 旋转位置编码经验
├── softmax/                # Softmax / log-softmax 经验
├── loss/                   # CrossEntropy / Label Smoothing 等 loss 经验
├── ssm/                    # SSM / Mamba / 线性注意力经验
├── conv/                   # Causal Conv1D 经验
├── embedding/              # Embedding 前后向经验
├── moe/                    # MoE 专项经验
├── misc/                   # 杂项融合经验（transformer block 等）
├── general_perf/           # 通用性能调优经验
├── general_stability/      # 通用精度稳定性经验
├── general_debug/          # 通用编译/调试经验
├── general_api/            # 通用 API 使用规范经验
└── general_ascend/         # Ascend 硬件特有经验
```

---

## _index.yaml Schema（v2）

`_index.yaml` 是所有经验的**扁平化全量索引**，LLM 在 Stage C Step 4 经验匹配时读取此文件。

### 文件头部（必须保留）

```yaml
# _index.yaml Schema Version: 2
# 字段说明:
#   experience_id    : 经验唯一标识符（命名规则见 builder/auto_ingestion.md §1）
#   category         : 算子类目（对应本目录的子目录名）
#   experience_type  : 经验描述的优化技术名称（1-4个词）
#   operator_type    : 主要算子类型（matmul/attention/normalization/...）
#   performance_boost: 声称的性能提升百分比（相对 PyTorch 基线）
#   hardware         : 适用的硬件列表
#   status           : 经验状态（draft/stable/review/deprecated），由 periodic_tasks 维护
#   composite_score  : 四维质量综合分（0-100），由 periodic_tasks 计算更新
#   usage_count      : 累计被检索采用次数（F3 high_impact + F4 accepted 时累加）
#   last_used        : 最后一次被采用的日期 YYYY-MM-DD（null 表示从未使用）
#
# 状态迁移规则: retrieval/quality_scoring.md §2
# 评分更新规则: log.md 反馈事件格式 + retrieval/quality_scoring.md §1
# 自动入库规则: builder/auto_ingestion.md §2-§3
```

### 条目格式（每条经验一个 YAML list item）

```yaml
# 最小格式（Source A 专家录入，无质量评分字段时）
- experience_id: exp_matmul_001
  category: matmul
  experience_type: 对角tiling
  operator_type: matmul
  performance_boost: 40%
  hardware: [910b2, 910b3]
  status: stable              # 新增字段（v2）
  composite_score: 62.25      # 新增字段（v2），null 表示尚未计算
  usage_count: 35             # 新增字段（v2）
  last_used: "2026-05-01"     # 新增字段（v2），null 表示从未使用

# 完整格式（Source B/C/D 自动入库，含来源追溯）
- experience_id: exp_matmul_039_20260505
  category: matmul
  experience_type: BLOCK_K=128 小矩阵优化
  operator_type: matmul
  performance_boost: 28%
  hardware: [910b3]
  status: draft
  composite_score: null
  usage_count: 0
  last_used: null
  source_type: source_b           # 来源标记（仅 Source B/C/D 有）
  source_session: matmul_512x512_fp16_run007
```

### 字段类型与约束

| 字段 | 类型 | 必填 | 约束 |
|------|------|------|------|
| `experience_id` | string | ✓ | 全局唯一，命名规则见 `auto_ingestion.md §1` |
| `category` | string | ✓ | 枚举值，与目录名完全一致 |
| `experience_type` | string | ✓ | 1-4个词，描述核心优化技术 |
| `operator_type` | string | ✓ | 枚举值：matmul/attention/normalization/... |
| `performance_boost` | string | ✓ | 格式：`<number>%`，如 `40%`；无性能提升类填 `0%` |
| `hardware` | list | ✓ | 枚举值：`[910b2]`、`[910b3]`、`[910b2, 910b3]` |
| `status` | string | ✓（v2新增） | 枚举值：draft / stable / review / deprecated；旧条目缺省时视为 `draft` |
| `composite_score` | float/null | ✓（v2新增） | 0-100 或 null（未计算）；由 `periodic_tasks` 写入，不手动填写 |
| `usage_count` | int | ✓（v2新增） | ≥ 0；由反馈事件驱动自动累加 |
| `last_used` | string/null | ✓（v2新增） | 格式 YYYY-MM-DD 或 null |
| `source_type` | string | 否 | source_b / source_c / source_d；Source A 无此字段 |
| `source_session` | string | 否 | 对应 log.md 中的 session_id |

### _index.yaml 维护规则

| 操作 | 执行时机 | 执行者 |
|------|---------|-------|
| **追加条目** | Source B/C/D 入库后 | `auto_ingestion.md §2 Step B5` |
| **更新 status/composite_score/usage_count** | 每 10 次 F3 后 | `periodic_tasks.md §2 Task 1 Step Q4` |
| **追加 conflict_with** | 冲突检测后 | `periodic_tasks.md §3 Task 2 Step CD1` |
| **标注 deprecated** | 人工操作 | 人工（Task 1 只标记 review，不自动 deprecated） |
