# OKF 算子卡片模板（Ascend C 源码转化版）

> 用法：P3 阶段复制本文件为 `output/<op_name>/<op_name>.md`，逐节填写。
> 章节与详设模板一一对应（括注对应模板章节号）；正文体裁对齐 `model/design-model/` 下三份 Triton 详设实例。
> frontmatter 枚举约束：`source_family` 工作流副本可写 `golden`，**KB 入库副本必须改 `ops`**（lint 受控词表 {reference, ops, runbooks, trajectory, curated, community}，无 golden）；`confidence ∈ {inferred, verified}`（回放验证通过后才可写 verified）。
> v1.1（2026-09-04 add 经验）：总述加壳算子标注；§1 加语义分层；§9 加 benchmark 语义超集；frontmatter 补入库 lint 注记。

```markdown
---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: golden
title: <OpName>（<一句话功能>）Ascend C → Triton 语义转化卡
description: <2~3 句：算子语义、kernel 类型（AIV-only/CV 融合）、核心计算模式、对 Triton 生成的关键决策点>
tags: [<算子族>, <计算模式: elementwise|reduce|matmul|cv_fusion|...>, <dtype>, <平台特性>, <op_name>]
confidence: inferred
status: draft
resource: <源码 pinned 位置：内部路径或 GitCode @sha>
created_at: "<ISO8601>"
updated_at: "<ISO8601>"
---

sources:
  - url: <同 resource>
    role: primary

# <OpName>（<中文功能名>）

<一段话总述：算什么、为什么这样实现、kernel 类型判定（AIV-only / CV 融合，依据 KERNEL_TASK_TYPE 与是否含 Mmad）。**壳算子必标注**：kernel 本体是 N 路分派壳（锚点），调度器 X 本体不可得（⚠️ inferred），§4/§5 机制有 op_common 源码锚点（atvoss/<家族>/）或维持推断；在仓内核心语义资产 = 计算 DAG（`*_dag.h` 精度策略）+ aclnn 层参数消化策略>

## 1. 功能与公式（详设 §1.1）

- 功能：<一句话>
- 计算公式：
  $$<公式，每个变量注明含义与 shape>$$
- **语义分层**（add 经验，壳算子/多入口算子必写）：
  - GEIR kernel 本体：<公式，锚点 op_kernel/>
  - aclnn 接口层：<参数消化策略，如 alpha 四路径分解，锚点 aclnn_*.cpp>
  - benchmark 层：<torch golden 公式；与 kernel 本体的差异（超集/子集）>
- 计算流程概述：<对齐模板「算子整体计算流程图」的线性步骤描述，标明 transpose/matmul 等关键操作>
- 锚点：`op_kernel/arch35/<file>:<line>`（Compute 实现）

## 2. 原型与约束（详设 §2 / §2.1）

### 输入输出（来源：`<op>_def.cpp:<line>` / aclnn 头文件）

| 参数名 | 入参类型 | 必选性 | dtype | format | shape | 语义 |
|---|---|---|---|---|---|---|
| x | 输入 | 必选 | fp32/bf16 | ND | [B, N] | ... |

### 属性

| 属性 | 类型 | 默认值 | 语义 |
|---|---|---|---|

### 原型约束

- shape 边界：...
- 值域边界：...
- 输入间依赖：...
- 拦截校验：<有/无，位置>

## 3. 规格约束（详设 §3）

| 类别 | 取值 | 来源锚点 |
|---|---|---|
| 支持 NPU 型号 | 如 910B/910_95（ascend950 config 存在与否） | `op_host/config/` |
| dtype 全集 | | def.cpp |
| format | | def.cpp |
| 确定性计算 | 是/否 | |
| 交付形式 | aclnn / python 直调 | |

### 架构差异（arch22 vs arch35，若两套 kernel 并存）

<差异点列表：指令集、数据通路、tiling 差异>

## 4. 分核策略（详设 §4.1.1 → Triton grid 设计）

- Ascend C 事实：`usedCoreNum` 计算式、normal/tail core 划分（锚点 `*_tiling.cpp:<line>`）
- **Triton 决策**：
  - 任务轴选择：<哪个/哪些轴做任务块，为什么（输出轴优先、规约轴进循环）>
  - 任务块总量公式：`NUM_BLOCKS = <表达式>`
  - 核分配范式（elementwise 家族默认 persistent-core 连续段循环，2026-09-07 P4 实测定型；勿写「一块一 program、grid=NUM_BLOCKS」——NPU 上超额 block 串行排队）：
    ```python
    # grid = min(cdiv(numel, BLOCK), vector_core_num)，核数动态读 torch_npu，禁硬编码
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    per_prog = tl.cdiv(tl.cdiv(numel, BLOCK), num_progs)
    for blk in range(pid * per_prog, tl.minimum((pid + 1) * per_prog, tl.cdiv(numel, BLOCK))):
        <轴索引解码>   # 连续区间，禁交织；尾块 mask 统一处理
    ```
    （算子存在跨核负载不均等特殊情况时可用 grid-stride 交织 `for i in range(pid, NUM_BLOCKS, NUM_CORE)`，须在卡片注明理由）

## 5. BLOCK / Tiling 参数（详设 §4.1.2 / §4.1.3）

### Ascend C tilingdata → Triton 超参映射

| tilingdata 字段 | 语义 | host 侧计算式 | Triton 对应 | 候选值/推导式 |
|---|---|---|---|---|
| 如 tUbFactor | UB 单循环处理量 | | BLOCK_SIZE_N | {128, 256, 512}，对齐 128 |
| 尾块字段 | 尾循环大小 | | mask 边界 | `offs < N` |

### tilingkey / 模板分支 → constexpr

| tilingkey 位 / TEMPLATE_MODE | 分支语义 | Triton 处理 |
|---|---|---|
| MODE 0/1 | 如 out_flag 区分输出路径 | `OUT_FLAG: tl.constexpr` 或独立 kernel |

## 6. 片上内存预算（详设 §5）

| Ascend C buffer | 位置 | 容量表达式 | Triton 等效 tile | 字节数校核 |
|---|---|---|---|---|
| inputQue_ | VECIN | align(N*4B, 512)*buf_num | tl.load 的 [BLOCK_N] tile | |

- UB 预算合计：<表达式> vs 上限 <192KB/核>，结论：<打满/有富余/需缩 BLOCK>
- double buffer 使用：<是/否，哪些队列>

## 7. 流水编排与 Triton 可达性（详设 §6 kernel 方案）

### 流水结构

<CopyIn→Compute→CopyOut 循环结构描述；双缓冲掩盖点>

### 可达性判定表

| Ascend C 手段 | 锚点 | 可达性 | Triton 写法 / 降级方案 | 预期影响 |
|---|---|---|---|---|
| DataCopyPad | | ✅ | `tl.load(ptrs, mask=..., other=0.0)` | |
| 队列双缓冲 | | ⚠️ 编译器自动 | `multibuffer=True` 或 num_stages | |
| Reg gather/scatter | | ❌ | 离散 `tl.load/tl.store` 下标访问 | 带宽劣化，预警 |

## 8. Triton 骨架（生成用伪码）

```python
@triton.autotune(configs=[...], key=[...])
@triton.jit
def <op>_kernel(<ptrs>, <dims>, <BLOCK_*: tl.constexpr>, <开关: tl.constexpr>):
    # 1. 任务块解码（§4 范式）
    # 2. offsets 构造
    # 3. tl.load(mask=..., other=...)
    # 4. 计算（精度提升点：.to(tl.float32) 累加）
    # 5. tl.store(mask=...)
```

wrapper 要点：grid 取物理核数、输出 dtype/device 分配、contiguous 断言。

## 9. 精度标杆与验收（详设 §测试设计）

- 标杆类别：<单标杆/高精度双标杆>
- 标杆实现要点：<如 torch fp32 参考；列关键代码形态>
- **算子级验收标准**：<可执行判定，如「topk 索引集合一致」/「双标杆 L1 通过」；禁止只写通用 rtol>
- **benchmark 语义超集检查**（add 经验）：逐条核对 benchmark case 是否使用源码公式之外的参数/形态（如 alpha≠1、广播形态、大数值输入域），全部列入验收域并注明来源 case；源码未覆盖的形态写「扩展参考（非验收域）」
- 典型用例 shape：<来自 tests/ 或需求场景>
- 边界/特殊值用例：<值域限制、非对齐 shape>

## 10. Agent actions（生成指引）

1. 识别算子类别：<elementwise / reduce / CV 融合 / ...>，kernel 类型 <AIV-only / CV>
2. 切分轴取 <...>，按 §4 公式展开 grid
3. BLOCK 候选取 <...>，UB 预算见 §6
4. 精度提升点：<...>（必须保留，否则精度不达标）
5. 不可达点 <...> 用降级写法，勿在 Triton 中尝试复现 <...>
6. **勿复刻源码侧分解**（add 经验）：aclnn 多 kernel 策略分解、Contiguous/Cast 中转、schMode 分派等为解决硬件限制的手段，Triton 一律单 kernel 融合——卡片此处须列明「不要复刻」清单
7. BLOCK 初选偏大导致核数占用不足时，优先降 BLOCK（对齐 base 库 preferMultiCore 策略）；交给 autotune 时确保候选含小 BLOCK
8. 验收按 §9 标准执行

## 11. 复现性声明

本卡由 Ascend C 源码静态分析转化（confidence: inferred）。未经硬件实测的点：<列出>。
回放验证：<未做 / 结果摘要 + validation_report 路径>。
```

## 填写红线

1. 章节不可删除，无内容写「不涉及」。
2. 每个「Ascend C 事实」必须带 `file:line` 锚点；「Triton 决策」必须有推导依据（锚点或 §引用）。
3. tags 禁用通用症状词（precision_mismatch / ub_overflow / performance_bottleneck），防检索误匹配。
4. 骨架伪码必须包含 mask/边界处理，不允许只写主循环。
5. ❌ 不可达项必须同时写降级方案，不允许只标注不写替代。
6. **决策点给两态权衡，禁止无实测依据的绝对禁令**（2026-09-07 P4 经验：「勿烘焙成常量」这类错禁令会压住指令遵循度高的生成路径不去实测，比沉默更糟；属性/形态类选择写「默认 A，benchmark/X 场景可 B（收益/代价）」）。
